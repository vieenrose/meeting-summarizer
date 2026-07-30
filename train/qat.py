"""Quantization-aware fine-tuning: simulate the *deployment* numerics during training.

Two quantizations are simulated, both with a straight-through estimator so gradients
flow to the unquantized weights/activations:

1. **KV cache — TurboQuant TQ3** (the real codec from ~/turboquant, the same one the
   litert fork runs on-device): K and V are quantize→dequantized after RoPE, i.e.
   exactly the values that get written to the packed side-cache. Applied by wrapping
   the model's attention function, so it composes with flash-attention-2 and packing.

2. **Weights — int4 block-32** fake-quant matching ai-edge-torch's
   `dynamic_int4_block32` export recipe (per-block symmetric absmax).

Training against these means the model absorbs the quantization error instead of
meeting it for the first time at export.
"""
import sys
import types

import torch

sys.path.insert(0, "/home/luigi/turboquant/turboquant")


# --------------------------------------------------------------------- weights
class _FakeQuantInt4Block(torch.autograd.Function):
    """Symmetric int4, per block of `bs` along the input dim; STE backward."""

    @staticmethod
    def forward(ctx, w: torch.Tensor, bs: int):
        out_f, in_f = w.shape[-2], w.shape[-1]
        pad = (-in_f) % bs
        if pad:
            w = torch.nn.functional.pad(w, (0, pad))
        wb = w.reshape(out_f, -1, bs)
        scale = wb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
        q = torch.clamp(torch.round(wb / scale), -8, 7)
        deq = (q * scale).reshape(out_f, -1)
        return deq[:, :in_f] if pad else deq

    @staticmethod
    def backward(ctx, g):
        return g, None


def fake_quant_weights_(model, block_size: int = 32) -> int:
    """Wrap every Linear so its weight is int4-block fake-quantized in forward."""
    n = 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.shape[-1] >= block_size:
            if getattr(mod, "_qat_wrapped", False):
                continue
            mod._qat_wrapped = True
            mod._qat_bs = block_size
            orig = mod.forward

            def fwd(self, x, _orig=orig):
                wq = _FakeQuantInt4Block.apply(self.weight, self._qat_bs)
                return torch.nn.functional.linear(x, wq, self.bias)

            mod.forward = types.MethodType(fwd, mod)
            n += 1
    return n


# -------------------------------------------------------------------- KV cache
class _TQ3RoundTrip(torch.autograd.Function):
    """quantize→dequantize with the real TurboQuant codec; STE backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, quantizer):
        shp, dt = x.shape, x.dtype
        flat = x.reshape(-1, shp[-1]).float()
        with torch.no_grad():
            deq = quantizer.dequantize(quantizer.quantize(flat))
        return deq.reshape(shp).to(dt)

    @staticmethod
    def backward(ctx, g):
        return g, None


_QUANTIZERS = {}


def _get_quantizer(dim: int, bits: int, device, dtype=torch.float32):
    key = (dim, bits, str(device))
    if key not in _QUANTIZERS:
        from turboquant.quantizer import TurboQuantMSE
        _QUANTIZERS[key] = TurboQuantMSE(dim=dim, bits=bits, device=device, dtype=dtype)
    return _QUANTIZERS[key]


def install_kv_fake_quant(model, bits: int = 3):
    """Wrap the model's *already-resolved* attention function so K and V are
    TQ3 round-tripped before attention. Must run after `from_pretrained`, because
    hub-kernel implementations (kernels-community/flash-attn2) only enter the
    registry when the model that requested them is loaded.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    impl = getattr(model.config, "_attn_implementation", "flash_attention_2")
    base = ALL_ATTENTION_FUNCTIONS.get(impl) or ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
    if getattr(base, "_kv_fake_quant", False):
        return f"{impl} (already wrapped)"

    # Wrap the entry IN PLACE and keep the implementation name: the FA2 kernel path
    # re-resolves `config._attn_implementation` at call time to lazy-import the hub
    # kernel, so a renamed entry makes it fail with "Could not find ... at kvqN".
    def attn(module, query, key, value, attention_mask, **kwargs):
        q = _get_quantizer(key.shape[-1], bits, key.device)
        key = _TQ3RoundTrip.apply(key, q)
        value = _TQ3RoundTrip.apply(value, q)
        return base(module, query, key, value, attention_mask, **kwargs)

    attn._kv_fake_quant = True
    ALL_ATTENTION_FUNCTIONS[impl] = attn
    return f"{impl} (wrapped in place)"
