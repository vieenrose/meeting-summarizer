"""Put a float LoRA fine-tune of Gemma-4-E2B mobile back on Google's integer grid with GPTQ.

Rounding base + delta to the nearest integer erases the fine-tune: the delta is below half a
2/4-bit step almost everywhere (0.001 % of the integers change, val loss back to the base's).
GPTQ instead rounds one input column at a time and pushes each column's rounding error onto the
columns not yet rounded, weighted by the inverse Hessian of the layer's inputs, so that the layer's
OUTPUT matches the fine-tuned float layer. The scales, bit widths and static activation ranges stay
Google's, so the result can be written into Google's .litertlm (distill/inject_litertlm.py).

Sequential: layers are quantized in groups, and the inputs of each group are collected with the
groups before it already on their final integers.

Usage: gptq_mobile.py --lora runs/sft/mobile/lora-v1s1/epoch0/qat_lora.safetensors --out mobile-gptq
"""
import argparse
import json
import os
import random
import re
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from distill.export_mobile import pack_int  # noqa: E402
from distill.sft_agent import encode, loss_of  # noqa: E402
from distill.sft_mobile_qat import QATLoRA, srq_ste, wrap  # noqa: E402


class GPTQLinear(QATLoRA):
    """QATLoRA in float mode, plus: Hessian collection, and a fixed integer weight once quantized."""

    def setup(self):
        self.H, self.n, self.collect, self.qint = None, 0, False, None
        self.rq_scale = None          # set when the layer is re-quantized to fewer bits on a new grid

    def scale(self):
        return self.rq_scale if self.rq_scale is not None else self.base.weight_scale.float()

    def requantize(self, bits, alphas=torch.linspace(0.3, 1.0, 15)):
        """Move this layer to a `bits`-bit grid with new per-row scales, chosen to minimize the rounding MSE
        of the fine-tuned float weight (GPTQ then rounds on that grid)."""
        w = self.target() * self.base.weight_scale.float()
        qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        amax = w.abs().amax(1, keepdim=True).clamp_min(1e-12)
        best, best_s = None, None
        for a in alphas.tolist():
            s = a * amax / qmax if qmax > 0 else amax
            err = ((torch.clamp(torch.round(w / s), qmin, qmax) * s - w) ** 2).sum(1, keepdim=True)
            if best is None:
                best, best_s = err, s
            else:
                m = err < best
                best, best_s = torch.where(m, err, best), torch.where(m, s, best_s)
        self.wf = w
        self.rq_scale, self.bits, self.qmin, self.qmax = best_s, bits, qmin, qmax

    def forward(self, x):
        b = self.base
        xq = srq_ste(x, b.input_activation_scale)
        if self.collect:
            f = xq.reshape(-1, xq.shape[-1]).float()
            if self.H is None:
                self.H = torch.zeros(f.shape[1], f.shape[1], device=f.device)
            self.H.addmm_(f.T, f)
            self.n += f.shape[0]
        s = b.weight_scale.to(x.dtype)
        if self.qint is not None:
            w = self.qint.to(x.dtype) * self.scale().to(x.dtype)
        else:
            w = (self.int_weights().to(x.dtype) + (self.scaling * self.lora_B.to(x.dtype) @ self.lora_A.to(x.dtype)) / s) * s
        out = torch.nn.functional.linear(xq, w, b.bias)
        return srq_ste(out, b.output_activation_scale)

    def target(self):
        """The fine-tuned float weight, in units of the row's scale (the new scale after requantize())."""
        if self.rq_scale is not None:
            return self.wf / self.rq_scale
        s = self.base.weight_scale.float()
        return self.int_weights().float() + (self.scaling * self.lora_B.float() @ self.lora_A.float()) / s


def gptq(W, H, qmin, qmax, block=128, damp=0.01, act_order=True):
    """W: target in integer units [rows, cols] (the row scale is 1 in these units); H: X^T X [cols, cols].
    Returns integers [rows, cols]."""
    W = W.clone()
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W, H = W[:, perm], H[perm][:, perm]
    eye = torch.eye(H.shape[0], device=H.device)
    for d in (damp, damp * 10, damp * 100, 1.0):     # a near-singular Hessian: raise the damping until it factors
        try:
            Hd = H + d * torch.mean(torch.diag(H)) * eye
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True)
            break
        except torch.linalg.LinAlgError:
            print(f"  cholesky failed at damp {d}, raising it", flush=True)
    else:
        raise RuntimeError("Hessian does not factor even at damp 1.0")
    Q = torch.zeros_like(W)
    cols = W.shape[1]
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        W1 = W[:, i1:i2].clone()
        Err = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            q = torch.clamp(torch.round(w), qmin, qmax)
            Q[:, i1 + i] = q
            e = (w - q) / Hi[i, i]
            W1[:, i:] -= e.unsqueeze(1) @ Hi[i, i:].unsqueeze(0)
            Err[:, i] = e
        W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    if act_order:
        Q = Q[:, torch.argsort(perm)]
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/gemma-4-E2B-it-qat-mobile")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--rows", default="data/train/agent_sft_rows_v6.jsonl")
    ap.add_argument("--out", required=True, help="packed mobile checkpoint (for inject_litertlm.py)")
    ap.add_argument("--calib", type=int, default=96, help="training segments used as calibration")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--group", type=int, default=5, help="decoder layers per sequential group")
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--val-sessions", type=int, default=8)
    ap.add_argument("--requant-mlp-bits", type=int, default=0,
                    help="re-quantize the MLP of --requant-layers to this many bits on new per-row scales "
                         "(E4B mobile ships int4 MLP everywhere; E2B uses int2 in layers 15-34)")
    ap.add_argument("--requant-layers", default="0-999", help="inclusive layer range, e.g. 14-41")
    a = ap.parse_args()
    if not os.path.isdir(a.base):
        from huggingface_hub import snapshot_download
        a.base = snapshot_download(a.base)
    random.seed(0)
    torch.manual_seed(0)
    rows = [json.loads(l) for l in open(a.rows, encoding="utf-8")]
    sessions = sorted({r["session"] for r in rows})
    val_s = set(random.Random(0).sample(sessions, min(a.val_sessions, max(1, len(sessions) // 4))))
    tok = AutoTokenizer.from_pretrained(a.base)
    data = [(r["session"], *encode(tok, r)) for r in rows]
    train = [(i, l) for s, i, l in data if s not in val_s]
    val = [(i, l) for s, i, l in data if s in val_s]
    calib = [i[:a.max_len] for i, _ in random.Random(1).sample(train, min(a.calib, len(train)))]

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).cuda().eval()
    meta = safe_open(a.lora, "pt").metadata()
    rank, alpha = int(meta["rank"]), int(float(meta["alpha"]))
    wrapped = wrap(model, rank, alpha)
    lo = load_file(a.lora)
    for n, q in wrapped.items():
        q.__class__ = GPTQLinear
        q.setup()
        q.lora_A.data.copy_(lo[n + ".lora_A"])
        q.lora_B.data.copy_(lo[n + ".lora_B"])

    def val_loss():
        tot = n = 0
        with torch.no_grad():
            for ids, labels in val:
                l, k = loss_of(model, ids, labels)
                tot, n = tot + l.item(), n + k
        return tot / max(1, n)

    print(f"calib {len(calib)} segments, val {len(val)}; float fine-tune val loss {val_loss():.4f}", flush=True)
    layer_of = {n: int(re.search(r"layers\.(\d+)\.", n).group(1)) for n in wrapped}
    n_layers = max(layer_of.values()) + 1
    changed = total = 0
    rq_lo, rq_hi = (int(x) for x in a.requant_layers.split("-"))
    requant = set()
    for g0 in range(0, n_layers, a.group):
        group = [n for n in wrapped if g0 <= layer_of[n] < g0 + a.group]
        for n in group:
            wrapped[n].collect = True
        with torch.no_grad():
            for ids in calib:
                model(input_ids=ids.unsqueeze(0).cuda(), use_cache=False, logits_to_keep=1)
        for n in group:
            q = wrapped[n]
            q.collect = False
            if a.requant_mlp_bits and ".mlp." in n and rq_lo <= layer_of[n] <= rq_hi:
                q.requantize(a.requant_mlp_bits)
                requant.add(n)
            H = q.H / max(1, q.n)
            Q = gptq(q.target(), H, q.qmin, q.qmax, damp=a.damp)
            q.qint = Q.to(torch.int8)
            q.H = None
            if n not in requant:
                base = q.int_weights().to(torch.int8)
                changed += int((q.qint != base).sum())
                total += base.numel()
        torch.cuda.empty_cache()
        print(f"layers {g0}-{min(g0 + a.group, n_layers) - 1} done, {changed}/{max(1, total)} integers changed "
              f"({changed / max(1, total):.2%}); {len(requant)} layers re-quantized", flush=True)
    print(f"GPTQ val loss {val_loss():.4f}", flush=True)
    for q in wrapped.values():                       # nearest rounding, for comparison
        q.qint, q._g = torch.clamp(torch.round(q.target()), q.qmin, q.qmax).to(torch.int8), q.qint
    print(f"nearest-rounding val loss {val_loss():.4f}", flush=True)

    f = safe_open(os.path.join(a.base, "model.safetensors"), "pt")
    tensors = {k: f.get_tensor(k) for k in f.keys()}
    for n, q in wrapped.items():
        key = n + ".weight" if n + ".weight" in tensors else "model." + n + ".weight"
        assert key in tensors, n
        packed = pack_int(q._g.cpu(), q.bits)
        if n in requant:
            tensors[key] = packed
            tensors[key[:-len(".weight")] + ".weight_scale"] = q.rq_scale.cpu().to(tensors[key[:-len(".weight")] + ".weight_scale"].dtype)
            continue
        assert packed.shape == tensors[key].shape and packed.dtype == tensors[key].dtype, (n, packed.shape, tensors[key].shape)
        tensors[key] = packed
    os.makedirs(a.out, exist_ok=True)
    save_file(tensors, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    for x in os.listdir(a.base):
        if x != "model.safetensors" and os.path.isfile(os.path.join(a.base, x)):
            shutil.copy(os.path.join(a.base, x), a.out)
    if requant:                                      # record the new bit width ahead of the generic MLP rule
        cfg_path = os.path.join(a.out, "config.json")
        cfg = json.load(open(cfg_path))
        mq = cfg["quantization_config"]["module_quant_configs"]
        pat = r"language_model\.layers\.(" + "|".join(str(i) for i in range(rq_lo, min(rq_hi, n_layers - 1) + 1)) + r")\.mlp\."
        cfg["quantization_config"]["module_quant_configs"] = {pat: {"num_bits": a.requant_mlp_bits}, **mq}
        json.dump(cfg, open(cfg_path, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
