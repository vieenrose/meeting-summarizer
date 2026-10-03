"""Export Gemma-4-E2B mobile (optionally with a QAT LoRA from distill/sft_mobile_qat.py) without
requantization error:

  1. merge: each fine-tuned layer's integers on the base's own grid (same scale, same bits);
  2. --mobile-out: the checkpoint in Google's packed mobile format (for LiteRT-LM and transformers);
  3. --gguf-out: a llama.cpp GGUF that holds the SAME integers and scales: int2/int4 tensors as
     Q4_0 blocks (block scale = the row's scale, so every block is exact), int8 tensors as Q8_0.
     Each quantized tensor is checked against the converter's float copy of the dequantized weight.

llama.cpp computes activations in float, LiteRT-LM in int8 with static ranges; the weights are
identical, which is what a backend comparison needs.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers.integrations import gemma_quant as gq

BASE = "unsloth/gemma-4-E2B-it-qat-mobile"
REF_CONFIG = "google/gemma-4-E2B-it-qat-q4_0-unquantized"   # same architecture, no quantization_config
LLAMA = os.path.expanduser("~/llama.cpp")


def pack_int(q, bits):
    """Inverse of gemma_quant._unpack_int2 / _unpack_int4."""
    q = q.to(torch.int16)
    if bits == 4:
        u = (q + 8).to(torch.uint8)
        u = torch.nn.functional.pad(u, (0, u.shape[-1] % 2))
        return (u[..., 0::2] | (u[..., 1::2] << 4)).to(torch.uint8)
    if bits == 2:
        u = (q + 2).to(torch.uint8)
        u = torch.nn.functional.pad(u, (0, (-u.shape[-1]) % 4))
        return (u[..., 0::4] | (u[..., 1::4] << 2) | (u[..., 2::4] << 4) | (u[..., 3::4] << 6)).to(torch.uint8)
    return q.to(torch.int8)


def unpack(packed, bits, width):
    if bits == 2:
        return gq._unpack_int2(packed, width)
    if bits == 4:
        return gq._unpack_int4(packed, width)
    return packed.to(torch.int8)


def bits_of(name, cfg):
    import re
    for pat, c in cfg["quantization_config"]["module_quant_configs"].items():
        if re.search(pat, name):
            return c["num_bits"]
    return cfg["quantization_config"]["num_bits"]


def q4_0_blocks(ints, scales_per_elem):
    """ints [rows, n] in [-8, 7]; scales_per_elem [rows, n] constant within each 32-block."""
    rows, n = ints.shape
    assert n % 32 == 0
    q = (ints.astype(np.int16) + 8).astype(np.uint8).reshape(rows, n // 32, 32)
    d = scales_per_elem.reshape(rows, n // 32, 32)[..., 0].astype(np.float16)
    qs = (q[..., :16] | (q[..., 16:] << 4)).astype(np.uint8)
    out = np.empty((rows, n // 32, 18), dtype=np.uint8)
    out[..., :2] = d.view(np.uint8).reshape(rows, n // 32, 2)
    out[..., 2:] = qs
    return out.reshape(rows, -1)


def q8_0_blocks(ints, scales_per_elem):
    rows, n = ints.shape
    assert n % 32 == 0
    d = scales_per_elem.reshape(rows, n // 32, 32)[..., 0].astype(np.float16)
    out = np.empty((rows, n // 32, 34), dtype=np.uint8)
    out[..., :2] = d.view(np.uint8).reshape(rows, n // 32, 2)
    out[..., 2:] = ints.astype(np.int8).reshape(rows, n // 32, 32).view(np.uint8)
    return out.reshape(rows, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--lora", help="qat_lora.safetensors from distill/sft_mobile_qat.py")
    ap.add_argument("--mobile-out")
    ap.add_argument("--gguf-out")
    ap.add_argument("--tmp", default="/dev/shm")
    ap.add_argument("--hf-out", help="keep the dequantized float checkpoint here (for litert-torch export_hf)")
    ap.add_argument("--float-merge", action="store_true",
                    help="--hf-out holds base + LoRA delta in float, NOT rounded on Google's grid (a delta below half a "
                         "2/4-bit step is erased by that rounding); for evaluation and for re-quantization")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    snap = snapshot_download(a.base) if not os.path.isdir(a.base) else a.base
    cfg = json.load(open(os.path.join(snap, "config.json")))
    f = safe_open(os.path.join(snap, "model.safetensors"), "pt")
    tensors = {k: f.get_tensor(k) for k in f.keys()}

    float_delta = {}
    if a.lora and a.float_merge:
        lo = load_file(a.lora)
        meta = safe_open(a.lora, "pt").metadata()
        scaling = float(meta["alpha"]) / float(meta["rank"])
        for k in (x[:-len(".lora_A")] for x in lo if x.endswith(".lora_A")):
            float_delta["model." + k + ".weight" if not k.startswith("model.") else k + ".weight"] = \
                scaling * (lo[k + ".lora_B"] @ lo[k + ".lora_A"])
        print(f"float merge: {len(float_delta)} layers")
    # 1. merge the LoRA on the base's grid
    if a.lora and not a.float_merge:
        lo = load_file(a.lora)
        meta = safe_open(a.lora, "pt").metadata()
        scaling = float(meta["alpha"]) / float(meta["rank"])
        changed = total = 0
        for k in sorted(x[:-len(".lora_A")] for x in lo if x.endswith(".lora_A")):
            wkey = "model." + k + ".weight" if not k.startswith("model.") else k + ".weight"
            if wkey not in tensors:
                wkey = k.replace("model.", "", 0) + ".weight"
            bits = bits_of(wkey[:-len(".weight")], cfg)
            s = tensors[wkey[:-len(".weight")] + ".weight_scale"].float()
            width = lo[k + ".lora_A"].shape[1]
            q0 = unpack(tensors[wkey], bits, width).float()
            w = q0 + scaling * (lo[k + ".lora_B"] @ lo[k + ".lora_A"]) / s
            qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
            q = torch.clamp(torch.round(w), qmin, qmax)
            changed += int((q != q0).sum())
            total += q.numel()
            tensors[wkey] = pack_int(q, bits)
        print(f"merged LoRA: {changed}/{total} integers changed ({changed / total:.3%})")

    # 2. the mobile checkpoint
    if a.mobile_out:
        os.makedirs(a.mobile_out, exist_ok=True)
        save_file(tensors, os.path.join(a.mobile_out, "model.safetensors"), metadata={"format": "pt"})
        for x in os.listdir(snap):
            if x != "model.safetensors" and os.path.isfile(os.path.join(snap, x)):
                shutil.copy(os.path.join(snap, x), a.mobile_out)
        print("wrote", a.mobile_out)

    if not a.gguf_out and not a.hf_out:
        return
    # 3a. a dequantized text-only checkpoint for the converter (names, shapes, float tensors)
    import gguf
    hf = a.hf_out or os.path.join(a.tmp, "mobile-dequant-hf")
    os.makedirs(hf, exist_ok=True)
    ints = {}            # hf weight name -> (int array, per-element scale array, bits)
    out = {}
    for k, t in tensors.items():
        if not k.startswith(("model.language_model.", "lm_head")):
            continue
        if k.endswith(("_activation_scale", ".weight_scale", ".embedding_scale", "_cache_scale")):
            continue
        if k.endswith(".embedding_quantized"):
            base = k[:-len(".embedding_quantized")]
            bits = bits_of(base, cfg)
            sc = tensors[base + ".embedding_scale"].float()
            width = cfg["text_config"]["hidden_size"] if base.endswith("embed_tokens") else \
                cfg["text_config"]["hidden_size_per_layer_input"] * cfg["text_config"]["num_hidden_layers"]
            q = unpack(t, bits, width)
            se = sc.repeat_interleave(width // sc.shape[1], dim=1)
            ints[base + ".weight"] = (q.numpy(), se.numpy(), bits)
            out[base + ".weight"] = (q.float() * se).to(torch.bfloat16)
        elif t.dtype == torch.uint8 or (t.dtype == torch.int8 and k + "_scale" in tensors):
            base = k[:-len(".weight")]
            bits = bits_of(base, cfg)
            sc = tensors[base + ".weight_scale"].float()
            width = {2: t.shape[1] * 4, 4: t.shape[1] * 2, 8: t.shape[1]}[bits]
            q = unpack(t, bits, width)
            se = sc.expand(-1, width)
            ints[k] = (q.numpy(), se.numpy(), bits)
            out[k] = (q.float() * se + float_delta.get(k, 0)).to(torch.bfloat16)
        else:
            out[k] = t
    save_file(out, os.path.join(hf, "model.safetensors"), metadata={"format": "pt"})
    ref = snapshot_download(REF_CONFIG, allow_patterns=["*.json", "*.jinja", "tokenizer*"])
    for x in os.listdir(ref):
        if os.path.isfile(os.path.join(ref, x)) and not x.startswith("model"):
            shutil.copy(os.path.join(ref, x), hf)
    if not a.gguf_out:
        print("wrote", hf)
        return
    f16 = os.path.join(a.tmp, "mobile-dequant-f16.gguf")
    subprocess.run([sys.executable, f"{LLAMA}/convert_hf_to_gguf.py", hf, "--outtype", "f16", "--outfile", f16], check=True)

    # 3b. rewrite the quantized tensors with exact blocks
    r = gguf.GGUFReader(f16)
    n_layers = cfg["text_config"]["num_hidden_layers"]
    tmap = gguf.get_tensor_name_map(gguf.MODEL_ARCH.GEMMA4, n_layers)
    g2h = {}
    for k in ints:
        for cand in (k, k.replace("model.language_model.", "model."), k.replace("model.language_model.", "")):
            g = tmap.get_name(cand, try_suffixes=(".weight",))
            if g:
                g2h[g + ".weight"] = k
                g2h[g] = k
                break
    w = gguf.GGUFWriter(a.gguf_out, arch="gemma4")
    for field in r.fields.values():
        if field.name.startswith("GGUF.") or field.name == "general.architecture":
            continue
        vt = field.types[0]
        if vt == gguf.GGUFValueType.ARRAY:
            sub = field.types[-1]
            vals = [field.parts[i].tolist() if sub != gguf.GGUFValueType.STRING else bytes(field.parts[i]).decode("utf-8", "replace")
                    for i in field.data]
            vals = [v[0] if isinstance(v, list) and len(v) == 1 else v for v in vals]
            w.add_array(field.name, vals)
        elif vt == gguf.GGUFValueType.STRING:
            w.add_string(field.name, bytes(field.parts[field.data[0]]).decode("utf-8", "replace"))
        else:
            w.add_key_value(field.name, field.parts[field.data[0]][0], vt)
    exact = 0
    for t in r.tensors:
        hfk = g2h.get(t.name)
        shape = [int(x) for x in reversed(t.shape)]          # numpy order (rows, cols)
        if hfk is None:
            w.add_tensor(t.name, np.array(t.data).reshape(shape), raw_dtype=t.tensor_type)
            continue
        q, se, bits = ints[hfk]
        q, se = q.reshape(shape), se.reshape(shape)
        conv = np.array(t.data, dtype=np.float32).reshape(shape)
        err = np.abs(conv - q * se).max() / (np.abs(q * se).max() + 1e-12)
        assert err < 1e-2, (t.name, hfk, err)   # same tensor, same layout as the converter's
        if bits in (2, 4):
            blk, qt = q4_0_blocks(q, se), gguf.GGMLQuantizationType.Q4_0
        else:
            blk, qt = q8_0_blocks(q, se), gguf.GGMLQuantizationType.Q8_0
        w.add_tensor(t.name, blk, raw_shape=blk.shape, raw_dtype=qt)
        exact += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.remove(f16)
    if not a.hf_out:
        shutil.rmtree(hf)
    print(f"wrote {a.gguf_out}: {exact} tensors as exact Q4_0/Q8_0")


if __name__ == "__main__":
    main()
