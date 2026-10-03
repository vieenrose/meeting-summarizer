"""Write fine-tuned Gemma-4-E2B mobile weights into Google's own .litertlm, keeping its graph,
kernels, activation ranges and speed. The integers of Google's prefill/decode .tflite are the
same as those of the HF mobile checkpoint (same per-channel scales, same int2/int4 packing; checked
bit for bit), so a model fine-tuned on that grid (distill/sft_mobile_qat.py, then
distill/export_mobile.py --mobile-out) only changes the integer bytes, in place, at equal size.

Usage: inject_litertlm.py --unpacked <litert-lm unpack dir> --mobile <fine-tuned mobile dir> --out x.litertlm
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

import numpy as np
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from distill.tflite_ints import TFL, TT, fb, pack, unpack  # noqa: E402

ROLE = {"q_einsum": "self_attn.q_proj", "k_einsum": "self_attn.k_proj", "v_einsum": "self_attn.v_proj",
        "attn_vec_einsum": "self_attn.o_proj", "gating_einsum1": "mlp.gate_proj", "gating_einsum2": "mlp.up_proj",
        "mlp/linear": "mlp.down_proj"}


def hf_name(tfl_name):
    m = re.search(r"layer_(\d+)/", tfl_name)
    if not m:
        return None
    for k, v in ROLE.items():
        if k in tfl_name:
            return f"model.language_model.layers.{m.group(1)}.{v}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unpacked", default=os.environ.get("LITERT_UNPACKED"), required="LITERT_UNPACKED" not in os.environ, help="`litert-lm unpack` of litert-community/gemma-4-E2B-it-litert-lm")
    ap.add_argument("--mobile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--litert-lm", default="litert-lm")
    ap.add_argument("--activation-type", choices=["fp16", "fp32"], default="fp16",
                    help="prefer_activation_type of the prefill_decode section (GPU); Google ships fp16")
    ap.add_argument("--no-prefill-1024", action="store_true",
                    help="disable the prefill_1024 signature (renamed in place): the runtime then prefills in blocks "
                         "of 128, whose attention buffers are 8x smaller; peak memory at a 4k context 1.8 -> 1.1 GB")
    a = ap.parse_args()
    from transformers.integrations import gemma_quant as gq
    src = os.path.join(a.unpacked, "Section10_TFLiteModel_tf_lite_prefill_decode.tflite")
    T = TFL(src)
    f = safe_open(os.path.join(a.mobile, "model.safetensors"), "pt")
    data = bytearray(T.buf)
    import json
    import struct
    cfg = json.load(open(os.path.join(a.mobile, "config.json")))
    rules = list(cfg["quantization_config"]["module_quant_configs"].items())

    def hf_bits(hf):
        for pat, c in rules:
            if re.search(pat, hf):
                return c["num_bits"]
        return cfg["quantization_config"]["num_bits"]

    # every tensor that reads a buffer, across signatures (prefill, decode, verify share the weights)
    users = {}
    for si in range(T.m.SubgraphsLength()):
        sg = T.m.Subgraphs(si)
        for ti in range(sg.TensorsLength()):
            users.setdefault(sg.Tensors(ti).Buffer(), []).append(sg.Tensors(ti))
    patched = changed = total = retyped = 0
    for t in T.tensors():
        name = t.Name().decode()
        hf = hf_name(name)
        if hf is None:
            continue
        typ = TT[t.Type()]
        sh = t.ShapeAsNumpy().tolist()
        bits = {"INT2": 2, "INT4": 4, "INT8": 8}[typ]
        nbits = hf_bits(hf)
        w = f.get_tensor(hf + ".weight")
        q = (gq._unpack_int2(w, sh[1]) if nbits == 2 else gq._unpack_int4(w, sh[1]) if nbits == 4 else w).numpy()
        old = unpack(T.raw(t), typ)[:sh[0] * sh[1]].reshape(sh)
        assert q.shape == old.shape, (name, q.shape, old.shape)
        hs = f.get_tensor(hf + ".weight_scale").numpy().ravel().astype(np.float32)
        off, n = T.span(t)
        if nbits != bits:
            # re-quantized to fewer bits (distill/gptq_mobile.py --requant-mlp-bits): write the new type,
            # the new scales and the shorter payload in place; the bytes after it are never read
            assert nbits < bits and nbits == 2, (name, bits, nbits)
            ntyp = "INT2"
            new = pack(q, ntyp)
            bb = T.m.Buffers(t.Buffer())
            o = bb._tab.Offset(8)
            assert bb.Offset() and o, ("buffer size field missing", name)
            struct.pack_into("<Q", data, bb._tab.Pos + o, len(new))
            for u in users[t.Buffer()]:
                data[u._tab.Pos + u._tab.Offset(6)] = fb.TensorType.INT2
                qp = u.Quantization()
                vo = qp._tab.Offset(8)
                vec = qp._tab.Vector(vo)
                assert qp._tab.VectorLen(vo) == hs.size, ("scale count", name)
                data[vec:vec + 4 * hs.size] = hs.tobytes()
            retyped += 1
        else:
            sc = t.Quantization().ScaleAsNumpy()
            assert np.allclose(sc, hs, rtol=1e-6), ("scales differ", name)
            new = pack(q, typ)
            changed += int((q != old).sum())
            total += q.size
        assert len(new) <= n
        data[off:off + len(new)] = new.tobytes()
        patched += 1
    if retyped:
        print(f"re-typed {retyped} weight tensors to INT2")
    print(f"patched {patched} weight tensors, {changed}/{total} integers changed ({changed / max(1, total):.3%})")
    work = a.out + ".dir"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(a.unpacked, work)
    if a.no_prefill_1024:
        n = 0
        for m in list(re.finditer(rb"prefill_1024", data)):
            data[m.start():m.end()] = b"prefilX_1024"
            n += 1
        print(f"prefill_1024 disabled ({n} names)")
    open(os.path.join(work, os.path.basename(src)), "wb").write(bytes(data))
    toml = os.path.join(work, "model.toml")
    t = open(toml).read()
    head, sep, tail = t.partition('model_type = "prefill_decode"')
    sec = head.rfind("[[section]]")
    m = re.search(r'value = "fp(16|32)"', head[sec:])
    assert sep and m, "prefill_decode section has no prefer_activation_type"
    i, j = sec + m.start(), sec + m.end()
    open(toml, "w").write(head[:i] + f'value = "{a.activation_type}"' + head[j:] + sep + tail)
    subprocess.run([a.litert_lm, "pack", os.path.join(work, "model.toml"), "--output", a.out], check=False)
    if not os.path.exists(a.out):
        subprocess.run([a.litert_lm, "pack", work, a.out], check=True)
    shutil.rmtree(work)
    print("wrote", a.out, os.path.getsize(a.out))


if __name__ == "__main__":
    main()
