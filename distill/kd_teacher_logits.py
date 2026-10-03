"""Top-k next-token distributions of a teacher (base + PEFT adapter) at the loss positions of SFT rows,
for logit distillation into another student (distill/sft_mobile_qat.py --kd). Same rows, same
encoding and same loss positions as distill/sft_agent.py.

Usage: kd_teacher_logits.py --adapter runs/sft/agent/dpo-v11/policy --out runs/kd/v11_top32.pt
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from distill.sft_agent import encode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-4-E2B-it-qat-q4_0-unquantized")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--rows", default="data/train/agent_sft_rows_v6.jsonl")
    ap.add_argument("--student-tokenizer", default="unsloth/gemma-4-E2B-it-qat-mobile")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(a.base)
    stok = AutoTokenizer.from_pretrained(a.student_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).cuda()
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload().eval()
    rows = [json.loads(l) for l in open(a.rows, encoding="utf-8")]
    out = []
    with torch.no_grad():
        for n, r in enumerate(rows):
            ids, labels = encode(tok, r)
            sids, _ = encode(stok, r)
            assert torch.equal(ids, sids), "teacher and student tokenize differently"
            pos = (labels[1:] != -100).nonzero().squeeze(-1)
            logits = model(input_ids=ids.unsqueeze(0).cuda(), logits_to_keep=pos.cuda(), use_cache=False).logits[0].float()
            lp = torch.log_softmax(logits, -1)
            v, i = lp.topk(a.k, -1)
            out.append({"pos": pos.to(torch.int32), "ids": i.to(torch.int32).cpu(), "lp": v.to(torch.float16).cpu()})
            if n % 100 == 0:
                print(f"{n}/{len(rows)}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"rows": a.rows, "adapter": a.adapter, "k": a.k, "items": out}, a.out)
    print("wrote", a.out, len(out))


if __name__ == "__main__":
    main()
