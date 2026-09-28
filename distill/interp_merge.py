"""base + alpha * (SFT - base): scale a LoRA adapter's delta by alpha and merge it.

The untrained base is contradicted least on map notes but covers little; the SFT covers more and
is contradicted ~3x the gold. Scaling the fine-tuning delta (a LoRA's B matrices by alpha) is the
cheapest point between them, and interpolation toward the base is known to limit forgetting.
"""
import argparse
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="openbmb/MiniCPM5-2B")
ap.add_argument("--adapter", required=True)
ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
m = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, a.adapter)
with torch.no_grad():
    for n, p in m.named_parameters():
        if "lora_B" in n:
            p.mul_(a.alpha)
m = m.merge_and_unload()
m.save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
print("saved", a.out)
