"""Weighted sum of LoRA adapters trained from the same lineage ("model soup"): the merged weight is
W0 + sum_i w_i * B_i A_i exactly (PEFT's "cat" combination stacks the ranks), so v8's reading and
v9's titles, say, can be traded without retraining. The result is a LoRA adapter for
distill/merge_agent_lora.py.
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

BASE = "google/gemma-4-E2B-it-qat-q4_0-unquantized"

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=BASE)
ap.add_argument("--adapters", required=True, help="comma list of adapter dirs")
ap.add_argument("--weights", required=True, help="comma list, same length")
ap.add_argument("--out", required=True)
a = ap.parse_args()
paths, weights = a.adapters.split(","), [float(w) for w in a.weights.split(",")]
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, paths[0], adapter_name="a0")
for i, p in enumerate(paths[1:], 1):
    model.load_adapter(p, adapter_name=f"a{i}")
model.add_weighted_adapter([f"a{i}" for i in range(len(paths))], weights, "soup", combination_type="cat")
model.save_pretrained(a.out, selected_adapters=["soup"])
print("saved", a.out + "/soup")
