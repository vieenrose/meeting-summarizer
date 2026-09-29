"""Merge an agent LoRA (distill/sft_agent.py) into the Gemma-4-E2B QAT weights and produce the
Q4_0 GGUF the phone runs: merge -> save HF checkpoint -> convert_hf_to_gguf (bf16) -> llama-quantize.
The base alone through the same path gives a Q4_0 of the same size as Google's QAT GGUF."""
import argparse
import glob
import os
import shutil
import subprocess

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "google/gemma-4-E2B-it-qat-q4_0-unquantized"
LLAMA = os.path.expanduser("~/llama.cpp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True, help="output .gguf (Q4_0)")
    ap.add_argument("--base", default=BASE)
    a = ap.parse_args()
    hf_dir = a.out.replace(".gguf", "-hf")
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.save_pretrained(hf_dir)
    AutoTokenizer.from_pretrained(a.base).save_pretrained(hf_dir)
    snap = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{a.base.replace('/', '--')}/snapshots/*/"))[0]
    for f in ("chat_template.jinja", "processor_config.json", "generation_config.json"):
        if os.path.exists(os.path.join(snap, f)):
            shutil.copy(os.path.join(snap, f), hf_dir)
    bf16 = a.out.replace(".gguf", "-bf16.gguf")
    subprocess.run([os.path.expanduser("~/.venvs/vllm/bin/python"), f"{LLAMA}/convert_hf_to_gguf.py", hf_dir, "--outtype", "bf16", "--outfile", bf16], check=True)
    subprocess.run([f"{LLAMA}/build-cuda/bin/llama-quantize", bf16, a.out, "Q4_0"], check=True)
    os.remove(bf16)
    shutil.rmtree(hf_dir)
    print("wrote", a.out, os.path.getsize(a.out))


if __name__ == "__main__":
    main()
