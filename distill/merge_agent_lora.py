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
    ap.add_argument("--tmp", default="/dev/shm", help="where the merged HF checkpoint and the f16 GGUF are staged")
    ap.add_argument("--keep-f16", action="store_true", help="keep the f16 GGUF (to measure the Q4_0 loss)")
    a = ap.parse_args()
    stem = (os.path.basename(os.path.dirname(os.path.abspath(a.out))) + "-" + os.path.basename(a.out)).replace(".gguf", "")
    hf_dir = os.path.join(a.tmp, stem + "-hf")
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.save_pretrained(hf_dir)
    AutoTokenizer.from_pretrained(a.base, trust_remote_code=True).save_pretrained(hf_dir)
    snap = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{a.base.replace('/', '--')}/snapshots/*/"))[0]
    # the base's own config: save_pretrained can rewrite fields the GGUF converter reads (Granite 4
    # loses layer_types and is then converted as a hybrid Mamba model)
    import json
    base_cfg = json.load(open(os.path.join(snap, "config.json")))
    same_arch = base_cfg.get("architectures") == json.load(open(os.path.join(hf_dir, "config.json"))).get("architectures")
    # a text-only save (Qwen3.5) drops the MTP head while the config still announces it: convert without it
    cfg = json.load(open(os.path.join(hf_dir, "config.json")))
    has_mtp = any((c or {}).get("mtp_num_hidden_layers") for c in (cfg, cfg.get("text_config")))
    for f in (["config.json"] if same_arch else []) + ["chat_template.jinja", "processor_config.json", "generation_config.json",
              "tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:   # the base's own
              # tokenizer files: a re-saved fast tokenizer loses tokenizer.model (Gemma 3 then fails to convert)
        if os.path.exists(os.path.join(snap, f)):
            shutil.copy(os.path.join(snap, f), hf_dir)
    # f16, not bf16: the one tensor quantize keeps unquantized (per_layer_model_proj) must be f16, as
    # in Google's QAT GGUF -- the phone CPU (ARMv8.2) has no bf16 and runs prefill ~13 % slower.
    bf16 = os.path.join(a.tmp, stem + "-f16.gguf")
    subprocess.run([os.path.expanduser("~/.venvs/vllm/bin/python"), f"{LLAMA}/convert_hf_to_gguf.py", hf_dir, "--outtype", "f16", "--outfile", bf16]
                   + (["--no-mtp"] if has_mtp and not same_arch else []), check=True)
    subprocess.run([f"{LLAMA}/build-cuda/bin/llama-quantize", bf16, a.out, "Q4_0"], check=True)
    if a.keep_f16:
        shutil.move(bf16, a.out.replace(".gguf", "-f16.gguf"))
    else:
        os.remove(bf16)
    shutil.rmtree(hf_dir)
    print("wrote", a.out, os.path.getsize(a.out))


if __name__ == "__main__":
    main()
