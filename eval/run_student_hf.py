"""Run a student model through the same notes pipeline the teacher used, on held-out sessions.

The pipeline is unchanged: same windows, same prompts, same retries, same checks. Only the chat
backend differs -- a local transformers model, optionally with a LoRA adapter -- so any gap to the
teacher's gold is the student's, not the harness's.

Windows are cut with the teacher's tokenizer, not the student's. The gold rows were produced from
those windows, so cutting differently would compare the student on different text.
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.pipeline import NotesPipeline, PipelineConfig  # noqa: E402

BASE = "google/gemma-4-E2B-it"


class HFChat:
    def __init__(self, base, adapter=None, max_new_tokens=1500):
        self.tok = AutoTokenizer.from_pretrained(base)
        self.model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda:0")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.calls, self.gen_tokens = 0, 0

    @torch.inference_mode()
    def __call__(self, messages):
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(self.model.device)
        out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)
        new = out[0][enc["input_ids"].shape[1]:]
        self.calls += 1
        self.gen_tokens += int(new.numel())
        return self.tok.decode(new, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="data/split.json")
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--window-tokenizer", default="Qwen/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    sessions = args.only or json.load(open(args.split))["heldout"]
    wtok = AutoTokenizer.from_pretrained(args.window_tokenizer)
    chat = HFChat(BASE, args.adapter)
    pipe = NotesPipeline(chat, lambda t: len(wtok.encode(t, add_special_tokens=False)), PipelineConfig())
    os.makedirs(args.out, exist_ok=True)

    for sid in sessions:
        dest = os.path.join(args.out, sid + ".json")
        if os.path.exists(dest):
            print("skip", sid); continue
        t0, c0, g0 = time.time(), chat.calls, chat.gen_tokens
        result = pipe.run(open(os.path.join(args.transcripts, sid + ".txt"), encoding="utf-8").read())
        result["elapsed_s"] = time.time() - t0
        result["calls"], result["generated_tokens"] = chat.calls - c0, chat.gen_tokens - g0
        json.dump(result, open(dest, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{sid}: {result['windows']} windows, {len(result['notes'])} notes, "
              f"{len(result['summary'])} points, {result['calls']} calls, {result['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
