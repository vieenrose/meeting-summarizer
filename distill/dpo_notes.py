"""DPO on the map step with synthetic hard negatives: gold notes preferred over the same notes with
one or two corrupted in the ways the student actually errs (distill/build_verify_rows.corrupt:
a figure swapped between notes, a status flipped, a figure changed).

Why DPO and not GRPO: the online judge made GRPO ~7 min a step, and the lexical rewards could not
see pairing or status errors. Here the negative is constructed, so the preference is exact and
costs nothing to label; the contrast is on the very tokens that carry the error.

The policy is a fresh LoRA on the SFT model merged into the base, so the reference (adapter off) is
the SFT model itself, as DPO assumes.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.build_verify_rows import corrupt  # noqa: E402
from summarizer.pipeline import parse_turn  # noqa: E402


def render(tag, ts, t):
    return f"- ({tag}) [{ts}] {t}" if tag else f"- [{ts}] {t}"


def pairs(rows_path, heldout, seed):
    rng = random.Random(seed)
    out = []
    for line in open(rows_path, encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] != "notes" or r["session"] in heldout:
            continue
        gold = parse_turn(r["messages"][-1]["content"]).notes
        if len(gold) < 2:
            continue
        bad = list(gold)
        hits = 0
        for i in rng.sample(range(len(gold)), min(len(gold), rng.choice([1, 1, 2]))):
            c, _ = corrupt(gold[i][2], [g[2] for j, g in enumerate(gold) if j != i], rng)
            if c and c != gold[i][2]:
                bad[i] = (gold[i][0], gold[i][1], c)
                hits += 1
        if not hits:
            continue
        out.append({"prompt": r["messages"][:-1],
                    "chosen": [{"role": "assistant", "content": "\n".join(render(*g) for g in gold)}],
                    "rejected": [{"role": "assistant", "content": "\n".join(render(*g) for g in bad)}]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-merged", default="runs/merged/runs__sft__minicpm5-v2-ivod-ali__final")
    ap.add_argument("--rows", default="data/train/train_ivod_ali.jsonl")
    ap.add_argument("--split", default="data/split_v2.json")
    ap.add_argument("--out", default="runs/dpo/minicpm5-notes-synth")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-tokens", type=int, default=5000)
    ap.add_argument("--pairs", default=None)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer
    from distill.sft_gemma import TEXT_TARGETS

    tok = AutoTokenizer.from_pretrained(args.sft_merged)
    if args.pairs:                                  # prebuilt pairs, e.g. distill/onpolicy_filter.py
        data = [{k: d[k] for k in ("prompt", "chosen", "rejected")} for d in map(json.loads, open(args.pairs, encoding="utf-8"))]
    else:
        data = pairs(args.rows, set(json.load(open(args.split))["heldout"]), 0)
    # Chosen and rejected run as one batch of two over a 130k vocabulary; a 7k-token pair OOMs a
    # 32 GB card in the backward pass. Pairs above --max-tokens are left out and counted.
    n0 = len(data)
    data = [d for d in data if len(tok.apply_chat_template(d["prompt"] + d["chosen"], tokenize=True,
                                                           return_dict=False, enable_thinking=False)) <= args.max_tokens]
    print(f"{len(data)} preference pairs ({n0 - len(data)} longer than {args.max_tokens} tokens left out)", flush=True)
    # SFT trained MiniCPM5 with thinking off (its prompt ends in an empty think block); DPOConfig has
    # no chat_template_kwargs here, so pin the flag on the tokenizer.
    import functools
    orig = tok.apply_chat_template
    tok.apply_chat_template = functools.wraps(orig)(lambda *a, **k: orig(*a, **{"enable_thinking": False, **k}))
    model = AutoModelForCausalLM.from_pretrained(args.sft_merged, dtype=torch.bfloat16, attn_implementation="sdpa")
    cfg = DPOConfig(
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=8, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_steps=10, beta=args.beta, max_length=args.max_tokens + 64,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, save_strategy="steps", save_steps=100, report_to="none",
        precompute_ref_log_probs=True,
    )
    peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TEXT_TARGETS, task_type="CAUSAL_LM")
    trainer = DPOTrainer(model=model, args=cfg, processing_class=tok, train_dataset=Dataset.from_list(data),
                         peft_config=peft)
    trainer.train()
    trainer.save_model(os.path.join(args.out, "final"))


if __name__ == "__main__":
    main()
