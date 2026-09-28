"""GRPO on the summary step, with the rule-based reward in distill/rewards.py.

SFT taught the student to take notes but not to summarise them: it saw 29 summaries, and at
evaluation it copied the prompt's own rules back or wrote one run-on point. Supervised learning on
so few examples can only imitate them. GRPO instead samples several summaries for the same notes
and pushes toward the ones the checker scores higher, so the student learns from its own attempts
against rules it can be scored on -- real citations, coverage of the whole meeting, the gold's key
facts, nothing the transcript does not say.

Training starts from an SFT adapter, which is itself trained further. No reference model is kept
(beta=0): the reward already forbids the degenerate outputs a KL term would guard against, and a
second 5B copy would not fit beside generation on one card.
"""
import argparse
import json
import os
import sys

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# TRL 0.24 decides optional packages are available from installed metadata, and on this host that
# metadata lies: vllm_ascend (a Huawei NPU build) and mergekit are "available" but not importable,
# and loading the GRPO trainer imports them. Re-derive every such flag from whether the module can
# actually be found, before any trainer module is imported.
import importlib.util  # noqa: E402

import trl.import_utils as _trl_imports  # noqa: E402

for _name in [n for n in dir(_trl_imports) if n.startswith("_") and n.endswith("_available")]:
    _module = _name[1:-len("_available")]
    if getattr(_trl_imports, _name) and importlib.util.find_spec(_module) is None:
        setattr(_trl_imports, _name, False)
# vLLM itself is real here (0.19, built for this host) but newer than TRL 0.24 expects -- its
# GuidedDecodingParams is gone. Generation runs in transformers, so hide vLLM from TRL entirely.
_trl_imports._vllm_available = False
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.grpo_common import build, make_reward  # noqa: E402

BASE = "google/gemma-4-E2B-it-qat-q4_0-unquantized"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="SFT LoRA to start from")
    ap.add_argument("--out", default="runs/grpo/gemma4-synth")
    ap.add_argument("--rows", default="data/train/gold_rows.jsonl")
    ap.add_argument("--split", default="data/split.json")
    ap.add_argument("--gold", default="runs/gold/w4000")
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    data = build(args.rows, args.split, args.gold, args.transcripts)
    print(f"{len(data)} synthesis prompts from training sessions")

    tok = AutoTokenizer.from_pretrained(BASE)
    # Gemma ends a chat turn with <turn|>, not <eos>. TRL stops generation and masks the completion
    # at tokenizer.eos_token_id, so left at <eos> every sample ran to the length cap and scored ~0
    # -- no stop, no signal. Plain generate() never showed this because generation_config lists
    # both ids; TRL uses the tokenizer's single id.
    tok.eos_token = "<turn|>"
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa")
    # TRL 0.24 records a flag in model.warnings_issued, a dict transformers 5 no longer creates.
    model.warnings_issued = {}
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)

    cfg = GRPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=2,
        num_generations=args.num_generations,
        # GRPOConfig defaults max_prompt_length to 512 and left-truncates. A synthesis prompt is 3-5k
        # tokens of notes, so the model saw only the prompt's closing instructions and "summarised"
        # those -- every sample then failed the echo check and scored 0. Keep the whole prompt.
        max_prompt_length=6144,
        max_completion_length=800,
        temperature=0.9,
        top_p=0.95,
        beta=0.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=4,
        report_to="none",
        log_completions=False,
    )
    trainer = GRPOTrainer(model=model, args=cfg, processing_class=tok,
                          reward_funcs=[make_reward(args.gold, args.transcripts)],
                          train_dataset=Dataset.from_list(data))
    trainer.train()
    trainer.save_model(os.path.join(args.out, "final"))


if __name__ == "__main__":
    main()
