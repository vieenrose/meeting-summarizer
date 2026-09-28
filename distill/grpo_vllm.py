"""GRPO on the summary step with vLLM rollouts (TRL 1.13, vLLM colocate).

Runs in ~/vllm019: vLLM 0.19.1, transformers 5.17, TRL 1.13. Same reward and data as
distill/grpo_synth.py, which generates with transformers and managed ~4 samples per ~30 s step;
vLLM generates ~1,400 tokens/s over a batch, so each prompt gets many more samples per step.

Two fixes are needed for vLLM to run Gemma-4-E2B at all, both applied here without touching
installed packages:

  base checkpoint   google/gemma-4-E2B-it. The QAT release drops k_proj/v_proj/k_norm for the
                    20 KV-shared layers, and vLLM's loader requires them.
  config            transformers 5.17 folds global_head_dim and num_kv_shared_layers into per-layer
                    configs, where vLLM cannot read them; it then builds 256-wide attention for
                    512-wide layers. The scalar text_config values are copied back from
                    config.json via hf_overrides. Verified: greedy output matched transformers
                    token for token over 200 tokens.
"""
import argparse
import glob
import json
import os
import sys

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE = "google/gemma-4-E2B-it"


class Gemma4Overrides:
    """hf_overrides callable restoring the scalar text_config values transformers 5.17 hides.

    A module-level class rather than a closure: when CUDA is already initialised in the calling
    process (e.g. after merging an adapter), vLLM starts its engine by spawning, which pickles
    hf_overrides, and a local function cannot be pickled.
    """

    def __init__(self, model_id):
        snap = glob.glob(os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots/*/config.json"))[0]
        self.raw = json.load(open(snap))["text_config"]

    def __call__(self, cfg):
        text = getattr(cfg, "text_config", None)
        for c in (cfg, text):
            if c is not None:
                c.allow_global_per_layer_attribute_access = True
        if text is not None:
            for k, v in self.raw.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    try:
                        setattr(text, k, v)
                    except Exception:
                        pass
        return cfg


def gemma4_overrides(model_id):
    """None for anything but a Gemma-4 checkpoint: the shim reads `text_config`, which only the
    heterogeneous per-layer Gemma-4 config carries, so applying it to e.g. Qwen3-32B raised
    KeyError before the engine ever started. vLLM accepts hf_overrides=None."""
    try:
        snap = glob.glob(os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots/*/config.json"))[0]
        if "text_config" not in json.load(open(snap)):
            return None
    except (IndexError, OSError, json.JSONDecodeError):
        return None          # local path or unreadable config: nothing to override
    return Gemma4Overrides(model_id)


def patch_trl_vllm(model_id):
    """TRL builds its colocated LLM itself, with no hook for hf_overrides; wrap its reference."""
    import trl.generation.vllm_generation as vg
    original = vg.LLM
    fix = gemma4_overrides(model_id)

    def llm_with_overrides(*args, **kwargs):
        kwargs.setdefault("hf_overrides", fix)
        if fix is not None:                          # multimodal Gemma only
            kwargs.setdefault("limit_mm_per_prompt", {"image": 0, "audio": 0})
        return original(*args, **kwargs)

    vg.LLM = llm_with_overrides


def main():
    from distill.grpo_common import build, build_notes, make_judged_notes_reward, make_notes_reward, make_reward

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["synth", "notes"], default="synth")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--judge", action="store_true",
                    help="notes task: add the window-judge faithfulness reward (distill/window_judge.py)")
    ap.add_argument("--adapter", default=None, help="LoRA to start from")
    ap.add_argument("--out", default="runs/grpo/g4it-vllm")
    ap.add_argument("--rows", default="data/train/gold_rows.jsonl")
    ap.add_argument("--split", default="data/split.json")
    ap.add_argument("--gold", default="runs/gold/w4000")
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=20)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--vllm-mem", type=float, default=0.35)
    args = ap.parse_args()

    patch_trl_vllm(args.base)
    from trl import GRPOConfig, GRPOTrainer

    if args.task == "synth":
        data = build(args.rows, args.split, args.gold, args.transcripts)
        reward = make_reward(args.gold, args.transcripts)
    else:
        data = build_notes(args.rows, args.split)
        reward = make_judged_notes_reward() if args.judge else make_notes_reward()
    print(f"{len(data)} {args.task} prompts from training sessions", flush=True)

    gemma = "gemma" in args.base.lower()
    tok = AutoTokenizer.from_pretrained(args.base)
    # The turn terminator, not the tokenizer's eos: Gemma ends a turn with <turn|>, MiniCPM5 with
    # <|im_end|>. GRPO stops and masks at eos_token_id, so the wrong one runs every sample to max length.
    tok.eos_token = "<turn|>" if gemma else "<|im_end|>"
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.warnings_issued = {}
    peft_config = None
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        # Without an adapter every one of the 5B parameters would be trained, and Adam's state alone
        # does not fit. Start a fresh LoRA on the language model instead, as SFT does.
        from peft import LoraConfig
        from distill.sft_gemma import TARGETS, TEXT_TARGETS
        peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                 target_modules=TARGETS if gemma else TEXT_TARGETS,
                                 task_type="CAUSAL_LM")

    cfg = GRPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        # vLLM generates the whole group in one batch, but the training forward pass materialises
        # completion logits over a 262k vocabulary: 8 x 800 tokens of them does not fit beside
        # vLLM on one card. Forward two samples at a time; accumulation keeps the same
        # num_generations x prompts_per_step samples per update.
        per_device_train_batch_size=2,
        gradient_accumulation_steps=args.num_generations * args.prompts_per_step // 2,
        num_generations=args.num_generations,
        max_completion_length=800 if args.task == "synth" else 1024,
        temperature=0.9,
        top_p=0.95,
        beta=0.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_mem,
        # Release vLLM's weights and KV cache during the backward pass; with them held, a 5k-token
        # prompt's activations do not fit beside training on one 32 GB card.
        vllm_enable_sleep_mode=True,
        vllm_max_model_length=8192,
        # MiniCPM5 thinks by default; SFT trained it with thinking off. Gemma ignores the flag.
        chat_template_kwargs={"enable_thinking": False},
        logging_steps=1,
        save_strategy="steps",
        save_steps=20 if args.task == "synth" else 25,
        save_total_limit=None,          # checkpoints are scored after the fact; never delete them
        report_to="none",
    )
    trainer = GRPOTrainer(model=model, args=cfg, processing_class=tok,
                          reward_funcs=[reward],
                          train_dataset=Dataset.from_list(data), peft_config=peft_config)
    trainer.train()
    trainer.save_model(os.path.join(args.out, "final"))


if __name__ == "__main__":
    main()
