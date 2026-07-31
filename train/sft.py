"""Full-parameter SFT of Qwen3-0.6B on the VoxSum task suite.

Launch on both GPUs (halve grad_accum so the effective batch — and therefore the LR
schedule and step count — matches the single-GPU recipe):
  uv run torchrun --nproc_per_node=2 train/sft.py --config configs/sft_ddp.json
Single GPU:
  CUDA_VISIBLE_DEVICES=1 uv run python train/sft.py

Targets are teacher completions; loss on assistant tokens only. The chat template is
applied with enable_thinking=False so the assistant turn carries Qwen3's empty
<think></think> prefix — matching what the .litertlm bundle's embedded template expects
at inference.
"""
import argparse
import sys
import json
from pathlib import Path

import torch._inductor.config as inductor_config
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# FlexAttention's compiled GEMMs otherwise hit "NoValidChoicesError: No choices exist
# for backend" on this box (sm_120 + triton 3.6, empty inductor cache): make sure the
# ATEN fallback is always an eligible autotune choice.
inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"
inductor_config.autotune_fallback_to_aten = True

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULTS = {
    "model": "Qwen/Qwen3-0.6B",
    "out": str(ROOT / "train/out/qwen3-0.6b-voxsum"),
    # single-pass hour-long transcripts: zh-TW hour ≈ 26k tokens -> native 32k max
    "max_len": 32768,
    "lr": 1.5e-5,
    "epochs": 2,
    "per_device_bs": 1,
    "grad_accum": 4,
    "warmup_ratio": 0.03,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(json.loads(Path(args.config).read_text()))

    tok = AutoTokenizer.from_pretrained(cfg["model"])

    # QAT: train against deployment numerics (TQ3 KV cache + int4-block32 weights) so
    # the model absorbs quantization error instead of meeting it at export time.
    attn_impl = cfg.get("attn", "kernels-community/flash-attn2")
    # Packed 32k training needs varlen attention: sdpa falls back to the O(n²) math
    # kernel on block-diagonal masks (182s/step), and flex_attention dies on this box
    # when a packed batch hits its decode path (NoValidChoicesError at step 22).
    # Prebuilt FA2 from the kernels hub is what TRL's packing path actually supports.
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype="bfloat16", attn_implementation=attn_impl,
    )
    # granite-4.0-h-1b is DENSE (num_experts=0) but still enters the MoE aux-loss
    # branch, where load_balancing_loss_func returns a plain int 0 and the model does
    # `aux_loss.to(loss.device)`. Returning a tensor instead just moves the failure to
    # DDP ("Tensors must be CUDA and dense"), so skip the branch outright: with no
    # experts there is nothing to balance.
    if model.config.model_type.startswith("granitemoehybrid"):
        _cls = model.__class__
        _orig_forward = _cls.forward

        def _forward_no_router(self, *a, **kw):
            kw["output_router_logits"] = False
            return _orig_forward(self, *a, **kw)

        _cls.forward = _forward_no_router
        print("[fix] granite dense: MoE aux-loss branch disabled")

    if cfg.get("qat_weight_bits"):
        from train.qat import fake_quant_weights_
        c = fake_quant_weights_(model, block_size=cfg.get("qat_block_size", 32),
                                bits=cfg["qat_weight_bits"])
        print(f"[qat] int{c['bits']}-block{cfg.get('qat_block_size', 32)} on {c['low']} Linears"
              f" (+{c['high']} kept int8: output head)")
    if cfg.get("qat_kv_bits"):
        from train.qat import install_kv_fake_quant
        how = install_kv_fake_quant(model, cfg["qat_kv_bits"])
        print(f"[qat] TQ KV fake-quant at {cfg['qat_kv_bits']} bits: {how}")

    # CPT mode: plain language modelling over raw text. Used to settle the 14,912 zh-TW
    # embeddings added by extend_tokenizer.py — in SFT the loss covers only completions,
    # so tokens that appear mainly inside transcripts would never get direct gradient.
    if cfg.get("cpt_data"):
        ds = load_dataset("json", data_files={"train": cfg["cpt_data"]})
        ds = ds.remove_columns([c for c in ds["train"].column_names if c != "text"])
        cfg["in_train_eval"] = False
    else:
        ds = load_dataset(
            "json",
            data_files={
                "train": cfg.get("train_file", str(ROOT / "data/sft/train.jsonl")),
                "val": str(ROOT / "data/sft/val.jsonl"),
            },
        )

    # prompt/completion conversational format: TRL masks the prompt tokens by
    # construction (Qwen3's chat template lacks the {% generation %} tag that
    # assistant_only_loss needs). The assistant turn carries no <think> block — the
    # .litertlm bundle's bare template means the model must answer directly.
    def to_prompt_completion(ex):
        return {"prompt": ex["messages"][:-1], "completion": ex["messages"][-1:]}

    if not cfg.get("cpt_data"):
        ds = ds.map(to_prompt_completion, num_proc=8)
        ds = ds.remove_columns([c for c in ds["train"].column_names
                                if c not in ("prompt", "completion")])
    # full val (5k packed rows) costs ~5 min/eval — a fixed 1k subsample tracks the
    # same curve at 1/5 the cost; final model selection re-runs the full eval anyway
    if not cfg.get("cpt_data"):
        ds["val"] = ds["val"].shuffle(seed=0).select(range(1024))

    sft_cfg = SFTConfig(
        output_dir=cfg["out"],
        max_length=cfg["max_len"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["per_device_bs"],
        per_device_eval_batch_size=cfg["per_device_bs"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=cfg["warmup_ratio"],
        bf16=True,
        # pack short examples into full 32k windows (median example ~1.7k tokens —
        # unpacked, the GPU is mostly padding and 2 epochs take ~10h instead of ~3h)
        packing=True,
        logging_steps=10,
        # granitemoehybrid returns MoE output fields that are None; accelerate's
        # gather_for_metrics then fails ("NoneType is not iterable"). These runs
        # select on the downstream judged eval anyway, so in-training eval is optional.
        eval_strategy="steps" if cfg.get("in_train_eval", True) else "no",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,  # disk is critically tight on this box
        # Gemma-3 + packed eval yields NaN eval_loss (an eval chunk can contain no
        # unmasked completion tokens). A NaN metric silently pins "best" to the FIRST
        # checkpoint, so best-model loading must be opt-in per model.
        load_best_model_at_end=cfg.get("best_model", True),
        metric_for_best_model="eval_loss" if cfg.get("best_model", True) else None,
        gradient_checkpointing=True,
        # 151k vocab × 32k positions of bf16 logits ≈ 10GB (+grad) — liger's fused CE
        # never materializes them; without it 32k-seq training OOMs even at bs=1
        use_liger_kernel=True,
        # TRL enables MoE aux-loss logging whenever the config looks MoE and
        # router_aux_loss_coef != 0; granite-4.0-h-1b is dense, so it then gathers a
        # None aux_loss and crashes. Zero the coefficient to turn that path off.
        router_aux_loss_coef=cfg.get("router_aux_loss_coef", 0.0),
        report_to="none",
        dataset_num_proc=8,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["val"] if cfg.get("in_train_eval", True) else None,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(cfg["out"] + "/final")
    tok.save_pretrained(cfg["out"] + "/final")


if __name__ == "__main__":
    main()
