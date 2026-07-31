"""DPO on the model's own long-document failures, under the same QAT numerics.

SFT can only show the model good answers; it is never penalised for inventing. The v3
SFT model still scored 44% faith<=2 and 22% polarity inversions on >12k-token
transcripts. These preference pairs are sampled from that model on exactly those long
prompts, with `chosen` = a faithful sample and `rejected` = an inverted/low-faith one,
so the gradient points directly away from the observed failure mode.

Weight fake-quant stays active so the preference is learned under deployment numerics
(the model ships int4); KV fake-quant follows the config, matching the SFT variant.

Launch:
  uv run torchrun --nproc_per_node=2 train/dpo.py --config configs/dpo_long.json
"""
import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULTS = {
    "model": str(ROOT / "train/out/qwen35-0.8b-voxsum-q4-kvf16/final"),
    "out": str(ROOT / "train/out/qwen35-0.8b-voxsum-q4-kvf16-dpo"),
    "pairs": str(ROOT / "data/dpo/long_pairs.jsonl"),
    "max_len": 32768,
    "max_prompt_len": 31000,
    "lr": 5e-7,          # DPO needs a much smaller LR than SFT
    "beta": 0.1,
    "epochs": 1,
    "per_device_bs": 1,
    "grad_accum": 8,
    "qat_weight_bits": 4,
    "qat_block_size": 32,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(json.loads(Path(args.config).read_text()))

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype="bfloat16",
        attn_implementation=cfg.get("attn", "kernels-community/flash-attn2"))

    if cfg.get("qat_weight_bits") == 4:
        from train.qat import fake_quant_weights_
        n = fake_quant_weights_(model, block_size=cfg.get("qat_block_size", 32))
        print(f"[qat] int4-block{cfg.get('qat_block_size', 32)} on {n} Linears")
    if cfg.get("qat_kv_bits"):
        from train.qat import install_kv_fake_quant
        print(f"[qat] KV {cfg['qat_kv_bits']}b: {install_kv_fake_quant(model, cfg['qat_kv_bits'])}")

    ds = load_dataset("json", data_files={"train": cfg["pairs"]})["train"]
    keep = {"prompt", "chosen", "rejected"}
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    print(f"{len(ds)} preference pairs")

    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(
            output_dir=cfg["out"],
            max_length=cfg["max_len"],
            max_prompt_length=cfg["max_prompt_len"],
            num_train_epochs=cfg["epochs"],
            per_device_train_batch_size=cfg["per_device_bs"],
            gradient_accumulation_steps=cfg["grad_accum"],
            learning_rate=cfg["lr"],
            beta=cfg["beta"],
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            bf16=True,
            logging_steps=5,
            save_strategy="no",
            gradient_checkpointing=True,
            report_to="none",
        ),
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(cfg["out"] + "/final")
    tok.save_pretrained(cfg["out"] + "/final")


if __name__ == "__main__":
    main()
