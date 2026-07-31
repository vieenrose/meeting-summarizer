"""Architecture probe: can a 270M+270M encoder-decoder match a 0.8B decoder-only on
meeting summarization?

Not a deployment candidate — llama.cpp has no `t5gemma2` arch and LiteRT-LM is
decoder-only, and its 32k limit cannot hold a 3-hour zh-TW meeting (~33k tokens). This
only answers whether the bidirectional-encoder shape is worth pursuing later.

Trains on the NOTES task (the deliverable format, and the strictest faithfulness test).
"""
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

ROOT = Path(__file__).resolve().parent.parent
MODEL = "google/t5gemma-2-1b-1b"
MAX_IN, MAX_OUT = 4096, 640     # probe scale; the real task needs 32k

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

ds = load_dataset("json", data_files={
    "train": str(ROOT / "data/sft/t5_notes.jsonl"),
    "val": str(ROOT / "data/sft/t5_val.jsonl")})


def prep(ex):
    m = tok(ex["input"], max_length=MAX_IN, truncation=True)
    lab = tok(text_target=ex["target"], max_length=MAX_OUT, truncation=True)
    m["labels"] = lab["input_ids"]
    return m


ds = ds.map(prep, remove_columns=["input", "target"], num_proc=8)

trainer = Seq2SeqTrainer(
    model=model,
    args=Seq2SeqTrainingArguments(
        output_dir=str(ROOT / "train/out/t5gemma-1b-probe"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,          # 4x the params of the 270m probe -> lower LR
        num_train_epochs=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=20,
        eval_strategy="no",
        save_strategy="no",
        gradient_checkpointing=True,
        report_to="none",
    ),
    train_dataset=ds["train"],
    data_collator=DataCollatorForSeq2Seq(tok, model=model),
)
trainer.train()
trainer.save_model(str(ROOT / "train/out/t5gemma-1b-probe/final"))
tok.save_pretrained(str(ROOT / "train/out/t5gemma-1b-probe/final"))
print("saved")
