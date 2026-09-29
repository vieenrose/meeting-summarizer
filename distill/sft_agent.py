"""LoRA SFT of Gemma-4-E2B on the teacher's realtime-agent conversations (distill/build_agent_sft.py).

One row is one conversation segment -- system, journal, then (window, actions) turns -- and the
loss falls on every kept teacher turn in it at once, so a 14k-token segment trains ~4 turns for
the price of one forward pass. Everything else (system prompt, journal, transcript windows,
rejected turns) is context only.

The base is the QAT checkpoint, which the deployed Q4_0 GGUF is made from: the adapter is merged
into those weights and requantized (distill/merge_agent_lora.py).

Known Gemma-4 traps (see distill/sft_gemma.py): the 262k-entry LM head is computed only at loss
positions (logits_to_keep), and LoRA is scoped to the language model so the vision and audio
towers are left alone.
"""
import argparse
import json
import math
import os
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

BASE = "google/gemma-4-E2B-it-qat-q4_0-unquantized"
TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def encode(tok, row):
    """input_ids and labels (-100 outside the loss turns) for one conversation segment."""
    msgs = row["messages"]
    kw = {"tokenize": True, "return_dict": False, "enable_thinking": False}
    ids = tok.apply_chat_template(msgs, **kw)
    labels = [-100] * len(ids)
    for i in row["loss_turns"]:
        start = len(tok.apply_chat_template(msgs[:i], add_generation_prompt=True, **kw))
        end = len(tok.apply_chat_template(msgs[:i + 1], **kw))
        assert ids[:end] == tok.apply_chat_template(msgs[:i + 1], **kw), "template is not prefix-stable"
        labels[start:end] = ids[start:end]
    return torch.tensor(ids), torch.tensor(labels)


def loss_of(model, ids, labels):
    ids, labels = ids.unsqueeze(0).cuda(), labels.cuda()
    positions = (labels[1:] != -100).nonzero().squeeze(-1)   # logit at t predicts token t+1
    out = model(input_ids=ids, logits_to_keep=positions, use_cache=False)
    logits = out.logits[0].float()
    return torch.nn.functional.cross_entropy(logits, labels[positions + 1], reduction="sum"), positions.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="data/train/agent_sft_rows.jsonl")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default="runs/sft/agent/lora")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--val-sessions", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke test")
    a = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)

    rows = [json.loads(l) for l in open(a.rows, encoding="utf-8")]
    sessions = sorted({r["session"] for r in rows})
    val_s = set(random.Random(0).sample(sessions, a.val_sessions))
    tok = AutoTokenizer.from_pretrained(a.base)
    data = [(r["session"], *encode(tok, r)) for r in rows]
    train = [(i, l) for s, i, l in data if s not in val_s]
    val = [(i, l) for s, i, l in data if s in val_s]
    n_tok = sum((l != -100).sum().item() for _, l in train)
    print(f"train segments {len(train)}, val {len(val)} ({len(val_s)} sessions), "
          f"loss tokens {n_tok}, longest {max(len(i) for i, _ in train)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(train) / a.accum) * a.epochs if a.max_steps < 0 else a.max_steps
    sched = get_cosine_schedule_with_warmup(opt, min(20, steps // 10), steps)

    def evaluate():
        model.eval()
        tot = n = 0
        with torch.no_grad():
            for ids, labels in val:
                l, k = loss_of(model, ids, labels)
                tot, n = tot + l.item(), n + k
        model.train()
        return tot / max(1, n)

    print(f"step 0 val loss {evaluate():.4f}", flush=True)
    step = 0
    model.train()
    for ep in range(a.epochs):
        order = list(range(len(train)))
        random.shuffle(order)
        for b in range(0, len(order), a.accum):
            batch = [train[j] for j in order[b:b + a.accum]]
            denom = sum((l != -100).sum().item() - (1 if l[0] != -100 else 0) for _, l in batch)
            run = 0.0
            for ids, labels in batch:
                l, _ = loss_of(model, ids, labels)
                (l / denom).backward()
                run += l.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0:
                print(f"ep {ep} step {step}/{steps} loss {run / denom:.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
            if 0 < a.max_steps <= step:
                break
        print(f"epoch {ep} val loss {evaluate():.4f}", flush=True)
        model.save_pretrained(os.path.join(a.out, f"epoch{ep}"))
        if 0 < a.max_steps <= step:
            break
    tok.save_pretrained(a.out)


if __name__ == "__main__":
    main()
