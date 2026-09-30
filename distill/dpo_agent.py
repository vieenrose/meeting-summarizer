"""DPO of the agent-SFT Gemma-4-E2B on on-policy correction pairs (distill/correct_onpolicy.py).

Each pair shares the student's context at a window it got wrong -- system, compacted journal of
its own earlier notes, the previous window and its reply, then the window -- and differs only in
the reply: the teacher's correction (chosen) against the student's (rejected). The policy starts
from the SFT adapter; the reference is the same adapter, frozen (a second PEFT adapter), so no
second model is loaded. Loss = DPO + alpha * NLL(chosen) (RPO), which keeps the chosen replies
likely rather than only pushing the rejected ones down.

Runs on one GPU, or data-parallel under torch.distributed.run (each rank takes its share of every
batch, and the LoRA gradients are summed across ranks before each optimizer step, so the updates
are the same as on one GPU).
"""
import argparse
import json
import math
import os
import random
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.realtime_agent import SYSTEM_V3, render, windows_of  # noqa: E402
from summarizer.ingest import parse_line  # noqa: E402

BASE = "google/gemma-4-E2B-it-qat-q4_0-unquantized"
KW = {"tokenize": True, "return_dict": False, "enable_thinking": False}


def context(rec, lines, wins, k, count, budget=2500):
    """The student's conversation just before window k, in its post-restart form."""
    notes = [n for n in rec["notes"] if n["window"] < k]
    key = {"DECISION": 0, "OPEN-ISSUE": 1, "ACTION": 2}
    order = sorted(range(len(notes)), key=lambda i: (key.get((notes[i]["tag"] or "").upper(), 3), -i))
    chosen, used = set(), 0
    for i in order:
        t = count(render(notes[i]))
        if used + t <= budget:
            chosen.add(i)
            used += t
    rest = len(notes) - len(chosen)
    journal = "\n".join(render(notes[i]) for i in sorted(chosen)) + (f"\n（另有 {rest} 則較早的筆記未列出）" if rest else "")
    msgs = [{"role": "system", "content": SYSTEM_V3},
            {"role": "user", "content": "## 筆記本（至今）\n" + (journal or "（尚無筆記）")},
            {"role": "assistant", "content": "NEXT"}]
    replies = {t["window"]: t["reply"] for t in rec["trace"] if "reply" in t and t["window"] != "overview"}
    if k > 1 and (k - 1) in replies:
        msgs += [{"role": "user", "content": f"## 逐字稿片段 {k - 1}\n" + "\n".join(l.render() for l in wins[k - 2])},
                 {"role": "assistant", "content": replies[k - 1]}]
    msgs.append({"role": "user", "content": f"## 逐字稿片段 {k}\n" + "\n".join(l.render() for l in wins[k - 1])})
    return msgs


def seq(tok, msgs, reply):
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, **KW)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": reply}], **KW)
    assert full[:len(p)] == p
    return torch.tensor(full), len(p)


def logp(model, ids, start):
    ids = ids.unsqueeze(0).cuda()
    pos = torch.arange(start - 1, ids.shape[1] - 1, device=ids.device)
    out = model(input_ids=ids, logits_to_keep=pos, use_cache=False)
    lp = torch.log_softmax(out.logits[0].float(), -1)
    tgt = ids[0, start:]
    return lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum(), tgt.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/train/agent_dpo_pairs.jsonl")
    ap.add_argument("--student", default="runs/student/onpol-ft-ep0")
    ap.add_argument("--transcripts", default="data/v2/transcripts")
    ap.add_argument("--sft-adapter", default="runs/sft/agent/lora/epoch0")
    ap.add_argument("--out", default="runs/sft/agent/dpo")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.2, help="weight of the NLL term on chosen")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    a = ap.parse_args()
    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    random.seed(0)
    from transformers import AutoTokenizer as AT
    wtok = AT.from_pretrained("Qwen/Qwen3.6-35B-A3B-FP8")
    count = lambda t: len(wtok.encode(t, add_special_tokens=False))  # noqa: E731
    tok = AutoTokenizer.from_pretrained(BASE)
    pairs = [json.loads(l) for l in open(a.pairs, encoding="utf-8")]
    cache, data = {}, []
    for p in pairs:
        sid = p["session"]
        if sid not in cache:
            rec = json.load(open(f"{a.student}/{sid}.json", encoding="utf-8"))
            lines = [parse_line(l) for l in open(os.path.join(a.transcripts, sid + ".txt"), encoding="utf-8")
                     .read().splitlines() if l.strip()]
            cache[sid] = (rec, lines, windows_of(lines, count))
        rec, lines, wins = cache[sid]
        msgs = context(rec, lines, wins, p["window"], count)
        c, sc = seq(tok, msgs, p["chosen"])
        r, sr = seq(tok, msgs, p["rejected"])
        data.append((c, sc, r, sr))
    if rank == 0:
        print(f"pairs {len(data)}, longest {max(len(c) for c, *_ in data)} tokens, {world} GPU(s)", flush=True)

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, a.sft_adapter, adapter_name="policy", is_trainable=True)
    model.load_adapter(a.sft_adapter, adapter_name="ref", is_trainable=False)
    model.set_adapter("policy")
    if world > 1:
        # after the adapters are loaded: under an initialised process group PEFT takes a
        # tensor-parallel loading path that fails on this transformers version
        dist.init_process_group("nccl")
    # Gradient checkpointing is only active in train mode; without it one 8k-token graph fills the
    # card. Dropout off, so the no-grad reference and margin passes are deterministic.
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    model.train()
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(data) / a.accum) * a.epochs
    sched = get_cosine_schedule_with_warmup(opt, max(1, steps // 10), steps)
    step = 0
    for ep in range(a.epochs):
        random.shuffle(data)
        for b in range(0, len(data), a.accum):
            batch = data[b:b + a.accum]
            stats = {"loss": 0.0, "margin": 0.0, "acc": 0}
            for c, sc, r, sr in batch[rank::world]:
                # Two 8k-token graphs at once do not fit a 32 GB card. The loss depends on the two
                # policy log-probs only through the margin, so compute everything without grad,
                # then backpropagate each sequence alone with its exact coefficient:
                #   dL/dpc = -beta * sigmoid(-m) - alpha / nc,   dL/dpr = beta * sigmoid(-m)
                with torch.no_grad():
                    model.set_adapter("ref")
                    rc, _ = logp(model, c, sc)
                    rr, _ = logp(model, r, sr)
                    model.set_adapter("policy")
                    pc0, nc = logp(model, c, sc)
                    pr0, _ = logp(model, r, sr)
                    margin = a.beta * ((pc0 - rc) - (pr0 - rr))
                    sig = torch.sigmoid(-margin).item()
                model.set_adapter("policy")
                pc, _ = logp(model, c, sc)
                ((-a.beta * sig - a.alpha / nc) * pc / len(batch)).backward()
                pr, _ = logp(model, r, sr)
                ((a.beta * sig) * pr / len(batch)).backward()
                loss = -torch.nn.functional.logsigmoid(margin).item() + a.alpha * (-pc0.item() / nc)
                stats["loss"] += loss / len(batch)
                stats["margin"] += margin.item() / len(batch)
                stats["acc"] += int(margin.item() > 0)
            if world > 1:
                for p in params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad)
                t = torch.tensor([stats["loss"], stats["margin"], stats["acc"]], device="cuda")
                dist.all_reduce(t)
                stats = {"loss": t[0].item(), "margin": t[1].item(), "acc": int(t[2].item())}
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if rank == 0 and (step % 5 == 0 or step == 1):
                print(f"step {step}/{steps} loss {stats['loss']:.4f} margin {stats['margin']:.3f} "
                      f"acc {stats['acc']}/{len(batch)}", flush=True)
    if rank == 0:
        model.save_pretrained(a.out, selected_adapters=["policy"])
        print("saved", a.out)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
