"""Multi-task GRPO for the deployed student: realtime reading turns, notes -> prose, notes -> title,
in one LoRA, starting from the SFT adapter (v6).

Per step: B prompts drawn from the mixed pool (distill/rl_prompts.py); G samples each from the
current policy (HF generate on this GPU); one judge call per sample (distill/rl_rewards.py, judge on
the other GPU); advantages normalised within each prompt's group; loss = -A * logp(sample) + beta
* KL(policy || reference), the reference being the starting adapter, frozen (a second PEFT adapter,
as in distill/dpo_agent.py). One update per batch, so the policy ratio is 1 and needs no clipping.
Each sequence is backpropagated alone, so an 8k-token reading turn fits one card.
"""
import argparse
import collections
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.rl_rewards import reward  # noqa: E402

BASE = "google/gemma-4-E2B-it-qat-q4_0-unquantized"
KW = {"tokenize": True, "return_dict": False, "enable_thinking": False}
MAX_NEW = {"read": 400, "prose": 600, "title": 48}
END = [1, 106]          # Gemma-4: <eos>, <turn|>


def end_ids(tok):
    """The ids that end a reply: eos plus the template's end-of-turn token, whatever the family."""
    ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    for t in ("<turn|>", "<|im_end|>", "<end_of_turn>", "<｜hy_place▁holder▁no▁2｜>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id and i >= 0:
            ids.add(i)
    return sorted(ids)


def token_logps(model, ids, start):
    pos = torch.arange(start - 1, ids.shape[0] - 1, device=ids.device)
    out = model(input_ids=ids.unsqueeze(0), logits_to_keep=pos, use_cache=False)
    lp = torch.log_softmax(out.logits[0].float(), -1)
    return lp.gather(-1, ids[start:].unsqueeze(-1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/train/rl_prompts.jsonl")
    ap.add_argument("--adapter", default="runs/sft/agent/lora-v6/epoch0")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--judge", default="http://127.0.0.1:8121/v1")
    ap.add_argument("--out", default="runs/sft/agent/grpo-v8")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=4, help="prompts per step")
    ap.add_argument("--group", type=int, default=6, help="samples per prompt")
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--task-weights", default="", help="e.g. read=0.4,prose=0.2,title=0.4; default: pool frequencies")
    ap.add_argument("--pairwise-title", action="store_true", help="title reward adds a comparison with the judge's title")
    ap.add_argument("--log", default="grpo_v8_log.jsonl")
    a = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    end = END if a.base == BASE else end_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 trust_remote_code=True).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, a.adapter, adapter_name="policy", is_trainable=True)
    model.load_adapter(a.adapter, adapter_name="ref", is_trainable=False)
    model.set_adapter("policy")
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, 5, a.steps)
    pool = ThreadPoolExecutor(16)
    log = open(os.path.join(os.path.dirname(a.out) or ".", a.log), "a")
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    for v in by_task.values():
        random.shuffle(v)
    weights = {t: len(v) for t, v in by_task.items()}
    if a.task_weights:
        weights = {k: float(w) for k, w in (kv.split("=") for kv in a.task_weights.split(","))}
    cursors = collections.Counter()

    def draw():
        t = random.choices(list(weights), weights=list(weights.values()))[0]
        r = by_task[t][cursors[t] % len(by_task[t])]
        cursors[t] += 1
        return r
    for step in range(1, a.steps + 1):
        t0 = time.time()
        batch = []
        while len(batch) < a.batch:
            batch.append(draw())
        # 1. sample
        model.eval()
        groups = []
        for r in batch:
            p = torch.tensor(tok.apply_chat_template(r["messages"], add_generation_prompt=True, **KW), device="cuda")
            with torch.no_grad():
                out = model.generate(input_ids=p.unsqueeze(0).repeat(a.group, 1), do_sample=True, temperature=a.temp,
                                     top_p=0.95, max_new_tokens=MAX_NEW[r["task"]], eos_token_id=end, pad_token_id=tok.pad_token_id or 0,
                                     use_cache=True)
            seqs = []
            for o in out:
                gen = o[len(p):]
                cut = next((i + 1 for i, t in enumerate(gen.tolist()) if t in end), len(gen))
                seqs.append(torch.cat([p, gen[:cut]]))
            texts = [tok.decode(s[len(p):], skip_special_tokens=True) for s in seqs]
            groups.append((r, len(p), seqs, texts))
        # 2. rewards (judge on the other GPU, in parallel)
        futs = [[pool.submit(reward, a.judge, r["task"], t, r, a.pairwise_title) for t in texts] for r, _, _, texts in groups]
        scored = [[f.result() for f in fs] for fs in futs]
        # 3. policy gradient with group-normalised advantages
        model.train()
        n_seq = sum(len(g[2]) for g in groups)
        stats = collections.defaultdict(list)
        for (r, plen, seqs, _), sc in zip(groups, scored):
            rew = torch.tensor([s for s, _ in sc])
            adv = (rew - rew.mean()) / (rew.std() + 1e-4)
            stats[r["task"]].append(rew.mean().item())
            for s, A in zip(seqs, adv.tolist()):
                if len(s) <= plen:
                    continue
                with torch.no_grad():
                    model.set_adapter("ref")
                    ref = token_logps(model, s, plen)
                model.set_adapter("policy")
                lp = token_logps(model, s, plen)
                kl = torch.exp(ref - lp) - (ref - lp) - 1          # k3 estimator, per token
                loss = (-A * lp + a.beta * kl).mean() / n_seq
                loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        rec = {"step": step, "s": round(time.time() - t0), **{k: round(sum(v) / len(v), 3) for k, v in stats.items()}}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)
        if step % a.save_every == 0 or step == a.steps:
            model.save_pretrained(os.path.join(a.out, f"step{step}"), selected_adapters=["policy"])


if __name__ == "__main__":
    main()
