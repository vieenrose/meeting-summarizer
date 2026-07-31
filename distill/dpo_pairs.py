"""Build DPO preference pairs that target hallucination directly.

SFT alone cannot fix the failure mode we measured (30-34% of native-English outputs
score faith<=2 even at temperature 0): the model was only ever shown good answers, never
penalised for inventing. DPO on (faithful, hallucinated) pairs *from the model's own
distribution* optimises exactly the axis that is broken.

For each prompt we sample K completions from the current fine-tune, judge each for
faithfulness + polarity inversion against the transcript, and emit a pair when the gap
is decisive: chosen = highest-faith (>=4, no inversion), rejected = a low-faith or
inverted sample. Prompts where every sample is good (or all bad) yield nothing — those
teach no preference.

Usage (student on :8089, judge on :8088):
  python distill/dpo_pairs.py --n-prompts 4000 --k 4 --out data/dpo/pairs.jsonl
"""
import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from faith_filter import PROMPT as JUDGE_PROMPT, transcript_of

STUDENT_URL = "http://127.0.0.1:8089/v1"
JUDGE_URL = "http://127.0.0.1:8088/v1"
student = AsyncOpenAI(base_url=STUDENT_URL, api_key="x", timeout=1800.0)
judge_c = AsyncOpenAI(base_url=JUDGE_URL, api_key="x", timeout=1800.0)
JUDGES = [judge_c]      # round-robin pool; judging is the throughput bottleneck
_jrr = [0]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
SEM = asyncio.Semaphore(8)

MAX_TOKENS = {"notes": 640, "summarize": 288, "actions": 384, "exec_summary": 224,
              "detailed_summary": 512, "decisions": 320, "topics": 256,
              "open_questions": 320, "risks_disagreements": 320, "title": 24}


async def sample(prompt: str, max_tokens: int, temp: float) -> str:
    async with SEM:
        try:
            r = await student.chat.completions.create(
                model="student", messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=temp, top_p=0.95,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return (r.choices[0].message.content or "").strip()
        except Exception:
            return ""


async def judge(transcript: str, output: str):
    ids = tok.encode(transcript)
    if len(ids) > 24000:
        transcript = tok.decode(ids[:24000])
    p = JUDGE_PROMPT.format(transcript=transcript, output=output[:3000])
    _jrr[0] = (_jrr[0] + 1) % len(JUDGES)
    jc = JUDGES[_jrr[0]]
    async with SEM:
        try:
            r = await jc.chat.completions.create(
                model="teacher", messages=[{"role": "user", "content": p}],
                max_tokens=24, temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            t = r.choices[0].message.content or ""
            f = re.search(r"FAITH:\s*(\d)", t)
            v = re.search(r"INVERT:\s*(YES|NO)", t, re.I)
            return (int(f.group(1)) if f else 0,
                    bool(v and v.group(1).upper() == "YES"))
        except Exception:
            return (0, False)


async def one_prompt(rec, k, out, stats):
    prompt = rec["prompt"]
    mt = MAX_TOKENS.get(rec["task"], 384)
    transcript = transcript_of(prompt)
    # one near-greedy + k-1 sampled: the greedy answer is what ships, the sampled ones
    # expose the model's failure modes
    temps = [0.0] + [0.9] * (k - 1)
    cands = await asyncio.gather(*[sample(prompt, mt, t) for t in temps])
    cands = [c for c in cands if c.strip()]
    if len(cands) < 2:
        return
    verdicts = await asyncio.gather(*[judge(transcript, c) for c in cands])
    scored = [(f, inv, c) for (f, inv), c in zip(verdicts, cands)]
    good = [s for s in scored if s[0] >= 4 and not s[1]]
    bad = [s for s in scored if s[1] or (s[0] and s[0] <= 2)]
    stats["prompts"] += 1
    stats["inverted"] += sum(1 for s in scored if s[1])
    if not good or not bad:
        return
    chosen = max(good, key=lambda s: s[0])
    rejected = min(bad, key=lambda s: (not s[1], s[0]))   # prefer an inverted sample
    if chosen[2] == rejected[2]:
        return
    out.write(json.dumps({
        "prompt": [{"role": "system", "content": "You are a helpful assistant."},
                   {"role": "user", "content": prompt}],
        "chosen": [{"role": "assistant", "content": chosen[2]}],
        "rejected": [{"role": "assistant", "content": rejected[2]}],
        "task": rec["task"], "lang": rec["lang"], "meeting_id": rec["meeting_id"],
        "chosen_faith": chosen[0], "rejected_faith": rejected[0],
        "rejected_inverted": rejected[1],
    }, ensure_ascii=False) + "\n")
    out.flush()
    stats["pairs"] += 1


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="distill/targets_single.jsonl")
    ap.add_argument("--out", default="data/dpo/pairs.jsonl")
    ap.add_argument("--n-prompts", type=int, default=4000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="only prompts whose transcript is at least this many tokens")
    ap.add_argument("--judge-urls", nargs="*", default=None,
                    help="one or more judge endpoints to round-robin across")
    args = ap.parse_args()
    if args.judge_urls:
        JUDGES.clear()
        JUDGES.extend(AsyncOpenAI(base_url=u, api_key="x", timeout=1800.0)
                      for u in args.judge_urls)

    rows = []
    seen = set()
    outp0 = Path(args.out)
    if outp0.exists():
        for line in outp0.open():
            try:
                d = json.loads(line)
                seen.add((d["meeting_id"], d["task"]))
            except Exception:
                pass
        print(f"resume: {len(seen)} prompts already produced pairs", flush=True)
    for line in Path(args.src).open():
        d = json.loads(line)
        if d["split"] != "train" or d["phase"] not in ("single", "map"):
            continue
        key = (d["meeting_id"], d["task"])
        if key in seen:
            continue
        if args.min_tokens:
            if len(tok.encode(transcript_of(d["prompt"]))) < args.min_tokens:
                continue
        seen.add(key)
        rows.append({k: d[k] for k in ("prompt", "task", "lang", "meeting_id")})
    random.Random(0).shuffle(rows)
    rows = rows[: args.n_prompts]
    print(f"{len(rows)} prompts", flush=True)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out = outp.open("a", encoding="utf-8")
    stats = {"prompts": 0, "pairs": 0, "inverted": 0}
    sem = asyncio.Semaphore(args.parallel)

    async def guarded(r):
        async with sem:
            await one_prompt(r, args.k, out, stats)

    tasks = [guarded(r) for r in rows]
    for i in range(0, len(tasks), 100):
        await asyncio.gather(*tasks[i:i + 100])
        print(f"{stats['prompts']} prompts -> {stats['pairs']} pairs "
              f"({stats['inverted']} inverted samples seen)", flush=True)
    out.close()


if __name__ == "__main__":
    asyncio.run(main())
