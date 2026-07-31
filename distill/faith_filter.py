"""Judge distillation targets for faithfulness and write a filtered dataset.

Motivation: v1/v2 training data was filtered for language, format and duplicates but
NEVER for factual correctness, so teacher hallucinations (including polarity inversions
— "approved" when the transcript says rejected) were trained into the student. Measured
consequence: 34% of native-English outputs score faith<=2 in held-out eval.

The judge sees the transcript slice the target was generated from plus the target, and
answers two questions:
  FAITH  1-5  — is every claim supported by the transcript?
  INVERT Y/N  — does it state the OPPOSITE of the transcript on any decision/outcome?

A single INVERT=Y is disqualifying regardless of FAITH.

Usage:
  python distill/faith_filter.py --in distill/targets_single.jsonl \
      --out distill/targets_single.judged.jsonl [--tasks notes summarize actions]
Resume-safe: already-judged (meeting_id, variant, task, phase) keys are skipped.
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI
from transformers import AutoTokenizer

BASE_URL = "http://127.0.0.1:8088/v1"
JUDGE = "teacher"
SEM = asyncio.Semaphore(8)
client = AsyncOpenAI(base_url=BASE_URL, api_key="x", timeout=1800.0)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

PROMPT = """You are auditing notes generated from a meeting transcript.

Answer EXACTLY in this format, nothing else:
FAITH: <1-5>
INVERT: <YES or NO>

FAITH 5 = every claim is directly supported by the transcript; 1 = mostly fabricated.
INVERT = YES if the notes state the OPPOSITE of the transcript about any decision,
approval, outcome, or commitment (e.g. saying something was approved/agreed/completed
when the transcript says it was rejected, postponed, pending, or failed). Otherwise NO.

Transcript:
{transcript}

Generated notes:
{output}"""


def transcript_of(prompt: str) -> str:
    """Recover the transcript slice from a training prompt (all task templates put it
    after a 'Transcript:' / '逐字稿:' marker)."""
    for marker in ("Transcript:\n", "逐字稿:\n", "Partial summaries:\n", "部分摘要:\n",
                   "Summary:\n", "摘要:\n", "Items:\n"):
        i = prompt.rfind(marker)
        if i >= 0:
            return prompt[i + len(marker):]
    return prompt


async def judge(transcript: str, output: str):
    ids = tok.encode(transcript)
    if len(ids) > 24000:
        transcript = tok.decode(ids[:24000])
    p = PROMPT.format(transcript=transcript, output=output[:3000])
    async with SEM:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=JUDGE, messages=[{"role": "user", "content": p}],
                    max_tokens=24, temperature=0.0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                t = (r.choices[0].message.content or "")
                f = re.search(r"FAITH:\s*(\d)", t)
                v = re.search(r"INVERT:\s*(YES|NO)", t, re.I)
                return (int(f.group(1)) if f else 0,
                        bool(v and v.group(1).upper() == "YES"))
            except Exception:
                if attempt == 2:
                    return (0, False)
                await asyncio.sleep(3 * (attempt + 1))
    return (0, False)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="only judge these tasks (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="judge at most N records (0=all)")
    ap.add_argument("--url", default=BASE_URL, help="judge endpoint")
    ap.add_argument("--shard", type=int, default=0, help="this worker's index")
    ap.add_argument("--of", type=int, default=1, help="total workers (disjoint shards)")
    args = ap.parse_args()
    global client
    client = AsyncOpenAI(base_url=args.url, api_key="x", timeout=1800.0)

    done = set()
    outp = Path(args.out)
    if outp.exists():
        for line in outp.open():
            try:
                d = json.loads(line)
                done.add((d["meeting_id"], d["variant"], d["task"], d["phase"]))
            except Exception:
                pass
        print(f"resume: {len(done)} already judged", flush=True)

    rows = []
    for line in Path(args.inp).open():
        d = json.loads(line)
        if args.tasks and d["task"] not in args.tasks:
            continue
        if (d["meeting_id"], d["variant"], d["task"], d["phase"]) in done:
            continue
        rows.append(d)
    if args.of > 1:   # disjoint shards so parallel workers never judge the same record
        rows = [r for i, r in enumerate(rows) if i % args.of == args.shard]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} records to judge", flush=True)

    f = outp.open("a", encoding="utf-8")
    stats = {"n": 0, "invert": 0, "low": 0}

    async def one(d):
        fa, inv = await judge(transcript_of(d["prompt"]), d["completion"])
        d["judge"] = {"faith": fa, "invert": inv}
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
        f.flush()
        stats["n"] += 1
        stats["invert"] += int(inv)
        stats["low"] += int(fa and fa <= 3)

    for i in range(0, len(rows), 200):
        await asyncio.gather(*[one(d) for d in rows[i:i + 200]])
        print(f"judged {stats['n']}/{len(rows)} | inversions {stats['invert']} "
              f"| faith<=3 {stats['low']}", flush=True)
    f.close()


if __name__ == "__main__":
    asyncio.run(main())
