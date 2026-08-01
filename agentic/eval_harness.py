"""Compare the memory-augmented harness against single-pass on the SAME long meetings.

Reports the two numbers the project actually cares about — inversion rate and faith<=2 —
plus the cost that decides whether it is deployable on ARM: total prompt tokens (a proxy
for prefill time) and number of LLM calls.
"""
import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "distill"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai import AsyncOpenAI

import tasks as T
from agentic.harness import Runner, SECTIONS, render, summarize
from configs.transcript_format import SpeakerStyle, render_transcript

JUDGE = """You are grading meeting notes against the transcript they came from.
FAITH: 1-5 — every claim supported by the transcript (5 = nothing invented)
COVER: 1-5 — the important content is captured
INVERT: YES or NO — do the notes state the OPPOSITE of the transcript about any decision,
approval, outcome or commitment (e.g. "approved" when it was rejected/postponed/pending)?
Answer exactly:
FAITH: <n>
COVER: <n>
INVERT: <YES|NO>

Transcript:
{transcript}

Notes:
{notes}"""


async def judge(client, model, transcript, notes, tok):
    ids = tok.encode(transcript)
    if len(ids) > 26000:
        transcript = tok.decode(ids[:26000])
    r = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content":
            JUDGE.format(transcript=transcript, notes=notes[:3000])}],
        max_tokens=32, temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    t = r.choices[0].message.content or ""
    f = re.search(r"FAITH:\s*(\d)", t); c = re.search(r"COVER:\s*(\d)", t)
    v = re.search(r"INVERT:\s*(YES|NO)", t, re.I)
    return (int(f.group(1)) if f else 0, int(c.group(1)) if c else 0,
            bool(v and v.group(1).upper() == "YES"))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-url", default="http://127.0.0.1:8089/v1")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8088/v1")
    ap.add_argument("--modes", nargs="+", default=["single", "map", "reread"])
    ap.add_argument("--min-tokens", type=int, default=12000)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--out", default="eval/report_agentic.json")
    args = ap.parse_args()

    runner = Runner(args.student_url, "student")
    jc = AsyncOpenAI(base_url=args.judge_url, api_key="x", timeout=1800.0)
    tok = runner.tok

    meetings = []
    for src in ("qmsum", "meetingbank", "vcsum"):
        for line in (Path("data/normalized") / f"{src}.jsonl").open():
            d = json.loads(line)
            if d["split"] != "test":
                continue
            tr = render_transcript(d, SpeakerStyle(
                "stags" if any(u["speaker"] for u in d["utterances"]) else "none"))
            n = len(tok.encode(tr))
            if args.min_tokens <= n <= 30000:
                meetings.append((d, tr, n))
    random.Random(0).shuffle(meetings)
    meetings = meetings[: args.n]
    print(f"{len(meetings)} long meetings "
          f"(median {sorted(m[2] for m in meetings)[len(meetings)//2]} tokens)", flush=True)

    results = {m: [] for m in args.modes}
    for k, (d, tr, ntok) in enumerate(meetings):
        lang = d["lang"]
        for mode in args.modes:
            c0, p0 = runner.calls, runner.prompt_tokens
            try:
                if mode == "single":
                    prompt = (T.NOTES_TEMPLATE_ZH % tr if lang == "zh-TW"
                              else T.NOTES_TEMPLATE % ("", tr))
                    notes = await runner.gen(prompt, T.NOTES_MAX_TOKENS)
                    nch = 1
                else:
                    state, nch = await summarize(runner, tr, lang, mode=mode,
                                                 chunk_budget=args.chunk)
                    notes = render(state)
                fa, co, inv = await judge(jc, "teacher", tr, notes, tok)
                results[mode].append({
                    "meeting": d["id"], "lang": lang, "ntok": ntok, "chunks": nch,
                    "faith": fa, "cover": co, "invert": inv,
                    "calls": runner.calls - c0, "prompt_tokens": runner.prompt_tokens - p0,
                    "notes": notes[:600]})
            except Exception as e:
                results[mode].append({"meeting": d["id"], "error": str(e)[:120]})
        print(f"  [{k+1}/{len(meetings)}] {d['id']}", flush=True)

    agg = {}
    for mode, rs in results.items():
        ok = [r for r in rs if "error" not in r]
        if not ok:
            continue
        agg[mode] = {
            "n": len(ok),
            "faith": round(sum(r["faith"] for r in ok) / len(ok), 2),
            "faith_le2": round(sum(r["faith"] <= 2 for r in ok) / len(ok), 3),
            "invert": round(sum(r["invert"] for r in ok) / len(ok), 3),
            "cover": round(sum(r["cover"] for r in ok) / len(ok), 2),
            "calls": round(sum(r["calls"] for r in ok) / len(ok), 1),
            "prompt_tokens": int(sum(r["prompt_tokens"] for r in ok) / len(ok)),
        }
    Path(args.out).write_text(json.dumps({"agg": agg, "results": results},
                                         ensure_ascii=False, indent=1))
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
