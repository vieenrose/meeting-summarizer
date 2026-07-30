"""Evaluate a student model (served on an OpenAI endpoint) on held-out test meetings,
replaying the app's flows; judge faithfulness with the teacher.

  python eval/run_eval.py --student-url http://127.0.0.1:8089/v1 --student qwen3-ft \
      --judge-url http://127.0.0.1:8088/v1 --n-per-source 12 --out eval/report_ft.json

Scores per (meeting, task):
  lang_ok  — output script matches effective target (validate_llm.py logic)
  fmt_ok   — no think/markdown/preamble, bullet & title limits respected
  faith    — teacher judge 1-5: faithful to transcript, no inventions
  cover    — teacher judge 1-5: covers the important content
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

from openai import AsyncOpenAI
from transformers import AutoTokenizer

import tasks as T
from configs.transcript_format import SpeakerStyle, render_transcript
from data.build_sft import lang_ok

# single-pass: whole transcript in one prompt, same budget the student was trained with
N_CTX = 32768
_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def fits(prompt: str, out_tokens: int) -> bool:
    return len(_tok.encode(prompt)) + out_tokens + 96 <= N_CTX

JUDGE_TMPL = """You are grading a meeting-notes model. Given a transcript (possibly truncated) and the model's output for the task "{task}", rate:
FAITH: 1-5 — everything in the output is supported by the transcript (5 = no inventions at all)
COVER: 1-5 — the output captures the most important content for the task (5 = nothing important missed)
Answer in exactly this format:
FAITH: <n>
COVER: <n>

Transcript:
{transcript}

Model output:
{output}"""

BAD = re.compile(r"<think>|^#{1,6}\s|^(here('s| is| are)|sure|okay|certainly)\b|以下是", re.I | re.M)


def fmt_ok(task: str, out: str, lang: str) -> bool:
    if not out.strip() or BAD.search(out):
        return False
    if task == "title":
        return "\n" not in out and (len(out.split()) <= 10 if lang == "en" else len(out) <= 20)
    if task == "summarize":
        return sum(1 for l in out.split("\n") if l.strip()) <= 12 and len(out) <= 1600
    return True


async def gen(client, model, prompt, max_tokens, sampler):
    sampler = dict(sampler)
    extra = {"chat_template_kwargs": {"enable_thinking": False}, **sampler.pop("extra_body", {})}
    r = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, extra_body=extra, **sampler)
    return (r.choices[0].message.content or "").strip()


async def judge(client, model, task, transcript, output):
    ids = _tok.encode(transcript)
    if len(ids) > 26000:  # keep judge prompt inside the teacher's 40k slot
        transcript = _tok.decode(ids[:26000])
    p = JUDGE_TMPL.format(task=task, transcript=transcript, output=output[:2000])
    r = await gen(client, model, p, 64, {"temperature": 0.0})
    f = re.search(r"FAITH:\s*(\d)", r)
    c = re.search(r"COVER:\s*(\d)", r)
    return (int(f.group(1)) if f else 0, int(c.group(1)) if c else 0)


async def eval_meeting(students, judge_c, args, meeting, results):
    transcript = render_transcript(meeting, SpeakerStyle(
        "stags" if any(u["speaker"] for u in meeting["utterances"]) else "none"))
    lang = meeting["lang"]
    cross = "zh-TW" if lang == "en" else "en"
    cases = [
        ("summarize", None), ("summarize", cross),
        ("actions", None), ("title", None),
        ("exec_summary", None), ("open_questions", None),
    ]
    sampler = dict(args.sampler)
    for task, tgt in cases:
        eff = tgt or lang
        zh_instr = eff == "zh-TW"
        try:
            if task == "summarize" or task == "title":
                m_ins, r_ins, m_tok, _ = T.STYLES["bullet"]
                clause = T.lang_clause(T.TARGET_LANG[tgt] if tgt else None)
                p = (T.MAP_TEMPLATE_ZH % transcript if zh_instr
                     else T.MAP_TEMPLATE % ("Summarize the key points of this transcript." + clause, m_ins, transcript))
                if not fits(p, m_tok):
                    continue
                out = await gen(students, args.student, p, m_tok, sampler)
                if task == "title" and out:
                    tp = T.TITLE_TEMPLATE_ZH % out if zh_instr else T.TITLE_TEMPLATE % (clause, out)
                    out = await gen(students, args.student, tp, 24, sampler)
                ref = transcript
            elif task == "actions":
                clause = T.action_lang_clause(T.TARGET_LANG[tgt] if tgt else None)
                p = T.ACTION_MAP_TEMPLATE % (clause, transcript)
                if not fits(p, 384):
                    continue
                out = await gen(students, args.student, p, 384, sampler)
                ref = transcript
            else:
                en_p, zh_p, mt = T.INSIGHT_TASKS[task]
                p = zh_p % transcript if zh_instr else en_p % ("", transcript)
                if not fits(p, mt):
                    continue
                out = await gen(students, args.student, p, mt, sampler)
                ref = transcript
            l_ok = lang_ok(eff, out) if out else False
            f_ok = fmt_ok(task, out, eff)
            fa, co = await judge(judge_c, args.judge, task, ref, out) if out else (0, 0)
            results.append({"meeting": meeting["id"], "src_lang": lang, "task": task,
                            "tgt": tgt, "lang_ok": l_ok, "fmt_ok": f_ok,
                            "faith": fa, "cover": co, "output": out[:400]})
        except Exception as e:
            results.append({"meeting": meeting["id"], "task": task, "tgt": tgt, "error": str(e)})


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-url", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--judge-url", default="http://127.0.0.1:8088/v1")
    ap.add_argument("--judge", default="teacher")
    ap.add_argument("--n-per-source", type=int, default=12)
    ap.add_argument("--out", required=True)
    # "plain" = bare nucleus sampling; "app" = Qwen3 non-thinking profile the registry
    # entry will ship (repetition control matters for cross-lingual on a 0.6B)
    ap.add_argument("--sampler", choices=["plain", "app"], default="plain")
    args = ap.parse_args()
    args.sampler = ({"temperature": 0.7, "top_p": 0.9} if args.sampler == "plain" else
                    {"temperature": 0.7, "top_p": 0.8,
                     "extra_body": {"top_k": 20, "presence_penalty": 1.0}})

    students = AsyncOpenAI(base_url=args.student_url, api_key="x")
    judge_c = AsyncOpenAI(base_url=args.judge_url, api_key="x")

    meetings = []
    for src in ["qmsum", "vcsum", "dialogsum", "meetingbank"]:
        rows = [json.loads(l) for l in (Path("data/normalized") / f"{src}.jsonl").open()
                if json.loads(l)["split"] == "test"]
        random.Random(1).shuffle(rows)
        meetings += rows[: args.n_per_source]

    results = []
    sem = asyncio.Semaphore(8)

    async def run(m):
        async with sem:
            await eval_meeting(students, judge_c, args, m, results)

    await asyncio.gather(*[run(m) for m in meetings])

    ok = [r for r in results if "error" not in r]
    agg = {}
    for task in sorted({r["task"] for r in ok}):
        rs = [r for r in ok if r["task"] == task]
        agg[task] = {
            "n": len(rs),
            "lang_ok": round(sum(r["lang_ok"] for r in rs) / len(rs), 3),
            "fmt_ok": round(sum(r["fmt_ok"] for r in rs) / len(rs), 3),
            "faith": round(sum(r["faith"] for r in rs) / len(rs), 2),
            "cover": round(sum(r["cover"] for r in rs) / len(rs), 2),
        }
    Path(args.out).write_text(json.dumps({"agg": agg, "results": results}, ensure_ascii=False, indent=1))
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
