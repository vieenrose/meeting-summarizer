"""Distillation generation — SINGLE-PASS mode (v2, 2026-07-29).

The app is dropping map-reduce entirely: the model gets the WHOLE transcript in one
call with an extended context (32k = Qwen3-0.6B native max; a zh-TW hour ≈ 26k tokens).
Every training example is therefore (task prompt over full transcript -> teacher answer),
plus the two summary-post-processing tasks that survive the redesign (title-from-summary,
shrink of an over-long summary).

Meetings whose prompt would exceed the token budget are skipped (logged) — no folding.
Token counts use the real Qwen3 tokenizer, not a chars-per-token heuristic.

Usage:
  python generate.py --out targets_single.jsonl \
      --caps dialogsum=1500,qmsum=232,vcsum=1359,meetingbank=800
Resume-safe on (meeting_id, variant) keys.
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai import AsyncOpenAI
from transformers import AutoTokenizer

import tasks as T
from configs.transcript_format import SpeakerStyle, render_transcript, strip_tags

BASE_URL = "http://127.0.0.1:8088/v1"
MODEL = "teacher"
N_CTX = 32768          # student & runtime context target
PROMPT_MARGIN = 96     # chat-template + safety, mirrors the app's old margin
SEM = asyncio.Semaphore(8)   # llama-server runs 4 × 32k slots; small queue on top

client = AsyncOpenAI(base_url=BASE_URL, api_key="x", timeout=1800.0)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def ntok(s: str) -> int:
    return len(tok.encode(s))


def too_long(summary: str) -> bool:  # SummaryText.tooLong
    return sum(1 for l in summary.split("\n") if l.strip()) > 12 or len(summary) > 1200


async def chat(prompt: str, max_tokens: int, temperature: float = 0.3) -> str:
    async with SEM:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens + 512,
                    temperature=temperature,
                    top_p=0.9,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                return (r.choices[0].message.content or "").strip()
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(3 * (attempt + 1))
    return ""


def rec(out, meeting, variant, task, phase, prompt, completion, **meta):
    if not completion.strip():
        return
    out.write(json.dumps({
        "meeting_id": meeting["id"], "source": meeting["source"], "split": meeting["split"],
        "variant": variant, "task": task, "phase": phase,
        "lang": meeting["lang"], "prompt": prompt, "completion": completion.strip(),
        "meta": meta,
    }, ensure_ascii=False) + "\n")
    out.flush()


def fits(prompt: str, out_tokens: int) -> bool:
    return ntok(prompt) + out_tokens + PROMPT_MARGIN <= N_CTX


# ------------------------------------------------------------------- single-pass tasks
async def run_summarize(out, meeting, variant, transcript, style_name, tgt, zh_instr, instr):
    m_ins, r_ins, m_tok, r_tok = T.STYLES[style_name]
    clause = T.lang_clause(T.TARGET_LANG[tgt] if tgt else None)
    p = (T.MAP_TEMPLATE_ZH % transcript if zh_instr
         else T.MAP_TEMPLATE % (instr + clause, m_ins, transcript))
    if not fits(p, m_tok):
        return "skip"
    summary = await chat(p, m_tok)
    rec(out, meeting, variant, "summarize", "single", p, summary, style=style_name, tgt=tgt)
    if not summary:
        return None

    if too_long(summary):
        sp = (T.SHRINK_TEMPLATE_ZH % summary if zh_instr
              else T.SHRINK_TEMPLATE % (instr + clause, r_ins, summary))
        shrunk = await chat(sp, r_tok)
        rec(out, meeting, variant, "summarize", "shrink", sp, shrunk, style=style_name, tgt=tgt)
        summary = shrunk or summary

    tp = (T.TITLE_TEMPLATE_ZH % summary if zh_instr
          else T.TITLE_TEMPLATE % (clause, summary))
    title = await chat(tp, 24)
    rec(out, meeting, variant, "title", "title", tp, title, style=style_name, tgt=tgt)
    return summary


async def run_actions(out, meeting, variant, transcript, tgt):
    clause = T.action_lang_clause(T.TARGET_LANG[tgt] if tgt else None)
    p = T.ACTION_MAP_TEMPLATE % (clause, transcript)
    if not fits(p, 384):
        return "skip"
    a = await chat(p, 384)
    rec(out, meeting, variant, "actions", "single", p, a, tgt=tgt)


async def run_insight(out, meeting, variant, transcript, task_name, tgt):
    en_p, zh_p, max_tok = T.INSIGHT_TASKS[task_name]
    zh = (tgt or meeting["lang"]) == "zh-TW"
    clause = "" if tgt is None else T.lang_clause(T.TARGET_LANG[tgt])
    p = zh_p % transcript if zh else en_p % (clause, transcript)
    if not fits(p, max_tok):
        return "skip"
    a = await chat(p, max_tok)
    rec(out, meeting, variant, task_name, "single", p, a, tgt=tgt)


# --------------------------------------------------------------------------- pipeline
def plan_variants(meeting, rng):
    lang = meeting["lang"]
    has_speakers = any(u["speaker"] for u in meeting["utterances"])
    spk_kinds = ["stags", "stags", "names"] if has_speakers else ["none"]
    cross = "zh-TW" if lang == "en" else "en"

    variants = []
    # two summarize styles per meeting: one auto-lang, one that may be cross-lingual
    for style, tgt_opts in [
        (rng.choice(list(T.STYLES)), [None]),
        (rng.choice(list(T.STYLES)), [None, cross, cross]),
    ]:
        tgt = rng.choice(tgt_opts)
        zh_instr = (lang == "zh-TW" and tgt is None) or tgt == "zh-TW"
        variants.append({
            "kind": "summarize", "spk": rng.choice(spk_kinds), "style": style, "tgt": tgt,
            "zh_instr": zh_instr,
            "instr": rng.choice(T.USER_INSTRUCTIONS_ZH if zh_instr else T.USER_INSTRUCTIONS_EN),
        })
    variants.append({"kind": "actions", "spk": rng.choice(spk_kinds),
                     "tgt": rng.choice([None, None, cross])})
    for t in rng.sample(list(T.INSIGHT_TASKS), k=4):
        variants.append({"kind": "insight", "task": t, "spk": rng.choice(spk_kinds),
                         "tgt": rng.choice([None, None, cross])})
    return variants


def variant_key(v):
    return "single|" + json.dumps({k: v[k] for k in sorted(v) if k != "instr"}, sort_keys=True)


async def process_meeting(out, meeting, done, skipped):
    rng = random.Random(meeting["id"] + "/v2")
    for v in plan_variants(meeting, rng):
        key = f"{meeting['id']}|{variant_key(v)}"
        if key in done:
            continue
        transcript = render_transcript(meeting, SpeakerStyle(v["spk"]))
        if not transcript.strip():
            continue
        if v["spk"] == "none" and rng.random() < 0.3:
            transcript = strip_tags(transcript)
        try:
            if v["kind"] == "summarize":
                r = await run_summarize(out, meeting, key, transcript, v["style"], v["tgt"],
                                        v["zh_instr"], v["instr"])
            elif v["kind"] == "actions":
                r = await run_actions(out, meeting, key, transcript, v["tgt"])
            else:
                r = await run_insight(out, meeting, key, transcript, v["task"], v["tgt"])
            if r == "skip":
                skipped[0] += 1
        except Exception as e:
            print(f"ERR {key}: {e}", file=sys.stderr)
        done.add(key)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["dialogsum", "qmsum", "vcsum", "meetingbank"])
    ap.add_argument("--out", default=str(Path(__file__).parent / "targets_single.jsonl"))
    ap.add_argument("--caps", default="dialogsum=1500,qmsum=232,vcsum=1359,meetingbank=800")
    ap.add_argument("--parallel-meetings", type=int, default=6)
    args = ap.parse_args()

    caps = dict(kv.split("=") for kv in args.caps.split(","))
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                done.add(json.loads(line)["variant"])
            except Exception:
                pass
        print(f"resume: {len(done)} variants already present", flush=True)

    norm = Path(__file__).parent.parent / "data/normalized"
    meetings = []
    for src in args.sources:
        rows = [json.loads(l) for l in (norm / f"{src}.jsonl").open()]
        # long-meeting emphasis: sort MeetingBank by length desc before capping, so the
        # hour-long tail is guaranteed in, not a lottery win
        if src == "meetingbank":
            rows.sort(key=lambda m: -sum(len(u["text"]) for u in m["utterances"]))
        else:
            random.Random(src).shuffle(rows)
        meetings.extend(rows[: int(caps.get(src, len(rows)))])
    random.Random(0).shuffle(meetings)
    print(f"{len(meetings)} meetings to process", flush=True)

    out = out_path.open("a", encoding="utf-8")
    skipped = [0]
    sem = asyncio.Semaphore(args.parallel_meetings)

    async def guarded(m):
        async with sem:
            await process_meeting(out, m, done, skipped)

    batch = [guarded(m) for m in meetings]
    for i in range(0, len(batch), 100):
        await asyncio.gather(*batch[i:i + 100])
        try:
            print(f"progress: {min(i + 100, len(batch))}/{len(batch)} meetings"
                  f" (ctx-skipped variants: {skipped[0]})", flush=True)
        except OSError:
            pass
    out.close()


if __name__ == "__main__":
    asyncio.run(main())
