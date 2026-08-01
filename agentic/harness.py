"""Memory-augmented harness: summarize a long transcript with a short-context SLM.

Design follows what the literature actually supports for sub-1B models (see
docs/RESEARCH-agentic-memory.md), which is *not* "let the model run an agent loop":

- **The harness owns structure; the model owns content.** Merging, deduping and section
  assembly are deterministic Python/Kotlin. The model is only ever asked to write a small
  bounded piece of text about a chunk it can see. This sidesteps the documented
  "Silent Failure" mode — 30% malformed memory *writes* at 3B — because the model never
  emits a memory operation, only content.
- **Typed memory, not a free-text running summary.** Free-text state loses recall
  66.9→26.8 on a weak model; typed state cuts that loss ~5x.
- **Verbatim anchors.** Every memory item carries the chunk id and timestamp it came from,
  so a later pass can re-read the evidence rather than trusting the note.
- **Bounded passes.** On ARM an 8k prefill costs ~63s; every re-read is a fresh prefill.
  Total calls are O(n_chunks) + a fixed constant, never model-decided.
- **Near-greedy decoding**, which is measured to arrest factual drift.

Rungs implemented:
  map      : per-chunk typed notes (no cross-chunk state) -> deterministic merge
  running  : per-chunk notes conditioned on the running typed state (rung 1)
  reread   : map, then one bounded re-read pass over chunks flagged as contested (rung 2)
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "distill"))

from openai import AsyncOpenAI
from transformers import AutoTokenizer

SECTIONS = ["SUMMARY", "DECISIONS", "ACTIONS", "OPEN", "TOPICS"]

CHUNK_PROMPT_EN = """Read this section of a meeting transcript and write notes about it.

Use EXACTLY this format, and put the timestamp of the supporting line in square brackets
at the end of every bullet:
SUMMARY:
- point [0:12]
DECISIONS:
- decision made in this section [1:03]
ACTIONS:
- owner: what they will do [2:20]
OPEN:
- unresolved question [3:04]
TOPICS:
- topic discussed

Only write what this section actually says. If a section has nothing, put exactly "-".
No preamble.

Transcript section {cid}:
{chunk}"""

CHUNK_PROMPT_ZH = """請閱讀以下會議逐字稿的其中一段，並針對這一段做筆記。

請「完全」使用這個格式，每一點結尾用方括號標註依據該點的時間戳記：
SUMMARY:
- 重點 [0:12]
DECISIONS:
- 這一段做成的決策 [1:03]
ACTIONS:
- 負責人: 要做的事 [2:20]
OPEN:
- 未解決的問題 [3:04]
TOPICS:
- 討論的議題

只寫這一段真的有講到的內容。若某區段沒有內容，該行只寫「-」。不要前言。

逐字稿片段 {cid}:
{chunk}"""

# The compress step is where recurrent summarization is documented to break, so it is
# scoped to ONE section at a time with the source lines in view.
COMPRESS_EN = """These are notes for the "{section}" section of one meeting, collected from
different parts of the transcript. Merge them into at most {n} bullets.

Rules: keep the [timestamp] anchors. If two bullets disagree, keep the LATER timestamp and
drop the earlier one — a later statement supersedes an earlier one. Do not invent anything.
Output only the bullets.

{items}"""

COMPRESS_ZH = """以下是同一場會議「{section}」區段的筆記，來自逐字稿的不同部分。請合併成最多 {n} 點。

規則：保留 [時間戳記]。若兩點互相矛盾，保留「時間較晚」的那一點並刪去較早的——後面的說法會取代前面的。不要杜撰。只輸出條列。

{items}"""

CONTESTED_EN = """Below are notes from one meeting. Some may contradict each other
(e.g. something approved and later rejected).

List ONLY the chunk ids that need re-reading to resolve a contradiction, as a comma-separated
list like: c2,c7
If nothing is contradictory, output exactly: none

{items}"""


def clock_to_sec(ts: str) -> int:
    p = [int(x) for x in ts.split(":")]
    return p[0] * 3600 + p[1] * 60 + p[2] if len(p) == 3 else p[0] * 60 + p[1]


def parse_notes(text: str) -> dict:
    """Split model output into typed sections. Malformed input degrades to empty, never
    to a crash — the harness must survive a bad generation."""
    out = {s: [] for s in SECTIONS}
    cur = None
    for line in (text or "").split("\n"):
        line = line.strip()
        m = re.match(r"^([A-Z]+):\s*$", line)
        if m and m.group(1) in out:
            cur = m.group(1)
            continue
        if cur and line.startswith("- ") and line[2:].strip() not in ("", "-"):
            out[cur].append(line[2:].strip())
    return out


ANCHOR = re.compile(r"\[(\d+:\d{2}(?::\d{2})?)\]\s*$")


def anchor_sec(item: str) -> int:
    m = ANCHOR.search(item)
    return clock_to_sec(m.group(1)) if m else -1


def render(state: dict, title: str = "") -> str:
    lines = [f"TITLE: {title}"] if title else []
    for s in SECTIONS:
        lines.append(f"{s}:")
        items = state.get(s) or []
        lines.extend(f"- {i}" for i in items) if items else lines.append("-")
    return "\n".join(lines)


# --------------------------------------------------------------------------- runtime
class Runner:
    def __init__(self, url, model, temp=0.0):
        self.c = AsyncOpenAI(base_url=url, api_key="x", timeout=1800.0)
        self.model = model
        self.temp = temp
        self.calls = 0
        self.prompt_tokens = 0
        self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

    async def gen(self, prompt: str, max_tokens: int) -> str:
        self.calls += 1
        self.prompt_tokens += len(self.tok.encode(prompt))
        r = await self.c.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=self.temp,
            extra_body={"chat_template_kwargs": {"enable_thinking": False},
                        "top_k": 1, "presence_penalty": 1.0})
        return (r.choices[0].message.content or "").strip()


def chunk_lines(transcript: str, tok, budget: int) -> list:
    """Split on line boundaries so a `[mm:ss] speaker: text` record is never cut in half."""
    chunks, cur, n = [], [], 0
    for line in transcript.split("\n"):
        t = len(tok.encode(line)) + 1
        if cur and n + t > budget:
            chunks.append("\n".join(cur)); cur, n = [], 0
        cur.append(line); n += t
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def merge_deterministic(per_chunk: list) -> dict:
    """Structural merge done in code, not by the model: concatenate by section, drop exact
    duplicates, order by anchor time. Later contradictions survive because ordering is by
    timestamp and the compress step is told later supersedes earlier."""
    state = {s: [] for s in SECTIONS}
    for notes in per_chunk:
        for s in SECTIONS:
            for item in notes[s]:
                if item not in state[s]:
                    state[s].append(item)
    for s in SECTIONS:
        state[s].sort(key=anchor_sec)
    return state


async def summarize(runner, transcript, lang, mode="reread", chunk_budget=4000,
                    max_bullets=(5, 5, 6, 4, 6), max_reread=2):
    tok = runner.tok
    chunks = chunk_lines(transcript, tok, chunk_budget)
    zh = lang == "zh-TW"
    cp = CHUNK_PROMPT_ZH if zh else CHUNK_PROMPT_EN

    per_chunk = []
    running = {s: [] for s in SECTIONS}
    for i, ch in enumerate(chunks):
        prompt = cp.format(cid=f"c{i}", chunk=ch)
        if mode == "running" and any(running[s] for s in SECTIONS):
            carry = render(running)
            prompt = (("目前的筆記：\n" if zh else "Notes so far:\n") + carry +
                      ("\n\n" ) + prompt)
        notes = parse_notes(await runner.gen(prompt, 420))
        for s in SECTIONS:
            notes[s] = [f"{x} [c{i}]" if f"[c{i}]" not in x else x for x in notes[s]]
        per_chunk.append(notes)
        if mode == "running":
            running = merge_deterministic(per_chunk)

    state = merge_deterministic(per_chunk)

    if mode == "reread" and len(chunks) > 1:
        flat = "\n".join(f"- {i}" for s in ("DECISIONS", "ACTIONS") for i in state[s])
        if flat.strip():
            ans = await runner.gen(CONTESTED_EN.format(items=flat), 40)
            ids = [c for c in re.findall(r"c(\d+)", ans) if int(c) < len(chunks)][:max_reread]
            for cid in ids:
                i = int(cid)
                notes = parse_notes(await runner.gen(cp.format(cid=f"c{i}", chunk=chunks[i]), 420))
                for s in SECTIONS:
                    notes[s] = [f"{x} [c{i}]" if f"[c{i}]" not in x else x for x in notes[s]]
                per_chunk.append(notes)
            state = merge_deterministic(per_chunk)

    # compress each section separately, with the anchors in view
    comp = COMPRESS_ZH if zh else COMPRESS_EN
    final = {}
    for s, n in zip(SECTIONS, max_bullets):
        items = state[s]
        if not items:
            final[s] = []
        elif len(items) <= n:
            final[s] = items
        else:
            body = "\n".join(f"- {i}" for i in items)
            out = await runner.gen(comp.format(section=s, n=n, items=body), 420)
            got = [l.strip()[2:].strip() for l in out.split("\n") if l.strip().startswith("- ")]
            final[s] = got[:n] or items[:n]
    return final, len(chunks)
