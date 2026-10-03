"""The two conversion calls VoxSumDroid makes at the end of a meeting, reproduced exactly
(core/reader/ReaderLane.title / .prose, v0.45.1): the whole journal, rendered as
ReaderProtocol.render writes it, in one user turn with no system turn; generation stops at <turn|>.

These are style conversions, not syntheses: the output must say nothing the notes do not say.
distill/convert_targets.py distills them from the teacher; eval/judge_prose_notes.py checks the
prose against the notes it was written from.
"""
import re

from eval.realtime_agent import render

TITLE_MAX, PROSE_MAX = 48, 600


def title_prompt(notes):
    return "以下是一場會議的筆記：\n\n" + "\n".join(map(render, notes)) + "\n\n為這場會議寫一個標題，不超過 20 個字。只輸出標題。"


def prose_prompt(notes):
    return ("以下是一場會議的筆記：\n\n" + "\n".join(map(render, notes)) +
            "\n\n根據這些筆記，用連貫的段落寫一份會議摘要（不要條列、不要標題），"
            "說明討論了什麼、決定了什麼、誰要做什麼、還有什麼沒解決。"
            "只寫筆記裡有的內容；提到某件事時在句尾附上筆記的時間，例如 [1:23]。")


CITE = re.compile(r"\[(\d+:\d{2}(?::\d{2})?)\]")


def clean_title(raw):
    """ReaderLane.title's sanitising: first non-blank line, stripped, at most 40 characters."""
    line = next((l for l in raw.split("<turn|>")[0].splitlines() if l.strip()), "")
    return line.strip().strip("「」\"*# ")[:40]


def clean_prose(raw, notes):
    """ReaderLane.prose's sanitising: citations not in the journal are removed, bullet and header
    lines dropped, paragraphs joined by a blank line."""
    known = {n["ts"] for n in notes}
    text = CITE.sub(lambda m: m.group(0) if m.group(1) in known else "", raw.split("<turn|>")[0])
    lines = [l.strip().lstrip("#").strip() for l in text.splitlines()]
    return "\n\n".join(l for l in lines if l and not l.startswith(("-", "*")))


def form(raw, notes):
    """How well a raw prose reply keeps the requested form, before any sanitising."""
    known = {n["ts"] for n in notes}
    body = raw.split("<turn|>")[0]
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    cites = CITE.findall(body)
    sents = [s for s in re.split(r"(?<=[。！？])", body.replace("\n", "")) if len(s.strip()) >= 6]
    return {"bullets": sum(l.startswith(("-", "*", "•")) or bool(re.match(r"^\d+[.、]", l)) for l in lines),
            "headers": sum(l.startswith("#") or (l.startswith("**") and l.endswith("**")) for l in lines),
            "citations": len(cites), "bad_citations": sum(c not in known for c in cites),
            "sentences": len(sents), "cited_sentences": sum(bool(CITE.search(s)) for s in sents),
            "chars": len(CITE.sub("", body).strip())}


def compact_notes(notes, budget_chars):
    """A journal that fits a small context (LiteRT-LM at 4k tokens): decisions first, then open issues,
    then actions, newest first within each, then the most recent other notes, kept in chronological
    order. The same priorities as the reading restart (docs/voxsumdroid-integration.md §4.4)."""
    key = {"DECISION": 0, "OPEN-ISSUE": 1, "ACTION": 2}
    order = sorted(range(len(notes)), key=lambda i: (key.get((notes[i].get("tag") or "").upper(), 3), -i))
    chosen, used = set(), 0
    for i in order:
        t = len(notes[i].get("text", "")) + 24
        if used + t <= budget_chars:
            chosen.add(i)
            used += t
    return [notes[i] for i in sorted(chosen)]
