"""targets.jsonl (teacher generations) -> filtered ChatML SFT dataset.

Filters (ported from VoxSumDroid tools/validate_llm.py where applicable):
  - output language/script must match the effective target (han for zh-TW without
    kana/hangul; latin for en without CJK)
  - stray think blocks / markdown headings / preamble lines are rejected (the app
    post-cleans, but the student should natively comply)
  - title: <= 8 words (en) / <= 16 chars (zh), single line
  - length caps per task, exact-dup removal
Split: follows the corpus split of the source meeting (train/val/test).

Output: data/sft/{train,val,test}.jsonl with {"messages": [...]} rows.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "sft"
OUT.mkdir(exist_ok=True)

SYSTEM = "You are a helpful assistant."


def kana(s):
    return any("ぁ" <= c <= "ゟ" or "゠" <= c <= "ヺ" or "ー" <= c <= "ヿ" for c in s)


def hangul(s):
    return any("가" <= c <= "힣" or "ᄀ" <= c <= "ᇿ" for c in s)


def han(s):
    return any("一" <= c <= "鿿" for c in s)


def latin(s):
    return any("a" <= c.lower() <= "z" for c in s)


def lang_ok(lang: str, out: str) -> bool:
    if out.strip() == "-":
        return True
    if lang == "zh-TW":
        return han(out) and not kana(out) and not hangul(out)
    if lang == "en":
        return latin(out) and not han(out) and not kana(out) and not hangul(out)
    return True


BAD_PATTERNS = re.compile(
    r"<think>|</think>|^#{1,6}\s|^(here('s| is| are)|sure|okay|certainly|of course)\b|以下是|好的[，,]",
    re.IGNORECASE | re.MULTILINE,
)


def effective_lang(d: dict) -> str:
    tgt = d["meta"].get("tgt")
    return tgt if tgt else d["lang"]


_BULLET = re.compile(r"^(\s*)[*•·–]\s+", re.MULTILINE)


def normalize(c: str) -> str:
    """Teacher outputs vary in bullet glyphs; train the student on plain '- ' dashes
    (the app's cleanSummary then renders them as '• ')."""
    c = _BULLET.sub(r"\1- ", c)
    return re.sub(r"[ \t]+$", "", c, flags=re.MULTILINE).strip()


NOTES_KEYS = ["TITLE:", "SUMMARY:", "DECISIONS:", "ACTIONS:", "OPEN:", "TOPICS:"]
# teacher failure mode: echoing the template placeholder / "none" instead of "-"
NOTES_PLACEHOLDER = re.compile(
    r"owner: task|name: what they|due: deadline|負責人[:：]\s*(工作內容|無)|"
    r"期限[:：]\s*(無|沒有)|^- (none|n/a|無|沒有)[。.]?$",
    re.IGNORECASE | re.MULTILINE,
)


def notes_ok(c: str) -> bool:
    """Strict wire-format check for the v2 NOTES task (docs/OUTPUT-FORMAT.md)."""
    if NOTES_PLACEHOLDER.search(c):
        return False
    lines = [l for l in c.split("\n") if l.strip()]
    if not lines or not lines[0].startswith("TITLE:"):
        return False
    title = lines[0][6:].strip()
    if not title or len(title) > 90 or len(title.split()) > 10:
        return False
    keys = [l.split()[0] for l in lines if re.match(r"^[A-Z]+:", l)]
    if keys != NOTES_KEYS:
        return False
    for l in lines[1:]:
        if re.match(r"^[A-Z]+:", l):
            if l.split(":", 1)[1].strip():   # only TITLE carries inline content
                return False
        elif not (l.startswith("- ") or l.strip() == "-"):
            return False
    return True


MIN_FAITH = 4   # judged targets below this are dropped (see distill/faith_filter.py)


def keep(d: dict) -> bool:
    # Faithfulness gate: v1/v2 trained on unjudged teacher output, and ~17% of it scored
    # faith<=3 with outright polarity inversions in it. Training on those taught the
    # student to invert. A judged record must clear the bar; unjudged records pass
    # through so the pipeline still works before/without a judging run.
    j = d.get("judge")
    if j:
        if j.get("invert"):
            return False
        if j.get("faith", 0) and j["faith"] < MIN_FAITH:
            return False
    c = d["completion"]
    if not c or len(c) > 2600:
        return False
    if d["task"] == "notes" and not notes_ok(c):
        return False
    if BAD_PATTERNS.search(c):
        return False
    if not lang_ok(effective_lang(d), c):
        return False
    if d["task"] == "title":
        if "\n" in c or c.count('"') or len(c) > 90:
            return False
        lang = effective_lang(d)
        if lang == "en" and len(c.split()) > 10:
            return False
        if lang == "zh-TW" and len(c) > 20:
            return False
    return True


def main(*targets_paths: str) -> None:
    seen = set()
    outs = {s: (OUT / f"{s}.jsonl").open("w", encoding="utf-8") for s in ("train", "val", "test")}
    stats = Counter()
    lines = (line for p in targets_paths for line in Path(p).open())
    for line in lines:
        d = json.loads(line)
        # Single-pass redesign (2026-07-29): the app dropped map-reduce, so "reduce"
        # prompts never occur at runtime — exclude them. Run-1 "map"/"single" records
        # are valid single-pass examples over (partial) transcripts; title/shrink stay.
        if d["phase"] == "reduce":
            stats["dropped_reduce"] += 1
            continue
        d["completion"] = normalize(d["completion"])
        stats["total"] += 1
        # Register the identity BEFORE filtering: the judged file and the raw file hold
        # the same records, so a target dropped for low faith would otherwise be let
        # back in by its unjudged twin — silently nullifying the faithfulness gate.
        key = hash((d["prompt"], d["completion"]))
        if key in seen:
            stats["dup"] += 1
            continue
        seen.add(key)
        if not keep(d):
            stats["filtered"] += 1
            j = d.get("judge") or {}
            if j.get("invert"):
                stats["dropped_inverted"] += 1
            elif j.get("faith", 0) and j["faith"] < MIN_FAITH:
                stats["dropped_lowfaith"] += 1
            continue
        row = {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": d["prompt"]},
                {"role": "assistant", "content": d["completion"]},
            ],
            "task": d["task"], "phase": d["phase"], "source": d["source"],
            "lang": effective_lang(d), "meeting_id": d["meeting_id"],
        }
        split = d["split"] if d["split"] in outs else "train"
        outs[split].write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[f"kept_{split}"] += 1
        stats[f"task_{d['task']}"] += 1
    for f in outs.values():
        f.close()
    for k, v in sorted(stats.items()):
        print(f"{k}: {v}")


if __name__ == "__main__":
    # judged file first: it carries the same records plus judge verdicts, and dedup
    # keeps the first occurrence, so judged copies win over their unjudged twins
    default = [str(HERE.parent / "distill/targets_judged.jsonl"),
               str(HERE.parent / "distill/targets_single.jsonl"),
               str(HERE.parent / "distill/targets.jsonl")]
    main(*(sys.argv[1:] or default))
