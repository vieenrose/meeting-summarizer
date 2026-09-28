"""Keep only the substantive corrections from distill/onpolicy_correct.py, and build DPO pairs.

A reviewed sample of the corrector's edits mixed real fixes (核心理術 -> 核心技術, 凍結 -> 解凍) with
rewrites, dropped examples and a few damaging edits (67 -> 七六七). An edit counts only if it changes
something the evaluation judge counts as a contradiction: a figure's value, a status word, a
negation, or the speaker label between two specific roles. A generic label made specific is
rejected (the gold never does it), and so is an edit that shortens the note by more than a third.

The preferred side is the student's own notes with only the kept edits applied, so the pair
differs exactly where a real error was fixed.
"""
import argparse
import json
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.numerals import values  # noqa: E402
from summarizer.pipeline import parse_turn  # noqa: E402

STATUS = ("通過", "保留", "凍結", "解凍", "刪減", "刪除", "建議", "決議", "同意", "反對", "不予", "協商", "撤回", "照列", "延宕", "提前")
NEG = re.compile(r"[不未無非沒]")
GENERIC = ("發言者", "某委員", "委員")


def speaker(t):
    m = re.match(r"^([^:：]{1,15})[:：]", t)
    return m.group(1).strip() if m else ""


def substantive(old, new):
    if len(new) < 0.67 * len(old):
        return None
    so, sn = speaker(old), speaker(new)
    if so != sn:
        if so in GENERIC and sn not in GENERIC:
            return None
        return "speaker"
    if values(old) != values(new):
        return "figure"
    if {s for s in STATUS if s in old} != {s for s in STATUS if s in new}:
        return "status"
    if len(NEG.findall(old)) != len(NEG.findall(new)):
        return "negation"
    return None


def render(tag, ts, t):
    return f"- ({tag}) [{ts}] {t}" if tag else f"- [{ts}] {t}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/train/onpolicy_corrected.jsonl")
    ap.add_argument("--out", default="data/train/onpolicy_pairs.jsonl")
    ap.add_argument("--kinds", default="status,negation")
    args = ap.parse_args()
    kinds, n_pairs, edits = {}, 0, []
    with open(args.out, "w", encoding="utf-8") as f:
        for line in open(args.inp, encoding="utf-8"):
            d = json.loads(line)
            s, c = parse_turn(d["student"]).notes, parse_turn(d["corrected"]).notes
            by_ts = {}
            for tag, ts, t in c:
                by_ts.setdefault(ts, (tag, t))
            chosen, changed = [], 0
            for tag, ts, t in s:
                new = by_ts.get(ts)
                k = substantive(t, new[1]) if new else None
                # Reviewed against the transcript (50 edits): status/negation edits were right 21/25,
                # figure edits 7/25 -- the corrector "fixes" ASR-literal figures toward the real
                # world (2015 -> 115) or invents values (1.17 -> 1014). Speaker edits were noisiest.
                if k and k in args.kinds.split(","):
                    chosen.append((tag, ts, new[1]))
                    changed += 1
                    kinds[k] = kinds.get(k, 0) + 1
                    edits.append({"session": d["session"], "kind": k, "old": t, "new": new[1]})
                else:
                    chosen.append((tag, ts, t))
            if changed:
                f.write(json.dumps({"session": d["session"], "window": d["window"], "prompt": d["messages"],
                                    "chosen": [{"role": "assistant", "content": "\n".join(render(*x) for x in chosen)}],
                                    "rejected": [{"role": "assistant", "content": "\n".join(render(*x) for x in s)}]},
                                   ensure_ascii=False) + "\n")
                n_pairs += 1
    json.dump(edits, open(args.out.replace(".jsonl", "_edits.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{n_pairs} pairs, {len(edits)} kept edits {kinds} -> {args.out}")


if __name__ == "__main__":
    main()
