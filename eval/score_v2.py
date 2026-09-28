"""Mechanical scores for student runs on the v2 held-out sessions, one line per run directory.

  summary pass  sessions whose numbered points pass the validator the gold passed
  prose pass    sessions whose prose passes the prose gate (length, >=6 valid citations, all
                three thirds, no Simplified characters)
  empty         windows that produced no notes, as a share of all windows
  notes         notes per session, against the gold's
  recall        gold note-fact recall: the share of the gold notes' figures and names that appear
                in the student's notes
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.write_prose import check as prose_check  # noqa: E402
from eval.compare_students import facts  # noqa: E402
from eval.validate_teacher import check  # noqa: E402
from summarizer.ingest import parse_line  # noqa: E402


def score_dir(d, split="data/split_v2.json", gold_dir="runs/v2/w4000", transcripts="data/v2/transcripts"):
    n = sp = pp = empty = windows = notes = gold_notes = 0
    hit = total = 0
    for sid in json.load(open(split))["heldout"]:
        path = os.path.join(d, sid + ".json")
        if not os.path.exists(path):
            continue
        lines = [parse_line(l) for l in open(os.path.join(transcripts, sid + ".txt"),
                                              encoding="utf-8").read().splitlines() if l.strip()]
        run = json.load(open(path, encoding="utf-8"))
        gold = json.load(open(os.path.join(gold_dir, sid + ".json"), encoding="utf-8"))
        n += 1
        sp += not check(run, lines, 0, 650, 0)
        pp += bool(run.get("prose")) and not prose_check(run["prose"], lines)
        windows += run.get("windows") or 0
        empty += len(run.get("windows_without_notes") or [])
        notes += len(run["notes"])
        gold_notes += len(gold["notes"])
        g = facts(" ".join(x["text"] for x in gold["notes"]))
        s = re.sub(r"\s|,", "", " ".join(x["text"] for x in run["notes"]))
        hit += sum(f in s for f in g)
        total += len(g)
    if not n:
        return f"{d}: no sessions"
    return (f"{d}: n={n} summary pass {sp/n:.0%}  prose pass {pp/n:.0%}  empty windows "
            f"{empty/max(1, windows):.0%}  notes/session {notes/n:.1f} (gold {gold_notes/n:.1f})  "
            f"note-fact recall {hit/max(1, total):.2f}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(score_dir(d))
