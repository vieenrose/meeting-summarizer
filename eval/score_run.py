"""One line per run: the numbers used to pick between checkpoints.

  pass      sessions passing the same validator the teacher's gold passed
  reward    mean summary reward (distill/rewards.py), the GRPO objective
  zeros     summaries that hit a hard-zero rule (copied prompt, run-on point, no points)
  notes     gold note-fact recall -- how much of the teacher's notes' figures and names the
            student's notes contain; GRPO on summaries should not erode this
  empty     windows that produced no notes
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.rewards import score_summary  # noqa: E402
from eval.compare_students import facts  # noqa: E402
from eval.validate_teacher import check  # noqa: E402
from summarizer.ingest import parse_line  # noqa: E402


def score_dir(d, split="data/split.json", gold_dir="runs/gold/w4000", transcripts="data/transcripts"):
    sessions = json.load(open(split))["heldout"]
    n = passed = zeros = empty = g = r = 0
    reward = 0.0
    for sid in sessions:
        path = os.path.join(d, sid + ".json")
        if not os.path.exists(path):
            continue
        lines = [parse_line(l) for l in open(os.path.join(transcripts, sid + ".txt"),
                                              encoding="utf-8").read().splitlines() if l.strip()]
        gold = json.load(open(os.path.join(gold_dir, sid + ".json"), encoding="utf-8"))
        run = json.load(open(path, encoding="utf-8"))
        s, detail = score_summary("\n".join(run["summary"]), lines, gold["summary"], detail=True)
        n += 1
        reward += s
        zeros += "reason" in detail
        passed += not check(run, lines, 0, 650, 0)
        empty += len(run.get("windows_without_notes") or [])
        gf = facts(" ".join(x["text"] for x in gold["notes"]))
        sn = re.sub(r"\s|,", "", " ".join(x["text"] for x in run["notes"]))
        g += len(gf)
        r += sum(1 for f in gf if f in sn)
    return {"n": n, "pass": passed, "reward": reward / max(1, n), "zeros": zeros,
            "notes_recall": r / max(1, g), "empty": empty}


def main():
    for spec in sys.argv[1:]:
        label, d = spec.split("=", 1) if "=" in spec else (os.path.basename(spec.rstrip("/")), spec)
        s = score_dir(d)
        print(f"{label:28} pass {s['pass']}/{s['n']}  reward {s['reward']:.3f}  zeros {s['zeros']}  "
              f"notes-recall {100 * s['notes_recall']:.0f}%  empty {s['empty']}", flush=True)


if __name__ == "__main__":
    main()
