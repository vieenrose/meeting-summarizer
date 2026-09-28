"""Post-hoc filters on a student's map-step notes, to measure what each removes before reduce.

  window       drop a note whose figures (compared by value: 一千三百萬 == 1300萬) are not all
               present in the window the note was written from. Cheap and deterministic, so it can
               run on the phone. It removes invented figures; it cannot catch a real figure pinned
               to the wrong article.
  consistency  keep a greedy note only if each of its figures and names also appears in at least
               K of N extra samples for the same window (distill/rft-style sampling, done by
               eval/sample_windows.py). Invented detail tends not to recur across samples.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.numerals import values  # noqa: E402
from distill.rewards import NAME, _norm  # noqa: E402


def facts(text, names=False):
    """Figures by value. Names are opt-in: the NAME pattern takes any 2-3 characters before 委員,
    which on gold notes yields fragments like 議三項 or 針對 far more often than names."""
    return values(text) | ({("name", n) for n in NAME.findall(_norm(text))} if names else set())


def window_texts(run):
    return {e["window"]: e["messages"][-1]["content"].split("TRANSCRIPT\n", 1)[-1]
            for e in run["log"] if e.get("window") and not e.get("retry")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["window", "consistency"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", help="consistency: JSON {session: {window: [sample notes texts]}}")
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()
    samples = json.load(open(args.samples)) if args.samples else {}
    os.makedirs(args.out, exist_ok=True)
    kept = total = 0
    for path in sorted(glob.glob(os.path.join(args.run_dir, "*.json"))):
        run = json.load(open(path, encoding="utf-8"))
        sid = os.path.basename(path)[:-5]
        wins = window_texts(run)
        keep = []
        for n in run["notes"]:
            f = facts(n["text"])
            if args.mode == "window":
                have = facts(wins.get(n["window"], ""))
                ok = all(x in have for x in f)
            else:
                others = [facts(s) for s in samples.get(sid, {}).get(str(n["window"]), [])]
                ok = all(sum(x in o for o in others) >= args.k for x in f) if others else True
            keep.append(ok)
        run["notes_dropped"] = [n for n, k in zip(run["notes"], keep) if not k]
        run["notes"] = [n for n, k in zip(run["notes"], keep) if k]
        kept, total = kept + sum(keep), total + len(keep)
        json.dump(run, open(os.path.join(args.out, sid + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{args.mode}: kept {kept}/{total} notes ({kept/total:.0%}) -> {args.out}")


if __name__ == "__main__":
    main()
