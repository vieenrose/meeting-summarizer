"""Compare student runs with the teacher's gold on the held-out sessions.

Three kinds of measure, because each misses something the others catch:

  well-formed   the same gate the teacher's output had to pass: real citations, every third of
                the meeting covered, a length band, no empty windows.
  grounded      the share of cited timestamps that exist in the transcript, and the share of
                names and figures in the summary that occur in the transcript at all -- the
                student analogue of the leak check the gold was repaired against.
  faithful      recall of the gold summary's key facts: its figures and proper names, found
                anywhere in the student's summary. A crude proxy for coverage (no human key-fact
                list exists yet), but it moves with what matters and it cannot be gamed by
                formatting.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.validate_teacher import check  # noqa: E402
from summarizer.ingest import parse_line, resolve_citation  # noqa: E402

CITE = re.compile(r"[\[［](\d{1,2}(?::\d{2}){1,2})[\]］]")
FIGURE = re.compile(r"\d[\d,.]*\s*(?:億|萬|千|%|人|案|條)")
NAME = re.compile(r"[一-鿿]{2,3}(?=委員|部長|次長|署長|主席|召委|院長)")


def facts(text):
    t = re.sub(r"\s|,", "", text)
    return set(FIGURE.findall(t)) | set(NAME.findall(t))


def score(run, lines, gold):
    summary = "\n".join(run.get("summary") or [])
    cites = CITE.findall(summary)
    real = sum(1 for c in cites if resolve_citation(c, lines) is not None)
    hay = re.sub(r"\s|,", "", " ".join(l.text for l in lines))
    toks = facts(summary)
    grounded = sum(1 for t in toks if t in hay)
    gold_facts = facts("\n".join(gold["summary"])) if gold else set()
    recalled = sum(1 for t in gold_facts if t in re.sub(r"\s|,", "", summary))
    return {
        "pass": not check(run, lines, 0, 650, 0),
        "points": len(run.get("summary") or []),
        "chars": len(re.sub(r"\[[^\]]*\]|\s", "", summary)),
        "notes": len(run.get("notes") or []),
        "empty_windows": len(run.get("windows_without_notes") or []),
        "cites": len(cites), "cites_real": real,
        "facts": len(toks), "facts_grounded": grounded,
        "gold_facts": len(gold_facts), "gold_recalled": recalled,
        "seconds": run.get("elapsed_s", 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=run_dir")
    ap.add_argument("--gold", default="runs/gold/w4000")
    ap.add_argument("--split", default="data/split.json")
    ap.add_argument("--transcripts", default="data/transcripts")
    args = ap.parse_args()
    sessions = json.load(open(args.split))["heldout"]

    table = {}
    for spec in args.runs:
        label, d = spec.split("=", 1)
        agg = {}
        for sid in sessions:
            path = os.path.join(d, sid + ".json")
            if not os.path.exists(path):
                continue
            lines = [parse_line(l) for l in open(os.path.join(args.transcripts, sid + ".txt"),
                                                  encoding="utf-8").read().splitlines() if l.strip()]
            gold = json.load(open(os.path.join(args.gold, sid + ".json"), encoding="utf-8"))
            s = score(json.load(open(path, encoding="utf-8")), lines, gold)
            for k, v in s.items():
                agg[k] = agg.get(k, 0) + v
            agg["n"] = agg.get("n", 0) + 1
        table[label] = agg

    def pct(a, b):
        return f"{100 * a / b:.0f}%" if b else "-"

    rows = [
        ("sessions", lambda a: a["n"]),
        ("pass validation", lambda a: f"{a['pass']}/{a['n']}"),
        ("avg points", lambda a: f"{a['points'] / a['n']:.1f}"),
        ("avg summary chars", lambda a: f"{a['chars'] / a['n']:.0f}"),
        ("avg notes", lambda a: f"{a['notes'] / a['n']:.1f}"),
        ("windows with no notes", lambda a: a["empty_windows"]),
        ("citations that exist", lambda a: f"{pct(a['cites_real'], a['cites'])} of {a['cites']}"),
        ("names/figures in transcript", lambda a: f"{pct(a['facts_grounded'], a['facts'])} of {a['facts']}"),
        ("gold key facts recalled", lambda a: f"{pct(a['gold_recalled'], a['gold_facts'])} of {a['gold_facts']}"),
        ("avg minutes / session", lambda a: f"{a['seconds'] / a['n'] / 60:.1f}"),
    ]
    labels = list(table)
    print(f"{'':30}" + "".join(f"{l:>18}" for l in labels))
    for name, fn in rows:
        print(f"{name:30}" + "".join(f"{str(fn(table[l])):>18}" for l in labels))


if __name__ == "__main__":
    main()
