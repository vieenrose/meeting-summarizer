"""Find figures whose citation doesn't locally support them -- a mechanical, scoped sibling to
check_derivable.py.

check_derivable.py asks "does this figure appear ANYWHERE in the transcript" -- it catches a
reference leak (a figure invented from outside knowledge) but not citation drift (a real figure,
attached to the wrong nearby line), because the figure genuinely exists somewhere else in the
same transcript. This asks the narrower question: does the figure appear near the specific line(s)
this point or note actually cites? Citation drift has been the single most common defect class
found across corpus v2's QA rounds by hand; this is an attempt to catch a slice of it for free.

Hits are candidates, not verdicts -- same discipline as check_derivable.py. A figure legitimately
introduced a few lines before its citation, or restated in a different numeral style nearby, can
still miss a fixed-radius window. Each hit needs a look before treating it as a real defect.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.ingest import extract_citations, parse_line, resolve_citation  # noqa: E402

FIGURE = re.compile(r"\d[\d,.]*\s*(?:億|萬|%)")
BRACKET = re.compile(r"[\[［][^\]］]*[\]］]")


def norm(s):
    return re.sub(r"[\s,，]", "", s)


def window_text(ts, lines, radius=3):
    idx = resolve_citation(ts, lines)
    if idx is None:
        return None
    return "".join(l.text for l in lines[max(0, idx - radius):idx + radius + 1])


def check_session(run, lines, radius=3):
    hits = []
    for n in run.get("notes") or []:
        figs = {norm(m) for m in FIGURE.findall(n["text"])}
        if not figs:
            continue
        ctx = window_text(n["ts"], lines, radius)
        if ctx is None:
            continue
        ctx = norm(ctx)
        missing = sorted(f for f in figs if f not in ctx)
        if missing:
            hits.append((f"note {n['id']} ({n['ts']})", missing))
    for i, point in enumerate(run.get("summary") or [], 1):
        cites = extract_citations(point)
        if not cites:
            continue
        ctxs = [c for c in (window_text(c, lines, radius) for c in cites) if c]
        if not ctxs:
            continue
        union_ctx = norm("".join(ctxs))
        body = BRACKET.sub("", point)
        figs = {norm(m) for m in FIGURE.findall(body)}
        missing = sorted(f for f in figs if f not in union_ctx)
        if missing:
            hits.append((f"point {i}", missing))
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/v2/w4000")
    ap.add_argument("--transcripts", default="data/v2/transcripts")
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    sessions = args.only or sorted(f[:-5] for f in os.listdir(args.run_dir) if f.endswith(".json"))
    total = 0
    for sid in sessions:
        run_path = os.path.join(args.run_dir, sid + ".json")
        tr_path = os.path.join(args.transcripts, sid + ".txt")
        if not (os.path.exists(run_path) and os.path.exists(tr_path)):
            continue
        run = json.load(open(run_path, encoding="utf-8"))
        lines = [parse_line(l) for l in open(tr_path, encoding="utf-8").read().splitlines() if l.strip()]
        hits = check_session(run, lines, args.radius)
        if hits:
            total += len(hits)
            print(f"\n{sid}: {len(hits)} location(s) with a figure not supported near its citation")
            for where, missing in hits:
                print(f"   {where}: {missing}")
    print(f"\n{total} candidate(s) across {len(sessions)} session(s)")


if __name__ == "__main__":
    main()
