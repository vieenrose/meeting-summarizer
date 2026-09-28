"""Gate teacher output before it is used as a reference or as training data.

Nothing the teacher produces is trusted on the strength of the teacher's name. Every summary is
checked mechanically, and only what passes may be used. Measured on 39 sessions with the same
Qwen3.6-35B-A3B teacher: 87% of notes-pipeline summaries pass, against 42% of single-pass ones, so
the procedure decides the yield far more than the model does.

Checks, all cheap and objective:
  * every point cites a timestamp that exists in the transcript;
  * the points together reach each third of the meeting;
  * length is inside the target band;
  * the summary is not degenerate (too few points, or repeated points);
  * for windowed runs, no window with real speech was left without notes.

Failures are written out with their reasons, so a rejected session can be regenerated rather than
silently dropped.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.ingest import parse_line  # noqa: E402
from summarizer.pipeline import NotesPipeline, PipelineConfig, uncoverable_thirds  # noqa: E402


def expected_scale(lines: list) -> tuple:
    """How much summary a meeting of this size should produce.

    A fixed floor punishes the right behaviour: several sessions in this corpus are near-silent
    procedural sittings, and a faithful three-point summary of one is not a defect. Expectations
    therefore scale with how much was actually said, while the upper bound stays fixed because the
    phone has a hard output budget.
    """
    speech = sum(len(l.text) for l in lines)
    if speech < 3000:
        return 2, 80
    if speech < 10000:
        return 3, 150
    return 4, 250


def check(run: dict, lines: list, min_chars: int, max_chars: int, min_points: int) -> list:
    checker = NotesPipeline(lambda m: "", len, PipelineConfig())
    problems = list(checker._check_summary(run["summary"], lines))
    points = run["summary"]
    chars = len(re.sub(r"\[[^\]]*\]|\s", "", "".join(points)))
    scale_points, scale_chars = expected_scale(lines)
    floor_points, floor_chars = max(min_points, scale_points), max(min_chars, scale_chars)
    if min_points <= 0:      # caller opted out of the fixed floor
        floor_points, floor_chars = scale_points, scale_chars
    if len(points) < floor_points:
        problems.append(f"only {len(points)} points, expected >= {floor_points}")
    if not (floor_chars <= chars <= max_chars):
        problems.append(f"{chars} chars outside {floor_chars}-{max_chars}")
    bodies = [re.sub(r"^\s*\d+[.、]\s*|\[[^\]]*\]", "", p).strip() for p in points]
    dupes = [b for b, n in Counter(bodies).items() if n > 1 and b]
    if dupes:
        problems.append(f"{len(dupes)} repeated point(s)")
    if uncoverable_thirds(lines):
        # A third with no intelligible speech cannot be cited, so its absence is not a defect.
        problems = [p for p in problems if "缺少會議" not in p]
    if run.get("windows_without_notes"):
        problems.append(f"{len(run['windows_without_notes'])} window(s) with no notes")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--out", default=None, help="manifest of passing sessions")
    ap.add_argument("--min-chars", type=int, default=0, help="0 = scale with transcript size")
    ap.add_argument("--max-chars", type=int, default=650)
    ap.add_argument("--min-points", type=int, default=0, help="0 = scale with transcript size")
    args = ap.parse_args()

    passed, failed = [], []
    for path in sorted(glob.glob(os.path.join(args.run_dir, "*.json"))):
        if path.endswith(".score.json"):
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        tpath = os.path.join(args.transcripts, f"{stem}.txt")
        if not os.path.exists(tpath):
            continue
        run = json.load(open(path, encoding="utf-8"))
        lines = [parse_line(l) for l in open(tpath, encoding="utf-8").read().splitlines() if l.strip()]
        problems = check(run, lines, args.min_chars, args.max_chars, args.min_points)
        exempt = uncoverable_thirds(lines)
        entry = {"id": stem, "path": path, "problems": problems}
        if exempt:
            names = {0: "前段", 1: "中段", 2: "後段"}
            entry["exempt_thirds"] = [names[t] for t in sorted(exempt)]
        (failed if problems else passed).append(entry)

    total = len(passed) + len(failed)
    print(f"{args.run_dir}: {len(passed)}/{total} pass ({len(passed)/max(1,total):.0%})")
    for e in passed:
        if e.get("exempt_thirds"):
            print(f"   {e['id']}: {'、'.join(e['exempt_thirds'])} exempt, no intelligible speech")
    reasons = Counter(p.split(" ")[0] if p[0].isdigit() else p[:12] for f in failed for p in f["problems"])
    for reason, n in reasons.most_common(6):
        print(f"   {n:3d}  {reason}…")
    for f in failed[:5]:
        print(f"   FAIL {f['id']}: {f['problems']}")

    if args.out:
        json.dump({"run_dir": args.run_dir, "passed": [p["id"] for p in passed],
                   "failed": failed}, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"   manifest -> {args.out}")


if __name__ == "__main__":
    main()
