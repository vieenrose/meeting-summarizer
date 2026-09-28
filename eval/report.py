"""Protocol metrics for a set of runs: the numbers that say whether the agent read the meeting.

These are mechanical (no judge): notes per window, windows left without notes, how citations were
classified, summary length and how the citations spread across the meeting. Quality scoring lives
in eval/score.py.
"""
import argparse
import glob
import json
import os
import re
import statistics as stats
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.ingest import parse_line, resolve_citation  # noqa: E402


def summarize_run(path: str, transcripts: str) -> dict:
    run = json.load(open(path, encoding="utf-8"))
    stem = os.path.splitext(os.path.basename(path))[0]
    tpath = os.path.join(transcripts, f"{stem}.txt")
    lines = [parse_line(l) for l in open(tpath, encoding="utf-8").read().splitlines() if l.strip()] \
        if os.path.exists(tpath) else []
    notes = run["notes"]
    rejected = [r for e in run["log"] for r in e.get("rejected", [])]
    invented = [r for r in rejected if r.startswith("invented timestamp")]
    out_of_window = [r for r in rejected if r.startswith("out-of-window timestamp")]
    template_copied = [r for r in rejected if "帶有前因後果" in r or "完整句子筆記" in r]
    retries = [e.get("retry") for e in run["log"] if e.get("retry")]
    summary = run.get("summary", [])
    thirds = set()
    cited = 0
    for point in summary:
        for c in re.findall(r"\[(\d{1,2}(?::\d{2}){1,2})\]", point):
            idx = resolve_citation(c, lines)
            if idx is not None:
                cited += 1
                thirds.add(min(2, 3 * idx // max(1, len(lines))))
    chars = len(re.sub(r"\[[^\]]*\]|\s", "", "".join(summary)))
    return {
        "id": stem,
        "windows": run["windows"],
        "notes": len(notes),
        "notes_per_window": len(notes) / max(1, run["windows"]),
        "empty_windows": len(run["windows_without_notes"]),
        "tagged": sum(1 for n in notes if n.get("tag")),
        "invented_citations": len(invented),
        "out_of_window_citations": len(out_of_window),
        "template_copied": len(template_copied),
        "other_rejected": len(rejected) - len(invented) - len(out_of_window),
        "retries": len(retries),
        "points": len(summary),
        "summary_chars": chars,
        "valid_point_citations": cited,
        "thirds_covered": len(thirds),
        "elapsed_s": run.get("elapsed_s"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    table = []
    for run_dir in args.run_dirs:
        rows = [summarize_run(p, args.transcripts) for p in sorted(glob.glob(os.path.join(run_dir, "*.json")))
                if not p.endswith(".score.json")]
        if not rows:
            print(f"{run_dir}: no runs")
            continue
        agg = {"run": run_dir, "n": len(rows)}
        for key in ("windows", "notes", "notes_per_window", "empty_windows", "tagged",
                    "invented_citations", "out_of_window_citations", "template_copied",
                    "other_rejected", "retries", "points",
                    "summary_chars", "valid_point_citations", "thirds_covered", "elapsed_s"):
            values = [r[key] for r in rows if r[key] is not None]
            agg[key] = round(stats.mean(values), 2) if values else None
        agg["all_thirds_share"] = round(sum(r["thirds_covered"] == 3 for r in rows) / len(rows), 2)
        agg["zero_empty_share"] = round(sum(r["empty_windows"] == 0 for r in rows) / len(rows), 2)
        table.append(agg)
        print(f"\n== {run_dir}  n={len(rows)}")
        print(f"   windows {agg['windows']}, notes {agg['notes']} ({agg['notes_per_window']}/window), "
              f"empty windows {agg['empty_windows']} (all-covered {agg['zero_empty_share']:.0%})")
        print(f"   tagged {agg['tagged']}, retries {agg['retries']}")
        print(f"   invented citations {agg['invented_citations']}, out-of-window {agg['out_of_window_citations']}, "
              f"template copied {agg['template_copied']}, other rejected {agg['other_rejected']}")
        print(f"   summary {agg['points']} points / {agg['summary_chars']} chars, "
              f"valid citations {agg['valid_point_citations']}, all thirds {agg['all_thirds_share']:.0%}")
        if agg["elapsed_s"]:
            print(f"   elapsed {agg['elapsed_s'] / 60:.1f} min/session")

    if args.out:
        json.dump(table, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
