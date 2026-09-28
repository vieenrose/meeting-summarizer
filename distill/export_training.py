"""Export teacher runs as student training data.

A run is not one training example: it is one example per window plus one for the synthesis. Those
per-window rows are the valuable part -- they teach the behaviour the phone actually performs, and
this project has already measured that per-decision rows beat whole trajectories for small students.

Nothing unvalidated is exported. A session whose summary fails the gate is dropped whole, including
its window rows, because a summary that lost the meeting's outcome was produced by note-taking that
missed it. Per-row filters then drop individual windows whose notes were rejected.

Output (JSONL, chat format):
  notes rows     system + window prompt -> the notes the teacher wrote for that window
  synthesis rows notes digest           -> the final summary
Each row records its session, window and the teacher, so any row can be traced back.
"""
import argparse
import glob
import json
import re
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.validate_teacher import check  # noqa: E402
from summarizer.ingest import parse_line  # noqa: E402
from distill.write_prose import (PROMPT as PROSE_PROMPT, MIN_CHARS as PROSE_MIN,  # noqa: E402
                                 MAX_CHARS as PROSE_MAX, check as prose_check)


HEADER = "以下是整場會議依序寫下的筆記：\n\n"
NOTES_BLOCK = re.compile(re.escape(HEADER) + r".*?\n\n(?=請寫出)", re.S)


def rows_from_run(run: dict, session: str, teacher: str) -> list:
    """One row per window that produced accepted notes, plus one synthesis row."""
    rows = []
    notes_by_window = {}
    for n in run["notes"]:
        notes_by_window.setdefault(n["window"], []).append(n)

    for entry in run["log"]:
        window = entry.get("window")
        if window is None or not entry.get("messages"):
            continue
        # A retry means the first answer was wrong; train on the state, not the mistake.
        if entry.get("retry"):
            continue
        accepted = notes_by_window.get(window, [])
        if not accepted:
            continue
        target = "\n".join(f"- ({n['tag']}) [{n['ts']}] {n['text']}" if n.get("tag")
                           else f"- [{n['ts']}] {n['text']}" for n in accepted)
        rows.append({"kind": "notes", "session": session, "window": window, "teacher": teacher,
                     "messages": list(entry["messages"]) + [{"role": "assistant", "content": target}]})

    # Any write attempt's message shape works as the template: the caller already validated the
    # session's CURRENT summary (main()'s check() call, post-QA-fixes), so a stale "problems" flag
    # on the attempt that happened to produce it at generation time no longer means anything -- and
    # dropping the row on that basis silently lost the synthesis example for a session already
    # confirmed good.
    write = [e for e in run["log"] if e.get("step") == "write"]
    if write and run.get("summary"):
        # The synthesis prompt embeds the notes as they were when the teacher wrote the summary.
        # Notes repaired since then (a garbled name the teacher had "corrected", a wrong pairing)
        # would survive in that frozen copy, giving a row whose input still carries the defect its
        # target no longer has. Rebuild the notes block from the current notes instead.
        messages = [dict(m) for m in write[-1]["messages"]]
        current = "\n".join(
            f"note {n['id']}: " + (f"({n['tag']}) " if n.get("tag") else "") + f"[{n['ts']}] {n['text']}"
            for n in run["notes"])
        for m in messages:
            if m["role"] == "user" and NOTES_BLOCK.search(m["content"]):
                m["content"] = NOTES_BLOCK.sub(lambda _: HEADER + current + "\n\n", m["content"], count=1)
        rows.append({"kind": "synthesis", "session": session, "window": None, "teacher": teacher,
                     "messages": messages
                                 + [{"role": "assistant", "content": "\n".join(run["summary"])}]})

    # The prose abstract is the summary the user reads. Its prompt is rebuilt from the current notes
    # for the same reason as the synthesis row; the caller has already checked the current prose.
    if run.get("prose"):
        notes = "\n".join(f"[{n['ts']}] {n['text']}" for n in run["notes"])
        prompt = PROSE_PROMPT.format(notes=notes, lo=PROSE_MIN, hi=PROSE_MAX)
        rows.append({"kind": "prose", "session": session, "window": None, "teacher": teacher,
                     "messages": [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": run["prose"]}]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="teacher run directory (…/w4000)")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--out", default="data/train/teacher_rows.jsonl")
    ap.add_argument("--keep-unvalidated", action="store_true",
                    help="export sessions whose summary failed the gate (off by default)")
    ap.add_argument("--exemptions", default=None,
                    help="JSON {session: reason} of sessions to keep despite a failing gate, "
                         "each individually investigated and confirmed not a real defect")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    exemptions = json.load(open(args.exemptions, encoding="utf-8")) if args.exemptions else {}

    kept, exempted, dropped, rows = 0, 0, [], []
    prose_dropped = []
    for path in sorted(glob.glob(os.path.join(args.run_dir, "*.json"))):
        if path.endswith(".score.json"):
            continue
        session = os.path.splitext(os.path.basename(path))[0]
        tpath = os.path.join(args.transcripts, f"{session}.txt")
        if not os.path.exists(tpath):
            continue
        run = json.load(open(path, encoding="utf-8"))
        lines = [parse_line(l) for l in open(tpath, encoding="utf-8").read().splitlines() if l.strip()]
        problems = check(run, lines, 0, 650, 0)
        if problems and not args.keep_unvalidated:
            if session in exemptions:
                exempted += 1
            else:
                dropped.append((session, problems))
                continue
        kept += 1
        new = rows_from_run(run, session, args.teacher)
        # A prose abstract that fails its own gate (bad citation, missing third) is not a target.
        if run.get("prose") and prose_check(run["prose"], lines):
            prose_dropped.append((session, prose_check(run["prose"], lines)))
            new = [r for r in new if r["kind"] != "prose"]
        rows.extend(new)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    kinds = Counter(r["kind"] for r in rows)
    print(f"sessions kept {kept} ({exempted} via documented exemption), dropped {len(dropped)}")
    for s, p in dropped[:5]:
        print(f"   dropped {s}: {p}")
    print(f"prose rows dropped by the prose gate: {len(prose_dropped)}")
    for s, p in prose_dropped[:10]:
        print(f"   prose {s}: {p}")
    print(f"rows: {dict(kinds)} -> {args.out}")
    if kinds:
        chars = sum(len(m["content"]) for r in rows for m in r["messages"])
        print(f"total {chars/1e6:.1f}M characters of training text")


if __name__ == "__main__":
    main()
