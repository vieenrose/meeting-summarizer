"""Manually replay the teacher on exactly the lines a generated session's notes skipped over,
splice the accepted notes back in, and add one summary point for them.

For the rare case where a full session regeneration hits the same gap twice -- a targeted retry
on just the missing lines is cheaper and doesn't disturb everything else already reviewed.
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.v2_gates import check_session  # noqa: E402
from summarizer.ingest import parse_line, resolve_citation  # noqa: E402
from summarizer.pipeline import (ChatClient, NotesPipeline, PipelineConfig,  # noqa: E402
                                 SYSTEM_PROMPT_V2, parse_turn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--start-line", type=int, required=True, help="0-indexed, inclusive")
    ap.add_argument("--end-line", type=int, required=True, help="0-indexed, exclusive")
    ap.add_argument("--run-dir", default="runs/v2/w4000")
    ap.add_argument("--transcripts", default="data/v2/transcripts")
    ap.add_argument("--base-url", default="http://127.0.0.1:8700/v1")
    ap.add_argument("--model", default="gemma4-31b-fp8-teacher")
    args = ap.parse_args()

    run_path = os.path.join(args.run_dir, args.session + ".json")
    tr_path = os.path.join(args.transcripts, args.session + ".txt")
    run = json.load(open(run_path, encoding="utf-8"))
    text = open(tr_path, encoding="utf-8").read()
    lines = [parse_line(l) for l in text.splitlines() if l.strip()]
    gap = lines[args.start_line:args.end_line]
    print(f"{args.session}: replaying {len(gap)} lines ({args.start_line}:{args.end_line}) "
          f"of {len(lines)} total")

    chat = ChatClient(args.base_url, args.model, max_tokens=1024)
    pipe = NotesPipeline(chat, len, PipelineConfig(prompt_version="v2"))
    block = "\n".join(l.render() for l in gap)
    messages = [{"role": "system", "content": SYSTEM_PROMPT_V2},
                {"role": "user", "content": pipe._window_prompt(1, 1, block, allow_nothing_new=False)}]
    reply = chat(messages)
    turn = parse_turn(reply)
    print("raw reply:\n", reply)

    existing_ids = [n["id"] for n in run["notes"]]
    next_id = (max(existing_ids) + 1) if existing_ids else 1
    # attribute the new notes to whichever existing window number brackets this line range
    window_for_line = {}
    # approximate: spread windows evenly by note density isn't reliable (see prior investigation),
    # so just tag these with the window number of the nearest existing note before the gap.
    before = [n for n in run["notes"] if resolve_citation(n["ts"], lines) is not None
              and resolve_citation(n["ts"], lines) < args.start_line]
    window_tag = before[-1]["window"] if before else run["notes"][0]["window"]

    added = []
    for tag, ts, note_text in turn.notes:
        if resolve_citation(ts, gap) is None and resolve_citation(ts, lines) is None:
            print(f"  rejected invented timestamp [{ts}] {note_text}")
            continue
        n = {"id": next_id, "window": window_tag, "ts": ts, "text": note_text,
             "tag": tag if tag else None}
        run["notes"].append(n)
        added.append(n)
        next_id += 1
    if not added:
        print("no notes accepted -- aborting, run file unchanged")
        return
    run["notes"].sort(key=lambda n: resolve_citation(n["ts"], lines) or 0)

    # one summary point drawn straight from the first accepted note, same style as the rest
    best = added[0]
    point_body = best["text"]
    if len(point_body) > 100:
        point_body = point_body[:97] + "..."
    new_point = f"{len(run['summary']) + 1}. {point_body} [{best['ts']}]"
    run["summary"].append(new_point)

    run["gates"] = [{"kind": k, "detail": d} for k, d in check_session(run, text)]
    json.dump(run, open(run_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"added {len(added)} note(s), 1 summary point. New gates: {run['gates']}")


if __name__ == "__main__":
    main()
