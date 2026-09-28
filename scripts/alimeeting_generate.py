"""Generate notes/summary/prose for AliMeeting sessions with the same teacher pipeline and output
shape as scripts/v2_generate.py, so the result is directly comparable to runs/v2/w4000/*.json.

Kept in its own output tree (runs/alimeeting/w4000), never merged into runs/v2 -- AliMeeting is a
different domain (mainland business meetings, not Legislative Yuan committees) and mixing it into
the VoxSumDroid training corpus is a decision for the corpus owner, not this script.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.v2_gates import asr_gate, auto_repair_names, check_session  # noqa: E402
from distill.write_prose import write_one as write_prose  # noqa: E402
from summarizer.ingest import parse_line  # noqa: E402
from summarizer.pipeline import ChatClient, Note, NotesPipeline, PipelineConfig  # noqa: E402

REGEN_KINDS = {"structure", "hedge"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/alimeeting/manifest.json")
    ap.add_argument("--transcripts", default="data/alimeeting/transcripts")
    ap.add_argument("--out", default="runs/alimeeting/w4000")
    ap.add_argument("--model", default="gemma4-31b-fp8-teacher")
    ap.add_argument("--base-url", default="http://127.0.0.1:8700/v1")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--regen", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sessions = [s["session_id"] for s in json.load(open(args.manifest, encoding="utf-8"))["sessions"]]

    def one(sid):
        dest = os.path.join(args.out, sid + ".json")
        tr_path = os.path.join(args.transcripts, sid + ".txt")
        if os.path.exists(dest) or not os.path.exists(tr_path):
            return None
        text = open(tr_path, encoding="utf-8").read()
        ok, metrics, reasons = asr_gate(text)
        if not ok:
            return f"{sid}: gate rejected ({'; '.join(reasons)})"

        chat = ChatClient(args.base_url, args.model, max_tokens=4096)
        pipe = NotesPipeline(chat, len, PipelineConfig(prompt_version="v2"))
        t0 = time.time()
        run = pipe.run(text)
        run["asr"] = metrics
        repaired, _ = auto_repair_names(run, text)
        problems = check_session(run, text)

        lines = [parse_line(l) for l in text.splitlines() if l.strip()]
        best = (run["summary"], problems)
        attempts = 0
        while attempts < args.regen and any(k in REGEN_KINDS for k, _ in best[1]):
            attempts += 1
            notes = [Note(id=n["id"], window=n["window"], ts=n["ts"], text=n["text"], tag=n.get("tag"))
                     for n in run["notes"]]
            summary, log = pipe._write(lines, notes)
            trial = dict(run, summary=summary)
            auto_repair_names(trial, text)
            trial_problems = check_session(trial, text)
            run.setdefault("log", []).extend(log)
            if len(trial_problems) < len(best[1]):
                best = (trial["summary"], trial_problems)
        run["summary"], problems = best
        run["regenerations"] = attempts

        prose, prose_problems, prose_log = write_prose(chat, run, lines, turns=4)
        run["prose"], run["prose_problems"] = prose, prose_problems
        auto_repair_names(run, text)
        run["gates"] = [{"kind": k, "detail": d} for k, d in check_session(run, text)]
        run["teacher"], run["elapsed_s"] = args.model, time.time() - t0
        run["source"] = "AliMeeting (OpenSLR-119, CC BY-SA 4.0)"
        json.dump(run, open(dest, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return (f"{sid}: {len(run['notes'])} notes, {len(run['summary'])} points, "
                f"{len(repaired)} names repaired, {attempts} regenerations, "
                f"{len(run['gates'])} open problems, {run['elapsed_s'] / 60:.0f} min")

    pending = [s for s in sessions if not os.path.exists(os.path.join(args.out, s + ".json"))]
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for line in pool.map(one, pending):
            if line:
                print(line, flush=True)
    done = sum(os.path.exists(os.path.join(args.out, s + ".json")) for s in sessions)
    print(f"generated {done}/{len(sessions)}", flush=True)


if __name__ == "__main__":
    main()
