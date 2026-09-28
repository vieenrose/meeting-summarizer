"""Corpus v2 generation: transcript -> gated teacher notes, summary and prose, per session.

For each session whose transcript exists and has not been generated yet:

  1. ASR gate (distill/v2_gates.asr_gate): skip transcripts that are mostly non-speech, looping, or
     unrelated to the reference transcript. No teacher call is spent on them.
  2. Teacher run: NotesPipeline with the v2 prompt (rules from v1's defect classes).
  3. Deterministic name repair: real-name substitutions go back to the transcript's spelling.
  4. Gates: structure, conflicting adopted amounts, guesses stated as fact, ambiguous names.
  5. Regenerate the summary step (not the whole session) up to --regen times while it fails a
     structural or hedge gate, keeping the attempt with the fewest problems.
  6. Prose summary from the final notes, held to its own checks.

What the gates cannot see -- chiefly an outcome paired with the wrong case -- is left to the
sampled review, and every remaining problem is written into the session file for it.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.v2_gates import asr_gate, auto_repair_names, check_session  # noqa: E402
from distill.write_prose import check as prose_check, write_one as write_prose  # noqa: E402
from eval.zen_client import ZenChat  # noqa: E402
from summarizer.ingest import parse_line  # noqa: E402
from summarizer.pipeline import ChatClient, Note, NotesPipeline, PipelineConfig  # noqa: E402

REGEN_KINDS = {"structure", "hedge"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/v2/pilot_manifest.json")
    ap.add_argument("--transcripts", default="data/v2/transcripts")
    ap.add_argument("--reference", default="data/v2/reference")
    ap.add_argument("--out", default="runs/v2/w4000")
    ap.add_argument("--model", default="go:union-alpha")
    ap.add_argument("--backend", choices=["zen", "vllm"], default="zen",
                    help="zen: OpenCode gateway (ZenChat); vllm: local OpenAI-compatible server")
    ap.add_argument("--base-url", default="", help="required with --backend vllm")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--regen", type=int, default=2)
    ap.add_argument("--wait", action="store_true", help="keep polling for new transcripts")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B-FP8")
    count = lambda t: len(tok.encode(t, add_special_tokens=False))  # noqa: E731
    os.makedirs(args.out, exist_ok=True)
    rejected_path = os.path.join(os.path.dirname(args.out.rstrip("/")), "asr_rejected.json")
    rejected_lock = threading.Lock()
    sessions = [f"ivod_{s['ivod_id']}" for s in json.load(open(args.manifest, encoding="utf-8"))["sessions"]]

    def one(sid):
        dest = os.path.join(args.out, sid + ".json")
        tr_path = os.path.join(args.transcripts, sid + ".txt")
        if os.path.exists(dest) or not os.path.exists(tr_path):
            return None
        text = open(tr_path, encoding="utf-8").read()
        ref_path = os.path.join(args.reference, sid + ".txt")
        ref = open(ref_path, encoding="utf-8").read() if os.path.exists(ref_path) else None
        ok, metrics, reasons = asr_gate(text, ref)
        if not ok:
            with rejected_lock:
                rej = json.load(open(rejected_path)) if os.path.exists(rejected_path) else {}
                rej[sid] = {"metrics": metrics, "reasons": reasons}
                json.dump(rej, open(rejected_path, "w"), ensure_ascii=False, indent=1)
            return f"{sid}: ASR gate rejected ({'; '.join(reasons)})"

        if args.backend == "vllm":
            chat = ChatClient(args.base_url, args.model, max_tokens=4096)
        else:
            chat = ZenChat(args.model, max_tokens=16000, timeout=900, retries=3)
        pipe = NotesPipeline(chat, count, PipelineConfig(prompt_version="v2"))
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
        json.dump(run, open(dest, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return (f"{sid}: {len(run['notes'])} notes, {len(run['summary'])} points, "
                f"{len(repaired)} names repaired, {attempts} regenerations, "
                f"{len(run['gates'])} open problems, {run['elapsed_s'] / 60:.0f} min")

    while True:
        pending = [s for s in sessions if not os.path.exists(os.path.join(args.out, s + ".json"))
                   and os.path.exists(os.path.join(args.transcripts, s + ".txt"))]
        if pending:
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                for line in pool.map(one, pending):
                    if line:
                        print(line, flush=True)
        done = sum(os.path.exists(os.path.join(args.out, s + ".json")) for s in sessions)
        rejected = json.load(open(rejected_path)) if os.path.exists(rejected_path) else {}
        if not args.wait or done + len(rejected) >= len(sessions):
            break
        time.sleep(60)
    print(f"generated {done}/{len(sessions)}, ASR-rejected {len(rejected)}", flush=True)


if __name__ == "__main__":
    main()
