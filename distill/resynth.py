"""Rewrite a session's summary from its current notes, after the notes changed.

Used when a gap re-run added notes: the old summary was written without them, so it cannot cover
the recovered material. Reuses the pipeline's own synthesis prompt and checks, so a resynthesised
summary is held to exactly the standard every other summary was.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.zen_client import ZenChat  # noqa: E402
from summarizer.ingest import parse_line  # noqa: E402
from summarizer.pipeline import Note, NotesPipeline, PipelineConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/gold/w4000")
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--model", default="minimax-m3")
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--max-output-tokens", type=int, default=24000)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B-FP8")
    chat = ZenChat(args.model, max_tokens=args.max_output_tokens, timeout=900, retries=3)
    pipe = NotesPipeline(chat, lambda t: len(tok.encode(t, add_special_tokens=False)),
                         PipelineConfig())

    for s in args.sessions:
        path = os.path.join(args.run_dir, f"{s}.json")
        run = json.load(open(path, encoding="utf-8"))
        lines = [parse_line(l) for l in
                 open(os.path.join(args.transcripts, f"{s}.txt"), encoding="utf-8").read().splitlines()
                 if l.strip()]
        notes = [Note(id=n["id"], window=n["window"], ts=n["ts"], text=n["text"], tag=n.get("tag"))
                 for n in run["notes"]]
        summary, log = pipe._write(lines, notes)
        problems = log[-1]["problems"] if log else ["no write step"]
        print(f"{s}: {len(notes)} notes -> {len(summary)} points, problems={problems}")
        for p in summary:
            print(f"    {p[:100]}")
        if args.apply and summary:
            run["summary"] = summary
            run["log"].extend(log)
            run.setdefault("repairs", []).append("summary rewritten after gap notes were added")
            json.dump(run, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print("    written")


if __name__ == "__main__":
    main()
