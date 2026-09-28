"""Ask the teacher for notes on a stretch of transcript its notes step skipped.

Some windows come back with notes clustered at one end: the outcomes-first instruction works, but a
long procedural middle can be passed over entirely, leaving a third of the meeting with nothing to
summarise from. Rather than invent a point, this replays that exact line range to the same teacher
under the same note-taking prompt, so the filled gap is still the teacher's own words, and merges
the result into the run's notes in timestamp order.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.zen_client import ZenChat  # noqa: E402
from summarizer.ingest import parse_line, resolve_citation  # noqa: E402
from summarizer.pipeline import SYSTEM_PROMPT, Note, parse_turn  # noqa: E402

CITE = re.compile(r"[\[［](\d{1,2}(?::\d{2}){1,2})[\]］]")


def uncovered_thirds(summary, lines):
    n = max(1, len(lines))
    got = set()
    for p in summary:
        for c in CITE.findall(p):
            i = resolve_citation(c, lines)
            if i is not None:
                got.add(min(2, 3 * i // n))
    return sorted({0, 1, 2} - got)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/gold/w4000")
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--model", default="minimax-m3")
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--max-output-tokens", type=int, default=24000)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    chat = ZenChat(args.model, max_tokens=args.max_output_tokens, timeout=900, retries=3)

    for s in args.sessions:
        path = os.path.join(args.run_dir, f"{s}.json")
        run = json.load(open(path, encoding="utf-8"))
        lines = [parse_line(l) for l in
                 open(os.path.join(args.transcripts, f"{s}.txt"), encoding="utf-8").read().splitlines()
                 if l.strip()]
        n = len(lines)
        added_any = False
        for third in uncovered_thirds(run["summary"], lines):
            lo, hi = third * n // 3, (third + 1) * n // 3
            block = "\n".join(l.render() for l in lines[lo:hi])
            name = {0: "前段", 1: "中段", 2: "後段"}[third]
            user = (f"以下是會議逐字稿的{name}，先前的筆記遺漏了這一段。\n\n"
                    f"本片段有實質發言，必須寫下筆記，不得輸出 NOTHING-NEW。\n\n"
                    f"TRANSCRIPT\n{block}")
            reply = chat([{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}])
            turn = parse_turn(reply)
            kept = []
            for tag, ts, text in turn.notes[:6]:
                idx = resolve_citation(ts, lines[lo:hi])
                if idx is None:
                    continue           # must cite a line inside the gap it was asked about
                kept.append((tag, ts, text))
            print(f"{s}: {name} gap lines [{lo},{hi}) -> {len(kept)} notes "
                  f"({len(turn.notes)} offered, {len(turn.rejected)} unparsed)"
                  f"{' | ' + (chat.last_error or '') if chat.last_error else ''}")
            for tag, ts, text in kept:
                print(f"    [{ts}] {text[:88]}")
            if kept and args.apply:
                win = max((nt["window"] for nt in run["notes"]), default=0)
                for tag, ts, text in kept:
                    run["notes"].append({"id": len(run["notes"]) + 1, "window": win,
                                         "ts": ts, "text": text, "tag": tag})
                run["notes"].sort(key=lambda nt: resolve_citation(nt["ts"], lines) or 0)
                for i, nt in enumerate(run["notes"], 1):
                    nt["id"] = i
                run.setdefault("repairs", []).append(
                    f"teacher re-ran on the {name} gap, {len(kept)} notes added")
                added_any = True
        if added_any:
            json.dump(run, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"   {s}: notes written ({len(run['notes'])} total)")


if __name__ == "__main__":
    main()
