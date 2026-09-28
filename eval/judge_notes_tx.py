"""Transcript-grounded judgment of every note in sampled map outputs (data/train/rft_samples.jsonl),
for preference data on the map step.

Each note is judged against its own window's lines from 30 s before its anchor to 150 s after --
the same span eval/judge_prose_tx.py uses -- with the same verdicts. Output: one JSON line per
window with, per sample, the verdict of each note.
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.judge_prose_tx import PROMPT  # noqa: E402
from summarizer.pipeline import ChatClient, parse_turn  # noqa: E402

LINE = re.compile(r"^\[(\d+:\d{2}(?::\d{2})?)\]")


def secs(ts):
    p = [int(x) for x in ts.split(":")]
    return p[0] * 3600 + p[1] * 60 + p[2] if len(p) == 3 else p[0] * 60 + p[1]


def excerpt(window, ts):
    t0 = secs(ts)
    keep = [l for l in window.splitlines() if (m := LINE.match(l)) and -30 <= secs(m.group(1)) - t0 <= 150]
    return "\n".join(keep[:120])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="data/train/rft_samples.jsonl")
    ap.add_argument("--judge-url", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=16)
    args = ap.parse_args()
    chat = ChatClient(args.judge_url, args.judge_model, max_tokens=200)
    rows = [json.loads(l) for l in open(args.candidate, encoding="utf-8")]
    jobs = []
    for ri, r in enumerate(rows):
        window = r["messages"][1]["content"].split("TRANSCRIPT\n", 1)[-1]
        for si, text in enumerate(r["samples"]):
            for ni, (_, ts, t) in enumerate(parse_turn(text).notes):
                jobs.append((ri, si, ni, excerpt(window, ts), f"{t} [{ts}]"))

    def one(j):
        ri, si, ni, ex, note = j
        if not ex:
            return ri, si, ni, "uncited"
        for _ in range(2):
            m = re.search(r"\{.*\}", chat([{"role": "user", "content": PROMPT.format(excerpt=ex, sentence=note)}]), re.S)
            try:
                v = json.loads(m.group(0)).get("verdict")
                if v in ("supported", "contradicted", "unsupported"):
                    return ri, si, ni, v
            except (AttributeError, json.JSONDecodeError):
                pass
        return ri, si, ni, "unparsed"

    verdicts = {}
    for ri, si, ni, v in ThreadPoolExecutor(args.parallel).map(one, jobs):
        verdicts.setdefault(ri, {}).setdefault(si, {})[ni] = v
    with open(args.out, "w", encoding="utf-8") as f:
        for ri, r in enumerate(rows):
            f.write(json.dumps({"session": r["session"], "window": r["window"],
                                "verdicts": {si: [v[k] for k in sorted(v)] for si, v in verdicts.get(ri, {}).items()}},
                               ensure_ascii=False) + "\n")
    c = sum(v == "contradicted" for r in verdicts.values() for s in r.values() for v in s.values())
    n = sum(1 for r in verdicts.values() for s in r.values() for v in s.values())
    print(f"{n} notes judged, {c / max(1, n):.1%} contradicted -> {args.out}")


if __name__ == "__main__":
    main()
