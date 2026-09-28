"""Pick, per training session, the least-contradicted of N sampled prose abstracts (judged against
the transcript by eval/judge_prose_tx.py) and write it as a prose training row.

A session is kept only when its best sample has strictly fewer contradicted sentences than its
median sample and at most one contradiction: a tie teaches nothing, and a contradicted best is
still a wrong target. The prompt is the student's own notes, as at inference.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.write_prose import prompt_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--prefix", default="runs/student/rft-prose-s")
    ap.add_argument("--out", default="data/train/rft_prose_rows.jsonl")
    args = ap.parse_args()
    score = collections.defaultdict(dict)
    for i in range(args.n):
        for tag in ("ivod_", "ali_"):
            p = f"reports/judge_prose_tx_{tag}rft-prose-s{i}.json"
            if not os.path.exists(p):
                continue
            by = collections.defaultdict(lambda: [0, 0])
            for r in json.load(open(p, encoding="utf-8"))["rows"]:
                by[r["id"]][0] += r["verdict"] == "contradicted"
                by[r["id"]][1] += r["verdict"] == "supported"
            for sid, v in by.items():
                score[sid][i] = v
    kept = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for sid, s in score.items():
            if len(s) < 2:
                continue
            ranked = sorted(s.items(), key=lambda kv: (kv[1][0], -kv[1][1]))
            best_i, (bc, _) = ranked[0]
            median_c = sorted(v[0] for v in s.values())[len(s) // 2]
            if bc < median_c and bc <= 1:
                run = json.load(open(f"{args.prefix}{best_i}/{sid}.json", encoding="utf-8"))
                f.write(json.dumps({"kind": "prose", "session": sid, "window": None, "teacher": "rft-prose",
                                    "messages": [{"role": "user", "content": prompt_for(run)},
                                                 {"role": "assistant", "content": run["prose"]}]},
                                   ensure_ascii=False) + "\n")
                kept += 1
    print(f"{kept}/{len(score)} sessions kept -> {args.out}")


if __name__ == "__main__":
    main()
