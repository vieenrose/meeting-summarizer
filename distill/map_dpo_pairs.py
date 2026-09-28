"""DPO pairs for the map step from the student's own sampled notes, each note judged against the
transcript (eval/judge_notes_tx.py) -- the factuality-preference recipe (Tian et al. 2023) with
real student errors and a validated judge.

Per window: chosen = the sample with the lowest contradicted share (ties broken by more supported
notes), rejected = the highest. Kept only when the gap is at least MIN_GAP and the chosen sample
has as many notes as the median sample, so the preference is not "write fewer notes".
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/train/rft_samples.jsonl")
    ap.add_argument("--verdicts", default="reports/judge_notes_tx_rft_samples.json")
    ap.add_argument("--out", default="data/train/map_dpo_pairs.jsonl")
    ap.add_argument("--min-gap", type=float, default=0.15)
    args = ap.parse_args()
    samples = {(r["session"], r["window"]): r for r in map(json.loads, open(args.samples, encoding="utf-8"))}
    kept = total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for line in open(args.verdicts, encoding="utf-8"):
            v = json.loads(line)
            r = samples[(v["session"], v["window"])]
            stats = {}
            for si, vs in v["verdicts"].items():
                judged = [x for x in vs if x in ("supported", "contradicted", "unsupported")]
                if judged:
                    stats[int(si)] = (sum(x == "contradicted" for x in judged) / len(judged),
                                      -sum(x == "supported" for x in judged), len(vs))
            total += 1
            if len(stats) < 2:
                continue
            order = sorted(stats, key=lambda s: stats[s][:2])
            best, worst = order[0], order[-1]
            median_n = sorted(x[2] for x in stats.values())[len(stats) // 2]
            if stats[worst][0] - stats[best][0] < args.min_gap or stats[best][2] < median_n:
                continue
            f.write(json.dumps({"session": r["session"], "window": r["window"], "prompt": r["messages"][:-1],
                                "chosen": [{"role": "assistant", "content": r["samples"][best].strip()}],
                                "rejected": [{"role": "assistant", "content": r["samples"][worst].strip()}]},
                               ensure_ascii=False) + "\n")
            kept += 1
    print(f"{kept}/{total} windows -> {args.out}")


if __name__ == "__main__":
    main()
