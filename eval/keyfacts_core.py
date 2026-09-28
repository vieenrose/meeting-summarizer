"""Reduce silver key facts to the ones a 500-character summary is actually expected to carry.

The raw silver lists average 34 facts per session (up to 93) because extraction runs per window and
keeps everything checkable. A summary of about six points cannot cover 34 facts, so coverage scored
against the raw list tops out near 0.2-0.5 however good the summary is: the metric is mis-scaled,
not the pipeline failing. It can rank two pipelines against each other, but it cannot be a gate.

This produces a subset shaped like the human gold list (data/keyfacts/README.md: 8-15 facts,
decisions and actions first), so coverage becomes interpretable in absolute terms. It is still
silver, and still not the gate.
"""
import argparse
import glob
import json
import os

PRIORITY = {"DECISION": 0, "ACTION": 1, "NUMBER": 2, "OPEN": 3, "POSITION": 4}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/keyfacts_silver")
    ap.add_argument("--out", default="data/keyfacts_silver_core")
    ap.add_argument("--max-facts", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sizes = []
    for path in sorted(glob.glob(os.path.join(args.src, "*.json"))):
        data = json.load(open(path, encoding="utf-8"))
        facts = data["facts"]
        # Decisions and actions first, then the rest, but keep chronological order inside each tier
        # so the subset still spans the meeting.
        ranked = sorted(enumerate(facts), key=lambda kv: (PRIORITY.get(kv[1]["type"], 9), kv[0]))
        kept = sorted((i for i, _ in ranked[:args.max_facts]))
        data["facts"] = [dict(facts[i], id=n + 1) for n, i in enumerate(kept)]
        data["source"] = (f"{data.get('source', '')} | reduced to <= {args.max_facts} "
                          "decision/action-first facts for interpretable coverage")
        json.dump(data, open(os.path.join(args.out, os.path.basename(path)), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        sizes.append(len(data["facts"]))
    if sizes:
        print(f"{len(sizes)} sessions, {sum(sizes) / len(sizes):.1f} facts each "
              f"(was up to {args.max_facts * 8})")


if __name__ == "__main__":
    main()
