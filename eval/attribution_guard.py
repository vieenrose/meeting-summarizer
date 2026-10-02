"""A mechanical guard against attribution errors (a quarter of the contradictions, see
eval/contradiction_types.py): a note that names a body or an official's role the transcript does not
name near the cited line gets the generic 官員 instead. The note keeps its claim; only an actor the
model inferred, rather than read, is dropped.

Writes a copy of a run with the guarded notes and minutes text, for eval/judge_prose_tx.py.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.build_contrast_pairs import ACTOR  # noqa: E402
from summarizer.ingest import parse_line  # noqa: E402


def secs(t):
    p = [int(x) for x in t.split(":")]
    return p[0] * 3600 + p[1] * 60 + p[2] if len(p) == 3 else p[0] * 60 + p[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--transcripts", default="data/v2/transcripts")
    ap.add_argument("--lines", type=int, default=2, help="lines either side of the cited one")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    changed = total = 0
    for f in sorted(os.listdir(a.run)):
        if not f.endswith(".json"):
            continue
        rec = json.load(open(os.path.join(a.run, f), encoding="utf-8"))
        lines = [parse_line(l) for l in open(os.path.join(a.transcripts, f[:-5] + ".txt"), encoding="utf-8").read().splitlines() if l.strip()]
        ts = [secs(l.render()[1:].split("]")[0]) for l in lines]
        for n in rec["notes"]:
            total += 1
            t = secs(n["ts"])
            i = min(range(len(ts)), key=lambda k: abs(ts[k] - t))
            near = "".join(l.render() for l in lines[max(0, i - a.lines):i + a.lines + 1])
            new = n["text"]
            for m in ACTOR.finditer(n["text"]):
                actor = m.group(0)
                # "行政院版", "國防部主管預算": a bill version or a budget, not who spoke
                if re.match(r"\s*(版|草案|提案|函|主管|所屬|預算|委員會|門|主任|處|司)", n["text"][m.end():]):
                    continue
                if actor in ("主席", "召委"):     # the chair is named by the chair's role; not guarded
                    continue
                if actor not in near:
                    new = new.replace(actor, "官員", 1)
            if new != n["text"]:
                changed += 1
                for key in ("prose", "minutes"):
                    if isinstance(rec.get(key), str):
                        rec[key] = rec[key].replace(n["text"], new)
                n["text"] = new
        json.dump(rec, open(os.path.join(a.out, f), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{a.run}: {changed}/{total} notes guarded ({changed / max(1, total):.1%})")


if __name__ == "__main__":
    main()
