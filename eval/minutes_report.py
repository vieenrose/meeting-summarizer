"""Per-section report for sectioned minutes (eval/journal_agent.py, eval/structured_cr.py).

Reads a run dir (each session's "minutes" with 【section】 headers) and the transcript-grounded
judgments of its sentences (reports/judge_prose_tx_<run>.json), and prints per section:
items, contradicted / unsupported shares, plus decision recall -- how many of the gold's
DECISION notes have a decisions-section item citing a line within 90 s of the gold note.
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.judge_prose_tx import sentences  # noqa: E402

SECTION = re.compile(r"【(.+?)】")
CITE = re.compile(r"[\[［](\d+:\d{2}(?::\d{2})?)")


def secs(t):
    p = [int(x) for x in t.split(":")]
    return p[0] * 3600 + p[1] * 60 + p[2] if len(p) == 3 else p[0] * 60 + p[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir name under runs/student")
    ap.add_argument("--gold", default="runs/v2/w4000")
    a = ap.parse_args()
    d = f"runs/student/{a.run}"
    verdict = {(r["id"], r["sentence"]): r["verdict"]
               for r in json.load(open(f"reports/judge_prose_tx_{a.run}.json", encoding="utf-8"))["rows"]}
    stats = collections.defaultdict(collections.Counter)
    hit = total = 0
    for f in sorted(os.listdir(d)):
        if not f.startswith("ivod_"):
            continue
        sid = f[:-5]
        run = json.load(open(os.path.join(d, f), encoding="utf-8"))
        sec, dec_ts = "?", []
        for line in run.get("minutes", "").splitlines():
            m = SECTION.search(line)
            if m:
                sec = m.group(1)
                continue
            if not line.startswith("- ") or line[2:].strip() == "無":
                continue
            item = line[2:].strip()
            item = item if item.endswith("。") else item + "。"
            for s in sentences(item):
                v = verdict.get((sid, s), "unjudged")
                stats[sec][v] += 1
            if sec.startswith("決議"):
                dec_ts += [secs(t) for t in CITE.findall(line)]
        gold = json.load(open(os.path.join(a.gold, f), encoding="utf-8"))
        for n in gold["notes"]:
            if n.get("tag") == "DECISION" or "(DECISION)" in n["text"]:
                total += 1
                hit += any(abs(secs(n["ts"]) - t) <= 90 for t in dec_ts)
    for sec, c in stats.items():
        j = c["supported"] + c["contradicted"] + c["unsupported"]
        print(f"{sec:<8} items {sum(c.values()):4d}  contradicted {c['contradicted'] / max(1, j):.0%}  "
              f"unsupported {c['unsupported'] / max(1, j):.0%}  unjudged {c['unjudged'] + c['uncited']}")
    print(f"decision recall vs gold DECISION notes: {hit}/{total} ({hit / max(1, total):.0%})")


if __name__ == "__main__":
    main()
