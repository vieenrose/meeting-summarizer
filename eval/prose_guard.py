"""Deterministic post-check of a prose abstract against the notes its sentences cite.

Reduce is where most contradictions are made (transcript-judged: gold notes 7% -> student prose
from those notes ~20%). A sentence that states a figure or a status word none of its cited notes
contain has, by construction, made it up or moved it; this guard drops such sentences. Figures are
compared by value (distill/numerals.py), statuses by the word itself. Runs on the phone: no model.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.numerals import values  # noqa: E402

CITE = re.compile(r"[\[［]([\d:,\s]+)[\]］]")
STATUS = ("通過", "保留", "凍結", "解凍", "刪減", "刪除", "決議", "同意", "反對", "不予", "協商", "撤回", "照列", "延宕")


def guard(prose, notes):
    by_ts = {}
    for n in notes:
        by_ts.setdefault(n["ts"], []).append(n["text"])
    out, dropped = [], 0
    for para in prose.split("\n"):
        keep = ""
        for sent in re.split(r"(?<=[。！？])", para):
            if not sent.strip():
                continue
            cited = " ".join(t for g in CITE.findall(sent) for ts in g.split(",") for t in by_ts.get(ts.strip(), []))
            body = CITE.sub("", sent)
            bad = cited and (not values(body) <= values(cited)
                             or any(s in body and s not in cited for s in STATUS))
            if bad:
                dropped += 1
            else:
                keep += sent
        out.append(keep)
    return "\n".join(p for p in out if p.strip()), dropped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = n = 0
    for p in glob.glob(os.path.join(a.run_dir, "*.json")):
        r = json.load(open(p, encoding="utf-8"))
        r["prose"], k = guard(r.get("prose", ""), r["notes"])
        d += k
        n += 1
        json.dump(r, open(os.path.join(a.out, os.path.basename(p)), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"dropped {d} sentences over {n} sessions ({d / max(1, n):.1f}/session)")
