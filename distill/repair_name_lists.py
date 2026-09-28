"""Collapse proposer lists whose names the teacher resolved from world knowledge.

Sessions that open by reading every proposer aloud ("委員邱正軍等十九人、委員陳超明等十八人…")
come out of ASR with most names garbled. The teacher rewrote them as the real legislators --
韓國瑜等19人 for 邱正軍等十九人, 賴士葆 for 賴世寶 -- which is correct about the world and
underivable from the transcript, the same leak as 馬文軍 -> 馬文君, thirty times over.

Aligning each name back to its garble by position and headcount was tried and rejected: the lists
drift, and it produced confident wrong pairings. A wrong name is worse than no name. So a list that
contains any name absent from the transcript is replaced by what the transcript does support --
how many proposer groups were read. Names the transcript spells as the gold does are left alone
when they stand outside such a list.
"""
import argparse
import json
import os
import re

ENTRY = r"[一-鿿]{2,4}等\d+(?:餘)?人"
RUN = re.compile(rf"(?:委員)?{ENTRY}(?:[、，](?:委員)?{ENTRY}){{2,}}")
NAME = re.compile(r"([一-鿿]{2,4})等\d+(?:餘)?人")


def collapse(text, transcript):
    changed = []

    def sub(m):
        names = [n.lstrip("委員") for n in NAME.findall(m.group(0))]
        leaked = [n for n in names if n not in transcript]
        if not leaked:
            return m.group(0)
        changed.append(leaked)
        return f"{len(names)}組委員（提案人姓名辨識不清，未逐一列出）"

    return RUN.sub(sub, text), changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/gold/w4000")
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for f in sorted(os.listdir(args.run_dir)):
        if not f.endswith(".json") or f.endswith(".score.json"):
            continue
        sid = f[:-5]
        tr = open(os.path.join(args.transcripts, sid + ".txt"), encoding="utf-8").read()
        path = os.path.join(args.run_dir, f)
        d = json.load(open(path, encoding="utf-8"))
        leaked = []
        for n in d["notes"]:
            n["text"], c = collapse(n["text"], tr); leaked += c
        new = []
        for p in d["summary"]:
            p, c = collapse(p, tr); leaked += c; new.append(p)
        d["summary"] = new
        if d.get("prose"):
            d["prose"], c = collapse(d["prose"], tr); leaked += c
        if not leaked:
            continue
        names = sorted({x for group in leaked for x in group})
        print(f"{sid}: {len(leaked)} list(s) collapsed; unreadable names: {'、'.join(names)}")
        if not args.dry_run:
            d.setdefault("repairs", []).append({
                "stage": "derivability", "point": None, "kind": "REFERENCE LEAK",
                "before": "、".join(names), "after": "N組委員（提案人姓名辨識不清）",
                "why": "提案人名單中這些委員姓名在逐字稿裡是其他辨識結果，教師依真實立委名單改寫；"
                       "逐一對齊回辨識結果會產生錯配，故改記可從逐字稿得知的提案組數。"})
            json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
