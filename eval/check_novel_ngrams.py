"""Find character strings in the gold that no transcript in the corpus ever contains.

The dominant defect in this dataset is the teacher replacing garbled ASR with the real world:
李昭傑 -> 李召叡, 列壘建 -> 慶富獵雷艦, 莊仲泰 -> 卓榮泰. check_derivable.py catches these only
when a title follows the name. This check has no such dependency.

A summary paraphrases, so an arbitrary gold 3-gram will often be absent from its own
transcript. But ordinary Chinese recurs: across 78 hours of committee transcripts, almost every
common 3-gram occurs somewhere. A 3-gram occurring in NO transcript at all is overwhelmingly a
proper noun the audio never produced -- the exact leak. Candidates are ranked by how many gold
surfaces repeat them, since a leak is usually carried from notes into summary and prose.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

CJK = re.compile(r"[一-鿿]+")


def grams(text, n=3):
    for run in CJK.findall(text):
        for i in range(len(run) - n + 1):
            yield run[i:i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/gold/w4000")
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--reference", default="data/reference")
    args = ap.parse_args()

    corpus = set()
    for f in glob.glob(os.path.join(args.transcripts, "*.txt")):
        corpus.update(grams(re.sub(r"\s", "", open(f, encoding="utf-8").read())))

    total = 0
    for f in sorted(glob.glob(os.path.join(args.run_dir, "ivod_*.json"))):
        sid = os.path.basename(f)[:-5]
        d = json.load(open(f, encoding="utf-8"))
        surfaces = [("notes", n["text"]) for n in d["notes"]] + \
                   [("summary", s) for s in d["summary"]] + [("prose", d.get("prose") or "")]
        # Summary wording that no speaker used is absent from every transcript too, and swamps the
        # list. What separates a leak is that the CLEANER transcript of the same audio contains it:
        # the teacher wrote what was really said, not what our ASR heard.
        ref_path = os.path.join(args.reference, sid + ".txt")
        ref = set(grams(re.sub(r"\s", "", open(ref_path, encoding="utf-8").read()))) \
            if os.path.exists(ref_path) else set()
        found = defaultdict(set)
        for where, text in surfaces:
            for g in grams(text):
                if g not in corpus and g in ref:
                    found[g].add(where)
        if not found:
            continue
        # Merge overlapping novel 3-grams into the longer strings they come from.
        spans = defaultdict(set)
        for where, text in surfaces:
            for run in CJK.findall(text):
                i = 0
                while i < len(run) - 2:
                    if run[i:i + 3] in found:
                        j = i
                        while j < len(run) - 2 and run[j:j + 3] in found:
                            j += 1
                        spans[run[i:j + 2]].add(where)
                        i = j
                    else:
                        i += 1
        print(f"\n{sid}:")
        for s, where in sorted(spans.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            print(f"   {s:14} {','.join(sorted(where))}")
            total += 1
    print(f"\n{total} novel string(s)")


if __name__ == "__main__":
    main()
