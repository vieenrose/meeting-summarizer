"""Find names in the gold that are one character away from a name in the transcript.

Every name leak this dataset has produced has the same shape. The ASR mishears one character of
a legislator's name, and the teacher, recognising the real person, writes the real spelling:
林國成 -> 李國成, 賴世寶 -> 賴士葆, 廖偉祥 -> 廖偉翔, 李昭傑 -> 李召叡. Ten review rounds kept finding
them one or two at a time, because a reviewer has to notice that a plausible name is not the one
on the page.

The shape is mechanical, so this checks it directly: a three-character string in the gold that
starts with a surname, does not occur in the session's transcript, and differs by exactly one
character from a three-character string that does. That is a leak unless the two are different
people, which is rare enough to review by hand.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

SURNAMES = set("陳林黃張李王吳劉蔡楊許鄭謝郭洪曾邱廖賴徐周葉蘇莊呂江何蕭羅高潘簡朱鍾游彭詹胡施沈余"
               "盧梁趙顏柯翁魏孫戴范方宋鄧杜傅侯曹薛丁卓馬阮董唐溫藍蔣石古紀姚連馮歐程湯黎田康白"
               "涂尤巫韓龔嚴袁鐘")
STRICT = True
CJK3 = re.compile(r"(?=([一-鿿]{3}))")


def names(text):
    return {m for m in CJK3.findall(text) if m[0] in SURNAMES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/gold/w4000")
    ap.add_argument("--transcripts", default="data/transcripts")
    args = ap.parse_args()
    total = 0
    for f in sorted(glob.glob(os.path.join(args.run_dir, "ivod_*.json"))):
        sid = os.path.basename(f)[:-5]
        tr = re.sub(r"\s", "", open(os.path.join(args.transcripts, sid + ".txt"), encoding="utf-8").read())
        tr_names = names(tr)
        # index transcript names by each one-character wildcard, e.g. 林?成
        index = defaultdict(set)
        for n in tr_names:
            for i in range(3):
                index[n[:i] + "?" + n[i + 1:]].add(n)
        d = json.load(open(f, encoding="utf-8"))
        gold = " ".join([n["text"] for n in d["notes"]] + d["summary"] + [d.get("prose") or ""])
        hits = {}
        for g in names(gold):
            if g in tr:
                continue
            near = set()
            for i in range(3):
                near |= index.get(g[:i] + "?" + g[i + 1:], set())
            # Require the variant to look like a name in the transcript, i.e. be followed by a
            # title somewhere, so ordinary words sharing two characters do not flood the list.
            if STRICT:
                near = {n for n in near if re.search(re.escape(n) + r"(委員|部長|次長|署長|處長|召委|院長|主委|等)", tr)}
            if near:
                hits[g] = near
        if hits:
            print(sid)
            for g, near in sorted(hits.items()):
                print(f"   gold {g}  transcript {'/'.join(sorted(near))}")
                total += 1
    print(f"{total} near-miss name(s)")


if __name__ == "__main__":
    main()
