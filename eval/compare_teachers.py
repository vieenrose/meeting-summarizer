"""Score raw teacher outputs against the repaired gold corpus.

The 39 repaired gold sessions are a benchmark no teacher has seen: every defect the v1 teacher
made in them was found and fixed over 12 review rounds. A candidate teacher's raw output is scored
against that gold on the same sessions, alongside the v1 teacher's own raw output, so the question
"is a cheaper teacher good enough?" is answered by measurement.

Per teacher:
  pass          sessions passing the structural validator
  reward        summary reward against the gold summary (distill/rewards.py)
  notes-recall  share of the gold notes' figures and names present in the teacher's notes
  leaks         names in the output one character away from a name in the transcript -- the
                v1 defect class that cost the most repairs
  empty         windows with no notes
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.rewards import score_summary  # noqa: E402
from eval.compare_students import facts  # noqa: E402
from eval.validate_teacher import check  # noqa: E402
from summarizer.ingest import parse_line  # noqa: E402

SURNAMES = set("陳林黃張李王吳劉蔡楊許鄭謝郭洪曾邱廖賴徐周葉蘇莊呂江何蕭羅高潘簡朱鍾游彭詹胡施沈余"
               "盧梁趙顏柯翁魏孫戴范方宋鄧杜傅侯曹薛丁卓馬阮董唐溫藍蔣石古紀姚連馮歐程湯黎田康白"
               "涂尤巫韓龔嚴袁鐘")
TITLED = re.compile(r"([一-鿿]{3})(?=委員|部長|次長|署長|處長|召委|院長|主委|等\d|等[一二三四五六七八九十]+人)")


def leaks(text, transcript):
    tr = re.sub(r"\s", "", transcript)
    names = {m for m in TITLED.findall(tr) if m[0] in SURNAMES}
    found = set()
    for g in {m for m in TITLED.findall(re.sub(r"\s", "", text)) if m[0] in SURNAMES}:
        if g in tr:
            continue
        if any(sum(a != b for a, b in zip(g, n)) == 1 for n in names):
            found.add(g)
    return found


def main():
    specs = [a.split("=", 1) for a in sys.argv[1:]]
    sessions = sorted({f[:-5] for _, d in specs for f in os.listdir(d) if f.startswith("ivod_") and f.endswith(".json")})
    sessions = [s for s in sessions if all(os.path.exists(os.path.join(d, s + ".json")) for _, d in specs)]
    print(f"sessions compared: {len(sessions)} {sessions}")
    for label, d in specs:
        n = passed = empty = g = r = 0
        reward, leak_set, secs = 0.0, set(), 0.0
        for sid in sessions:
            tr_text = open(f"data/transcripts/{sid}.txt", encoding="utf-8").read()
            lines = [parse_line(l) for l in tr_text.splitlines() if l.strip()]
            gold = json.load(open(f"runs/gold/w4000/{sid}.json", encoding="utf-8"))
            run = json.load(open(os.path.join(d, sid + ".json"), encoding="utf-8"))
            n += 1
            passed += not check(run, lines, 0, 650, 0)
            empty += len(run.get("windows_without_notes") or [])
            reward += score_summary("\n".join(run.get("summary") or []), lines, gold["summary"])
            gf = facts(" ".join(x["text"] for x in gold["notes"]))
            body = re.sub(r"\s|,", "", " ".join(x["text"] for x in run.get("notes") or []))
            g += len(gf)
            r += sum(1 for f in gf if f in body)
            out = " ".join([x["text"] for x in run.get("notes") or []] + (run.get("summary") or []))
            leak_set |= {f"{sid}:{x}" for x in leaks(out, tr_text)}
            secs += run.get("elapsed_s", 0)
        print(f"{label:22} pass {passed}/{n}  reward {reward / n:.3f}  notes-recall {100 * r / max(1, g):.0f}%  "
              f"leaks {len(leak_set)}  empty {empty}  avg {secs / n / 60:.1f} min/session")
        if leak_set:
            print("   leaks:", " ".join(sorted(leak_set))[:300])


if __name__ == "__main__":
    main()
