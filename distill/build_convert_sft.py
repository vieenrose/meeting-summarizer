"""Training rows for the two conversion turns (title, prose) from teacher targets.

A prose target is kept only if the judge found every sentence supported by the notes (no
contradicted, at most one unsupported), and its form is the requested one: no bullets, no
headers, no citation outside the journal, at least 70 % of sentences cited, 150-1000 characters.
A title is kept if it is one non-empty line of at most 20 characters. Rows have the shape
distill/sft_agent.py trains on: the user turn, the teacher's reply, loss on the reply only.
"""
import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.conversion_prompts import clean_title, form, prose_prompt, title_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--judged", required=True, help="eval/judge_prose_notes.py report on --targets")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    judged = json.load(open(a.judged, encoding="utf-8"))["sessions"]
    stats = collections.Counter()
    with open(a.out, "w", encoding="utf-8") as fo:
        for f in sorted(glob.glob(f"{a.targets}/*.json")):
            key = os.path.basename(f)[:-5]
            sid = key.split("__")[-1]
            r = json.load(open(f, encoding="utf-8"))
            notes = r["notes"]
            title = clean_title(r["title_raw"])
            if title and len(title) <= 20 and "\n" not in r["title_raw"].strip():
                fo.write(json.dumps({"session": sid, "kind": "title", "loss_turns": [1], "messages": [
                    {"role": "user", "content": title_prompt(notes)}, {"role": "assistant", "content": title}]},
                    ensure_ascii=False) + "\n")
                stats["title_kept"] += 1
            else:
                stats["title_dropped"] += 1
            v = judged.get(key, {})
            fm = form(r["prose_raw"], notes)
            ok = (v.get("contradicted", 0) == 0 and v.get("unsupported", 0) <= 1 and v.get("supported", 0) > 0
                  and fm["bullets"] == 0 and fm["headers"] == 0 and fm["bad_citations"] == 0
                  and fm["cited_sentences"] >= 0.7 * max(1, fm["sentences"]) and 150 <= fm["chars"] <= 1000)
            if ok:
                fo.write(json.dumps({"session": sid, "kind": "prose", "loss_turns": [1], "messages": [
                    {"role": "user", "content": prose_prompt(notes)},
                    {"role": "assistant", "content": r["prose_raw"].split("<turn|>")[0].strip()}]},
                    ensure_ascii=False) + "\n")
                stats["prose_kept"] += 1
            else:
                stats["prose_dropped"] += 1
                stats["prose_dropped_contradicted"] += v.get("contradicted", 0) > 0
                stats["prose_dropped_form"] += not (fm["bullets"] == 0 and fm["headers"] == 0 and fm["bad_citations"] == 0)
    print(dict(stats))


if __name__ == "__main__":
    main()
