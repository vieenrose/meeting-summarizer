"""Prompt pool for the multi-task GRPO (distill/grpo_multi.py): the three tasks the deployed model
performs, on training sessions only.

  read   one reading turn in its post-restart form: [system v5] [journal before window k] [NEXT]
         [window k]. Reference: the teacher's notes for window k (v7 relabeled types), for recall.
  prose  VoxSumDroid's notes -> prose call on a journal (teacher's or the student's own).
  title  VoxSumDroid's notes -> title call on the same journals.
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.conversion_prompts import prose_prompt, title_prompt  # noqa: E402
from eval.realtime_agent import SYSTEM_V5, render, windows_of  # noqa: E402
from summarizer.ingest import parse_line  # noqa: E402


def compact(notes, count, budget=2500):
    key = {"DECISION": 0, "OPEN-ISSUE": 1, "ACTION": 2}
    order = sorted(range(len(notes)), key=lambda i: (key.get((notes[i].get("tag") or "").upper(), 3), -i))
    chosen, used = set(), 0
    for i in order:
        t = count(render(notes[i]))
        if used + t <= budget:
            chosen.add(i)
            used += t
    rest = len(notes) - len(chosen)
    text = "\n".join(render(notes[i]) for i in sorted(chosen)) + (f"\n（另有 {rest} 則較早的筆記未列出）" if rest else "")
    return text or "（尚無筆記）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/teacher/rt-q38-v7")
    ap.add_argument("--student-journals", default="runs/student/onpol-ft-ep0")
    ap.add_argument("--transcripts", default="data/v2/transcripts,data/alimeeting/transcripts")
    ap.add_argument("--n-read", type=int, default=1200)
    ap.add_argument("--n-conv", type=int, default=400, help="per conversion task")
    ap.add_argument("--out", default="data/train/rl_prompts.jsonl")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    wtok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B-FP8")
    count = lambda t: len(wtok.encode(t, add_special_tokens=False))  # noqa: E731
    rng = random.Random(0)
    reads, journals = [], []
    for f in sorted(glob.glob(f"{a.teacher}/*.json")):
        sid = os.path.basename(f)[:-5]
        rec = json.load(open(f, encoding="utf-8"))
        path = next(os.path.join(d, sid + ".txt") for d in a.transcripts.split(",") if os.path.exists(os.path.join(d, sid + ".txt")))
        lines = [parse_line(l) for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]
        wins = windows_of(lines, count)
        notes = rec["notes"]
        journals.append((sid, notes))
        for k, win in enumerate(wins, 1):
            ref = [n for n in notes if n["window"] == k]
            msgs = [{"role": "system", "content": SYSTEM_V5},
                    {"role": "user", "content": "## 筆記本（至今）\n" + compact([n for n in notes if n["window"] < k], count)},
                    {"role": "assistant", "content": "NEXT"},
                    {"role": "user", "content": f"## 逐字稿片段 {k}\n" + "\n".join(l.render() for l in win)}]
            reads.append({"task": "read", "session": sid, "window": k, "messages": msgs,
                          "window_text": "\n".join(l.render() for l in win),
                          "ref": [{"ts": n["ts"], "tag": n.get("tag"), "text": n["text"]} for n in ref]})
    for f in sorted(glob.glob(f"{a.student_journals}/*.json")):
        notes = json.load(open(f, encoding="utf-8")).get("notes") or []
        if notes:
            journals.append((os.path.basename(f)[:-5], notes))
    rng.shuffle(reads)
    rng.shuffle(journals)
    rows = reads[:a.n_read]
    for sid, notes in journals[:a.n_conv]:
        rows.append({"task": "prose", "session": sid, "notes": notes,
                     "messages": [{"role": "user", "content": prose_prompt(notes)}]})
    for sid, notes in journals[-a.n_conv:]:
        rows.append({"task": "title", "session": sid, "notes": notes,
                     "messages": [{"role": "user", "content": title_prompt(notes)}]})
    rng.shuffle(rows)
    with open(a.out, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print({t: sum(r["task"] == t for r in rows) for t in ("read", "prose", "title")})


if __name__ == "__main__":
    main()
