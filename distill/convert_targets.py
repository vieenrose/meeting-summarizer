"""Title and prose targets for the style-conversion turns, written by a model from journals.

For each journal (the teacher's v5 journals, and the student's own, so the student learns to
convert the kind of notes it actually writes), the model is given VoxSumDroid's two calls exactly
(eval/conversion_prompts.py). The same script produces a student's outputs for evaluation, by
pointing --url at the student instead of the teacher.

Output: one JSON per session with notes, title_raw, prose_raw (judged by eval/judge_prose_notes.py
before becoming training rows in distill/build_convert_sft.py).
"""
import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.conversion_prompts import PROSE_MAX, TITLE_MAX, compact_notes, prose_prompt, title_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journals", nargs="+", required=True, help="run dirs whose JSONs carry 'notes'")
    ap.add_argument("--only", help="split JSON with a 'heldout' list: restrict to these sessions")
    ap.add_argument("--urls", required=True, help="comma-separated OpenAI-compatible base URLs")
    ap.add_argument("--model", default="q38")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--max-journal-chars", type=int, default=0,
                    help="compact each journal to this many characters first (a 4k-context runtime)")
    a = ap.parse_args()
    keep = set(json.load(open(a.only))["heldout"]) if a.only else None
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for d in a.journals:
        for f in sorted(glob.glob(f"{d}/*.json")):
            sid = os.path.basename(f)[:-5]
            if keep is not None and sid not in keep:
                continue
            notes = json.load(open(f, encoding="utf-8")).get("notes") or []
            if notes and a.max_journal_chars:
                notes = compact_notes(notes, a.max_journal_chars)
            if notes:
                jobs.append((f"{os.path.basename(d)}__{sid}" if len(a.journals) > 1 else sid, notes))
    urls = a.urls.split(",")

    def gen(i, content, max_tokens):
        r = requests.post(urls[i % len(urls)].rstrip("/") + "/chat/completions", timeout=900, json={
            "model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0.2, "chat_template_kwargs": {"enable_thinking": False}}).json()
        return re.sub(r"<think>.*?</think>", "", r["choices"][0]["message"]["content"] or "", flags=re.S).strip()

    def one(ij):
        i, (key, notes) = ij
        p = os.path.join(a.out, key + ".json")
        if os.path.exists(p):
            return
        try:
            rec = {"notes": notes, "title_raw": gen(i, title_prompt(notes), TITLE_MAX),
                   "prose_raw": gen(i, prose_prompt(notes), PROSE_MAX)}
        except Exception as e:                      # e.g. a journal longer than a server slot
            print(f"skip {key}: {str(e)[:120]}", flush=True)
            return
        json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False)

    list(ThreadPoolExecutor(a.parallel).map(one, enumerate(jobs)))
    print(f"{len(jobs)} journals -> {a.out}")


if __name__ == "__main__":
    main()
