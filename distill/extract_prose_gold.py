"""Regenerate reduce targets from the annotated extracts alone.

The first extractive reduce was trained on gold prose written from ALL gold notes, while only 65%
of them survive as spans: its targets held facts absent from its input, which teaches invention
(its eval loss was 1.45 against 0.39 for the note-based reduce). Here the local Flash-Next writes
each target from the extracts only, with the same prose gate as the gold (distill/write_prose.check,
up to 3 turns), and sessions whose prose still fails the gate are dropped.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, ".")
from distill.write_prose import prompt_for, write_one  # noqa: E402
from summarizer.ingest import parse_line  # noqa: E402

rows = [json.loads(l) for l in open("data/train/extract_act_reduce_rows.jsonl", encoding="utf-8")]
URLS = ["http://127.0.0.1:1919/v1", "http://127.0.0.1:2919/v1"]


def notes_of(r):
    import re
    return [{"ts": m.group(1), "text": m.group(2)} for m in re.finditer(r"^\[(\d+:\d{2}(?::\d{2})?)\] (.+)$", r["messages"][0]["content"], re.M)]


def one(ir):
    i, r = ir
    url = URLS[i % 2]

    def chat(m):
        return requests.post(url + "/chat/completions", json={"model": "fn-local", "messages": m, "temperature": 0.3,
                             "max_tokens": 2500, "chat_template_kwargs": {"enable_thinking": False}},
                             timeout=900).json()["choices"][0]["message"]["content"]
    tdir = "data/v2/transcripts" if r["session"].startswith("ivod") else "data/alimeeting/transcripts"
    lines = [parse_line(l) for l in open(f"{tdir}/{r['session']}.txt", encoding="utf-8").read().splitlines() if l.strip()]
    run = {"notes": notes_of(r)}
    try:
        prose, problems, _ = write_one(chat, run, lines, turns=3)
    except Exception as e:                           # noqa: BLE001
        return None, repr(e)
    if problems or not prose:
        return None, problems
    return {**r, "teacher": "flash-next-from-extracts",
            "messages": [{"role": "user", "content": prompt_for(run)}, {"role": "assistant", "content": prose}]}, None


with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(one, enumerate(rows)))
ok = [r for r, _ in res if r]
with open("data/train/extract_act_reduce_fn.jsonl", "w", encoding="utf-8") as f:
    for r in ok:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(ok)}/{len(rows)} proses pass the gate", flush=True)
