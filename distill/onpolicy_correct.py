"""On-policy correction data: the student's own notes for training windows, corrected by a
teacher that reads the window.

Synthetic corruptions (distill/build_verify_rows.py) taught a verifier that catches 87% of them
and fixes 1.2% of the student's real notes, and DPO on them left the contradiction rate at 30%:
the student's real errors are misreadings -- the wrong speaker or object for a figure, a sense
inverted (延宕 read as 過早), an ASR garble taken literally, a proposal written as a decision --
not swaps. So the negatives must be the student's own notes.

The corrector is the local Flash-Next, not Gemma-4-31B, which judges the evaluation. It is told to
keep every correct note verbatim, so an unchanged note is a no-op and the preference pair only
differs where something was wrong.
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.pipeline import parse_turn  # noqa: E402

PROMPT = """你是嚴格的會議筆記校對員。以下是一段會議逐字稿（語音辨識結果，可能有錯字、同音字；講者標籤 S1、S2 不可靠），以及一位助理根據它寫的筆記。

## 逐字稿
{window}

## 助理的筆記
{notes}

請逐則對照逐字稿校對，修正以下錯誤：
- 數字、金額、年度、條號、案號錯誤，或把數字寫到錯的對象、案號上；
- 把甲的發言寫成乙的（從內容判斷身分，例如主席、某委員、部長、局長；無法確定就寫「發言者」）；
- 意思寫反（例如延宕寫成提早、高寫成低）、把建議或主張寫成決議、把保留或協商寫成通過；
- 把語音辨識錯字當成字面意思而誤解；
- 逐字稿中找不到依據的內容（刪去該部分）。
正確的筆記一字不改原樣保留。語音辨識的錯字照抄不算錯誤，不要改成你認為正確的真實名稱。

依原格式輸出全部筆記（每則一行，以「- [時間] 」開頭，時間沿用原筆記），不要加任何說明。"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/train/rft_samples.jsonl")
    ap.add_argument("--out", default="data/train/onpolicy_corrected.jsonl")
    ap.add_argument("--judges", default="http://127.0.0.1:1919/v1,http://127.0.0.1:2919/v1")
    args = ap.parse_args()
    urls = args.judges.split(",")
    rows = [json.loads(l) for l in open(args.samples, encoding="utf-8")]
    done = set()
    if os.path.exists(args.out):
        done = {(d["session"], d["window"]) for d in map(json.loads, open(args.out, encoding="utf-8"))}

    def one(ir):
        i, r = ir
        student = r["samples"][0]
        notes = parse_turn(student).notes
        if not notes:
            return None
        window = r["messages"][1]["content"].split("TRANSCRIPT\n", 1)[-1]
        payload = {"model": "fn-local", "temperature": 0.0, "max_tokens": 2000,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [{"role": "user", "content": PROMPT.format(window=window, notes=student.strip())}]}
        try:
            c = requests.post(urls[i % len(urls)] + "/chat/completions", json=payload, timeout=900).json()["choices"][0]["message"]["content"]
        except Exception:                           # noqa: BLE001
            return None
        fixed = parse_turn(c).notes
        stamps = set(re.findall(r"^\[([\d:]+)\]", window, re.M))
        if not fixed or any(ts not in stamps for _, ts, _ in fixed):
            return None                             # corrector broke the format or invented an anchor
        return {"session": r["session"], "window": r["window"], "messages": r["messages"][:-1],
                "student": student.strip(), "corrected": c.strip(),
                "changed": sum(1 for a, b in zip(notes, fixed) if a[2] != b[2]) + abs(len(notes) - len(fixed)),
                "n": len(notes)}

    todo = [(i, r) for i, r in enumerate(rows) if (r["session"], r["window"]) not in done]
    with open(args.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(4 * len(urls)) as ex:
        for k, d in enumerate(ex.map(one, todo), 1):
            if d:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                f.flush()
            if k % 50 == 0:
                print(f"{k}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
