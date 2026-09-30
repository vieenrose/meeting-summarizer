"""On-policy correction pairs for DPO: the student's own reading turns, corrected by the teacher.

The fine-tuned student (runs/student/onpol-ft-ep0) read the training sessions with the deployment
harness; the judge flagged its contradicted notes (reports/judge_prose_tx_notes-onpol-ft-ep0.json).
For every window with at least one contradicted note, Qwen3.8-27B (NInfer) gets the transcript
window, the student's actions and the judge's reasons, and rewrites the actions in the same format:
correct notes kept as they are, wrong ones fixed from the transcript or removed. The pair is
(chosen = corrected, rejected = student), both conditioned on the same student context, so DPO
learns to prefer the fix in exactly the situations the student gets wrong.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.realtime_agent import ACT, NOTE, windows_of  # noqa: E402
from summarizer.ingest import parse_line, resolve_citation  # noqa: E402

PROMPT = """以下是一段會議逐字稿（語音辨識結果，可能有錯字；講者標籤 S1、S2 不可靠），以及一位助理讀完這段後寫的筆記動作。其中部分筆記經查核與逐字稿矛盾，理由附在後面。

## 逐字稿片段
{window}

## 助理的動作
{reply}

## 經查核與逐字稿矛盾的筆記
{bad}

請輸出改正後的完整動作：
- 沒有被列為矛盾的 NOTE 原樣保留，一字不改。
- 矛盾的 NOTE 依逐字稿改正；無法從逐字稿確定就整行刪除。
- 格式不變：NOTE [時間] (類型) 內容。類型為 DECISION、ACTION、NUMBER、OPEN-ISSUE 或 -。每則不超過 40 字。
- 不要寫出原文沒有明說的機關或人名，不要用 S1、S2；建議不是決議，保留不是通過。
- 最後一行是 NEXT。只輸出動作，不要解釋。"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="runs/student/onpol-ft-ep0")
    ap.add_argument("--judged", default="reports/judge_prose_tx_notes-onpol-ft-ep0.json")
    ap.add_argument("--transcripts", default="data/v2/transcripts")
    ap.add_argument("--urls", default="http://127.0.0.1:8120/v1,http://127.0.0.1:8121/v1")
    ap.add_argument("--out", default="data/train/agent_dpo_pairs.jsonl")
    ap.add_argument("--parallel", type=int, default=8)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    wtok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B-FP8")
    count = lambda t: len(wtok.encode(t, add_special_tokens=False))  # noqa: E731
    why = collections.defaultdict(dict)
    for r in json.load(open(a.judged, encoding="utf-8"))["rows"]:
        if r["verdict"] == "contradicted":
            why[r["id"]][r["sentence"]] = r.get("why", "")
    jobs = []
    for f in sorted(glob.glob(f"{a.student}/ivod_*.json")):
        sid = os.path.basename(f)[:-5]
        rec = json.load(open(f, encoding="utf-8"))
        lines = [parse_line(l) for l in open(os.path.join(a.transcripts, sid + ".txt"), encoding="utf-8")
                 .read().splitlines() if l.strip()]
        wins = windows_of(lines, count)
        replies = {t["window"]: t["reply"] for t in rec["trace"] if "reply" in t and t["window"] != "overview"}
        for n in rec["notes"]:
            key = n["text"].rstrip("。") + f" [{n['ts']}]。"
            if key in why[sid]:
                n["why"] = why[sid][key]
        by_win = collections.defaultdict(list)
        for n in rec["notes"]:
            if n.get("why") is not None:
                by_win[n["window"]].append(n)
        for k, bad in by_win.items():
            if k not in replies or k > len(wins):
                continue
            window = "\n".join(l.render() for l in wins[k - 1])
            badtxt = "\n".join(f"- [{n['ts']}] {n['text']} —— 理由：{n['why']}" for n in bad)
            jobs.append((sid, k, window, replies[k], badtxt, lines))

    urls = a.urls.split(",")

    def one(ij):
        i, (sid, k, window, reply, badtxt, lines) = ij
        r = requests.post(urls[i % len(urls)] + "/chat/completions", timeout=600, json={
            "model": "q38", "temperature": 0.2, "max_tokens": 800,
            "messages": [{"role": "user", "content": PROMPT.format(window=window, reply=reply, bad=badtxt)}],
            "chat_template_kwargs": {"enable_thinking": False}}).json()
        out = re.sub(r"<think>.*?</think>", "", r["choices"][0]["message"]["content"] or "", flags=re.S).strip()
        acts = [l for l in out.splitlines() if ACT.match(l)]
        notes_ok = all((n := NOTE.match(ACT.match(l).group(2).strip())) and resolve_citation(n.group(1), lines) is not None
                       for l in acts if ACT.match(l).group(1) == "NOTE")
        if not acts or acts[-1].strip() != "NEXT" or not notes_ok:
            return None
        chosen = "\n".join(acts)
        return {"session": sid, "window": k, "chosen": chosen, "rejected": reply} if chosen != reply.strip() else None

    with ThreadPoolExecutor(a.parallel) as ex, open(a.out, "w", encoding="utf-8") as fo:
        kept = 0
        for res in ex.map(one, enumerate(jobs)):
            if res:
                fo.write(json.dumps(res, ensure_ascii=False) + "\n")
                kept += 1
    print(f"windows with a contradicted note: {len(jobs)}, pairs kept: {kept}")


if __name__ == "__main__":
    main()
