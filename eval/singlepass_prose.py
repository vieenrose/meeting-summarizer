"""Single pass: the whole transcript in one prompt, the prose abstract out -- no map, no reduce.

Same instructions and length gate as the pipeline's prose (distill/write_prose.PROMPT), with the
transcript in place of the notes, so eval/judge_prose_tx.py compares it directly with map-reduce.
Run against a long-context model served locally (Flash-Next, 250k KV) or any OpenAI-compatible API.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.write_prose import MAX_CHARS, MIN_CHARS, PROMPT, check  # noqa: E402
from summarizer.ingest import parse_line  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default="http://127.0.0.1:1919/v1,http://127.0.0.1:2919/v1")
    ap.add_argument("--model", default="fn-local")
    ap.add_argument("--split", default="data/split_v2.json")
    ap.add_argument("--transcripts", default="data/v2/transcripts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--turns", type=int, default=2)
    a = ap.parse_args()
    urls = a.urls.split(",")
    os.makedirs(a.out, exist_ok=True)
    head = PROMPT.split("{notes}")[0].replace("整場會議依序寫下的筆記", "一場會議的完整逐字稿（語音辨識結果，可能有錯字；講者標籤不可靠）")
    tail = PROMPT.split("{notes}")[1].replace("根據這些筆記", "根據這份逐字稿").replace("筆記中有的內容", "逐字稿中有的內容").replace("時間必須來自筆記", "時間必須照抄逐字稿中真實存在的一行")

    def one(i_sid):
        i, sid = i_sid
        text = open(os.path.join(a.transcripts, sid + ".txt"), encoding="utf-8").read()
        lines = [parse_line(l) for l in text.splitlines() if l.strip()]
        msgs = [{"role": "user", "content": head + text + tail.format(lo=MIN_CHARS, hi=MAX_CHARS)}]
        best, probs = "", ["未產生"]
        for _ in range(a.turns):
            r = requests.post(urls[i % len(urls)] + "/chat/completions", timeout=3600, json={
                "model": a.model, "messages": msgs, "temperature": 0.2, "max_tokens": 2500,
                "chat_template_kwargs": {"enable_thinking": False}}).json()
            reply = r["choices"][0]["message"]["content"].strip()
            p = check(reply, lines)
            if not best or len(p) < len(probs):
                best, probs = reply, p
            if not p:
                break
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "請修正以下問題後重新輸出完整紀要：\n" + "\n".join(p)}]
        json.dump({"notes": [], "prose": best, "prose_problems": probs},
                  open(os.path.join(a.out, sid + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
        return sid, probs

    sids = json.load(open(a.split))["heldout"]
    with ThreadPoolExecutor(2 * len(urls)) as ex:
        for sid, p in ex.map(one, enumerate(sids)):
            print(sid, "ok" if not p else p, flush=True)


if __name__ == "__main__":
    main()
