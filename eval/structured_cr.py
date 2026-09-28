"""Read once, ask by section, then verify: a structured minutes from one cached transcript prefill.

On the phone the transcript would be prefilled while the meeting is recorded; here the llama-server
prompt cache plays that role -- every question below shares the transcript prefix, so only the
question is new work. Sections: decisions, actions and owners, held/open items, overview. Each
item cites the transcript line it rests on. Then a verification turn (Chain-of-Verification):
the model rereads its own list against the same transcript and drops or fixes items.

Output: a run dir whose "prose" is the verified minutes, one cited sentence per item, so
eval/judge_prose_tx.py scores it like any prose; the unverified draft is kept beside it.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

HEAD = ("以下是一場會議的完整逐字稿（語音辨識結果，可能有錯字、同音字；講者標籤 S1、S2 不可靠，"
        "請從內容判斷身分）。\n\n{transcript}\n\n")
SECTIONS = [
    ("決議事項", "列出本會議所有已作成的決議（通過、照案、刪減、凍結、准予等），不要列建議或尚未決定的事項。"),
    ("待辦與負責人", "列出本會議所有待辦事項，寫出負責的機關或人員與期限（若有）。"),
    ("保留與未決", "列出本會議中被保留、另行協商、擇期再議或仍有爭議未決的事項。"),
    ("會議概要", "用三到五句話說明這場會議的目的、主要爭點與結果。"),
]
RULE = ("\n每項一行，以「- 」開頭，句尾附上出處時間 [時間]，時間必須照抄逐字稿中真實存在的一行；"
        "數字、案號、結果照逐字稿寫，不要推測；沒有就寫「- 無」。只輸出清單。")
VERIFY = ("請逐項對照逐字稿檢查你剛才列出的清單：與逐字稿不符的項目（數字、對象、誰主張、建議或決議、通過或保留）"
          "請改正，找不到依據的項目請刪除。依相同格式輸出檢查後的完整清單。")


def ask(url, model, msgs, max_tokens=1200):
    r = requests.post(url + "/chat/completions", timeout=3600, json={
        "model": model, "messages": msgs, "temperature": 0.2, "max_tokens": max_tokens, "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False}}).json()
    return r["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--model", default="bonsai")
    ap.add_argument("--split", default="data/split_v2.json")
    ap.add_argument("--transcripts", default="data/v2/transcripts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    urls = a.urls.split(",")
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.out + "-draft", exist_ok=True)

    def one(i_sid):
        i, sid = i_sid
        url = urls[i % len(urls)]
        text = open(os.path.join(a.transcripts, sid + ".txt"), encoding="utf-8").read()
        head = HEAD.format(transcript=text)
        draft, final = [], []
        for name, q in SECTIONS:
            msgs = [{"role": "user", "content": head + q + RULE}]
            d = ask(url, a.model, msgs)
            v = ask(url, a.model, msgs + [{"role": "assistant", "content": d}, {"role": "user", "content": VERIFY}])
            draft.append(f"【{name}】\n{d}")
            final.append(f"【{name}】\n{v}")

        def as_prose(blocks):
            items = [l[2:].strip() for b in blocks for l in b.splitlines() if l.startswith("- ") and l[2:].strip() != "無"]
            return "".join(x if x.endswith("。") else x + "。" for x in items)
        for d, blocks in ((a.out, final), (a.out + "-draft", draft)):
            json.dump({"notes": [], "prose": as_prose(blocks), "minutes": "\n\n".join(blocks)},
                      open(os.path.join(d, sid + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
        return sid

    sids = json.load(open(a.split))["heldout"]
    with ThreadPoolExecutor(len(urls)) as ex:
        for s in ex.map(one, enumerate(sids)):
            print(s, flush=True)


if __name__ == "__main__":
    main()
