"""Meeting titles from the agent's journal, and a judge's score for them.

The agent writes notes, not a title. At stop, one short call gets the journal and returns a zh-TW
title of at most 20 characters -- VoxSumDroid's own call, reproduced exactly. The judge scores it
1-5 against the gold key points and the opening of the transcript (main topic right and specific,
nothing invented, length kept). The same journals can be titled by several models, which
separates the journal's quality from the titler's.

  python3 eval/title_eval.py --run runs/student/ali-ft-v5 --split data/split_ali20.json \
    --gold runs/v2/ali-gold --transcripts data/alimeeting/transcripts \
    --titler http://127.0.0.1:8140/v1:rt --judge http://127.0.0.1:8700/v1:judge --tag v5 --out reports/titles_ali_v5.json
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.conversion_prompts import compact_notes  # noqa: E402
from eval.realtime_agent import render  # noqa: E402

# VoxSumDroid's exact title call (core/reader/ReaderLane.title, v0.45.1): the whole journal, as
# ReaderProtocol.render writes it, in one user turn with no system turn.
TITLE = """以下是一場會議的筆記：

{journal}

為這場會議寫一個標題，不超過 20 個字。只輸出標題。"""

JUDGE = """以下是一場會議的重點（參考答案）與逐字稿開頭，以及系統為這場會議取的標題。

## 會議重點
{points}

## 逐字稿開頭
{opening}

## 標題
{title}

評分 1 到 5：
5 = 準確點出主要議題且具體；4 = 正確但略籠統或漏了次要主題；3 = 只抓到部分議題或過於籠統；2 = 主題偏差；1 = 錯誤或含捏造內容。
超過 20 字扣 1 分。只輸出 JSON：{{"score": 1-5, "why": "簡短理由"}}"""


def compact(notes, budget_chars=2500):
    key = {"DECISION": 0, "OPEN-ISSUE": 1, "ACTION": 2}
    order = sorted(range(len(notes)), key=lambda i: (key.get((notes[i].get("tag") or "").upper(), 3), -i))
    chosen, used = set(), 0
    for i in order:
        t = len(render(notes[i]))
        if used + t <= budget_chars:
            chosen.add(i)
            used += t
    return "\n".join(render(notes[i]) for i in sorted(chosen)) or "（尚無筆記）"


def chat(spec, content, max_tokens):
    url, model = spec.rsplit(":", 1)
    r = requests.post(url.rstrip("/") + "/chat/completions", timeout=600, json={
        "model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
        "temperature": 0.2, "chat_template_kwargs": {"enable_thinking": False}}).json()
    return re.sub(r"<think>.*?</think>", "", r["choices"][0]["message"]["content"] or "", flags=re.S).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--transcripts", default="data/v2/transcripts")
    ap.add_argument("--titler", required=True, help="url:model")
    ap.add_argument("--judge", required=True, help="url:model")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-journal-chars", type=int, default=0,
                    help="compact each journal to this many characters first (a 4k-context runtime)")
    a = ap.parse_args()
    sids = [s for s in json.load(open(a.split))["heldout"] if os.path.exists(f"{a.run}/{s}.json")]

    def one(sid):
        try:
            return one_(sid)
        except Exception as e:
            print(f"skip {sid}: {str(e)[:120]}", flush=True)
            return {"id": sid, "title": "", "chars": 0, "score": None}

    def one_(sid):
        notes = json.load(open(f"{a.run}/{sid}.json", encoding="utf-8"))["notes"]
        if a.max_journal_chars:
            notes = compact_notes(notes, a.max_journal_chars)
        journal = "\n".join(render(n) for n in notes)
        title = next((l for l in chat(a.titler, TITLE.format(journal=journal), 48).splitlines() if l.strip()), "").strip("「」\"'*# ")[:40]
        gold = json.load(open(f"{a.gold}/{sid}.json", encoding="utf-8"))
        points = "\n".join(gold.get("summary", [])[:8])
        opening = "\n".join(open(os.path.join(a.transcripts, sid + ".txt"), encoding="utf-8").read().splitlines()[:40])[:2500]
        for _ in range(3):
            m = re.findall(r"\{[^{}]*\"score\"[^{}]*\}", chat(a.judge, JUDGE.format(points=points, opening=opening, title=title), 200))
            try:
                v = json.loads(m[-1])
                return {"id": sid, "title": title, "chars": len(title), "score": int(v["score"]), "why": v.get("why", "")}
            except (IndexError, ValueError, KeyError):
                pass
        return {"id": sid, "title": title, "chars": len(title), "score": None}

    rows = list(ThreadPoolExecutor(8).map(one, sids))
    sc = [r["score"] for r in rows if r["score"] is not None]
    json.dump({"run": a.run, "tag": a.tag, "rows": rows}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{a.tag} {os.path.basename(a.run)}: n={len(rows)} mean score {sum(sc) / max(1, len(sc)):.2f}, "
          f">=4: {sum(s >= 4 for s in sc) / max(1, len(sc)):.0%}, over 20 chars: {sum(r['chars'] > 20 for r in rows)}")


if __name__ == "__main__":
    main()
