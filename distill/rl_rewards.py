"""Rewards for the multi-task GRPO, one per task, each from one judge call per sample.

  read   all NOTE lines of a sample, judged at once against the transcript window: per note a
         faithfulness verdict (supported / unsupported / contradicted) and, for DECISION and
         ACTION, whether the type holds (decided / assigned). Plus recall of the teacher's notes
         for the window, so writing less is not a way to score higher.
  prose  all sentences of a sample, judged at once against the notes; plus form.
  title  a 1-5 score against the notes; over 20 characters scores as a 1.
The judge is any OpenAI-compatible server (Qwen3.8-27B on NInfer during training).
"""
import json
import re

import requests

from eval.conversion_prompts import CITE, clean_title, form
from eval.realtime_agent import ACT, NOTE, render, similar

READ_JUDGE = """以下是一段會議逐字稿（語音辨識結果，可能有錯字），以及助理讀完後寫的筆記。

## 逐字稿
{window}

## 筆記
{notes}

逐則判斷：
- faith：supported（逐字稿支持）、unsupported（找不到依據）、contradicted（與逐字稿矛盾：數字、對象、誰主張、結果錯誤，或張冠李戴）。
- type（只有類型為 DECISION 或 ACTION 的筆記要判斷）：DECISION 是否為本次會議明確作成的決定、ACTION 是否為已指派的待辦（有負責者或期限）；是寫 ok，不是寫 wrong（例如只是建議、討論、宣讀上次紀錄、朗讀法規、說明現況）。其他類型寫 na。
只輸出 JSON 陣列，每則一個物件：[{{"id": 1, "faith": "...", "type": "ok|wrong|na"}}, ...]"""

PROSE_JUDGE = """以下是一場會議的筆記，以及根據筆記寫成的摘要（已逐句編號）。

## 筆記
{notes}

## 摘要
{sentences}

逐句判斷是否忠於筆記：supported（筆記中都有，改寫合併可以）、unsupported（含筆記沒有的事實）、contradicted（與筆記矛盾，或把建議寫成決定、把未決寫成已決）。
只輸出 JSON 陣列：[{{"id": 1, "verdict": "..."}}, ...]"""

TITLE_JUDGE = """以下是一場會議的筆記，以及為這場會議取的標題。

## 筆記
{notes}

## 標題
{title}

評分 1 到 5：5 = 準確點出主要議題且具體；4 = 正確但略籠統；3 = 只抓到部分議題或過於籠統；2 = 主題偏差；1 = 錯誤或含筆記沒有的內容。
只輸出 JSON：{{"score": 1-5}}"""


def ask(url, content, max_tokens=800):
    r = requests.post(url.rstrip("/") + "/chat/completions", timeout=900, json={
        "model": "judge", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
        "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}).json()
    return re.sub(r"<think>.*?</think>", "", r["choices"][0]["message"]["content"] or "", flags=re.S)


def parse_json(text, opener):
    i, j = text.find(opener), text.rfind("]" if opener == "[" else "}")
    try:
        return json.loads(text[i:j + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def _secs(t):
    p = [int(x) for x in t.split(":")]
    return p[0] * 3600 + p[1] * 60 + p[2] if len(p) == 3 else p[0] * 60 + p[1]


def read_reward(url, sample, row, max_notes=6):
    lines = [l for l in sample.split("\n") if l.strip()]
    notes, bad = [], 0
    for l in lines:
        m = ACT.match(l)
        if not m:
            bad += 1
            continue
        if m.group(1) == "NOTE":
            n = NOTE.match(m.group(2).strip())
            if n and len(notes) < max_notes:
                notes.append({"ts": n.group(1), "tag": (n.group(2) or "").upper(), "text": n.group(3)})
            elif not n:
                bad += 1
    ends_ok = bool(lines) and lines[-1].strip() == "NEXT"
    ref = row["ref"]
    hit = sum(any(abs(_secs(r["ts"]) - _secs(n["ts"])) <= 60 and similar(r["text"], n["text"]) >= 0.25 for n in notes) for r in ref)
    recall = hit / len(ref) if ref else (1.0 if not notes else 0.5)
    score = 0.0
    if notes:
        listing = "\n".join(f"{i}. [{n['ts']}] ({n['tag'] or '-'}) {n['text']}" for i, n in enumerate(notes, 1))
        verdicts = parse_json(ask(url, READ_JUDGE.format(window=row["window_text"], notes=listing)), "[") or []
        by = {v.get("id"): v for v in verdicts if isinstance(v, dict)}
        for i, n in enumerate(notes, 1):
            v = by.get(i, {})
            score += {"supported": 1.0, "unsupported": -0.5, "contradicted": -2.0}.get(v.get("faith"), -0.5)
            if n["tag"] in ("DECISION", "ACTION"):
                score += {"ok": 0.5, "wrong": -1.0}.get(v.get("type"), 0.0)
        score /= max(len(ref), len(notes), 1)
    return score + 1.0 * recall - 0.5 * bad - (0 if ends_ok else 0.5), {"notes": len(notes), "recall": recall}


def prose_reward(url, sample, row):
    notes = row["notes"]
    body = sample.split("<turn|>")[0]
    sents = [s.strip() for s in re.split(r"(?<=[。！？])", body.replace("\n", "")) if len(CITE.sub("", s).strip()) >= 6]
    if not sents:
        return -2.0, {"sentences": 0}
    listing = "\n".join(f"{i}. {s}" for i, s in enumerate(sents, 1))
    verdicts = parse_json(ask(url, PROSE_JUDGE.format(notes="\n".join(map(render, notes)), sentences=listing)), "[") or []
    by = {v.get("id"): v.get("verdict") for v in verdicts if isinstance(v, dict)}
    faith = sum({"supported": 1.0, "unsupported": -1.0, "contradicted": -2.0}.get(by.get(i), -1.0) for i in range(1, len(sents) + 1)) / len(sents)
    fm = form(body, notes)
    form_pen = (1.0 if fm["bullets"] or fm["headers"] else 0.0) + fm["bad_citations"] / max(1, fm["citations"]) \
        + max(0.0, 0.8 - fm["cited_sentences"] / max(1, fm["sentences"])) + (0.0 if 150 <= fm["chars"] <= 1000 else 1.0)
    return faith - form_pen, {"sentences": len(sents), "faith": faith}


def title_reward(url, sample, row):
    title = clean_title(sample)
    if not title or len(title) > 20:
        return -1.0, {"chars": len(title)}
    v = parse_json(ask(url, TITLE_JUDGE.format(notes="\n".join(map(render, row["notes"])), title=title), 60), "{") or {}
    try:
        s = int(v.get("score"))
    except (TypeError, ValueError):
        s = 3
    return (s - 3) / 2, {"score": s}


def reward(url, task, sample, row):
    return {"read": read_reward, "prose": prose_reward, "title": title_reward}[task](url, sample, row)
