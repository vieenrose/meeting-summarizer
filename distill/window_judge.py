"""LLM check of one window's notes against that window's transcript: the faithfulness reward.

Lexical rewards cannot see the errors that matter here: the right figure attached to the wrong
article, a proposal credited to the wrong side, "passed" written for a clause that was held (see
eval/local_grounding.py, whose correlation with the prose judge was -0.14). A model reading the
window can. It returns which notes are wrong; the reward is the share that are not.

The judge is the local Flash-Next (no thinking). It is never the evaluation judge: held-out prose
is scored by Gemma-4-31B, so the policy is not rewarded and graded by the same model.
"""
import json
import re

import requests

PROMPT = """你是嚴格的會議筆記查核員。以下是一段會議逐字稿（語音辨識結果，可能有錯字），以及根據這段逐字稿寫的筆記。

## 逐字稿
{window}

## 筆記
{notes}

逐條查核每則筆記是否與逐字稿相符。以下情況算錯誤：
- 數字、金額、條號、日期與逐字稿不符，或張冠李戴（把甲案的數字寫到乙案）；
- 把某人的主張寫成另一人的，或把提案、建議寫成已通過的決議，或把保留寫成通過；
- 逐字稿中找不到依據的內容。
逐字稿的錯字照抄不算錯誤；講者標籤（S1、S2）不可靠，不能以此判斷張冠李戴。

只輸出 JSON：{{"wrong": [錯誤筆記的編號], "why": {{"編號": "簡短理由"}}}}"""


class WindowJudge:
    def __init__(self, url="http://127.0.0.1:1919/v1", model="fn-local", timeout=600):
        self.url, self.model, self.timeout = url.rstrip("/") + "/chat/completions", model, timeout

    def wrong(self, window_text, notes):
        """notes: list of note strings. Returns the set of 0-based indices judged wrong, or None
        when the judge could not be read (the caller then scores the sample neutrally)."""
        if not notes:
            return set()
        body = "\n".join(f"{i}. {n}" for i, n in enumerate(notes, 1))
        payload = {"model": self.model, "temperature": 0.0, "max_tokens": 800,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [{"role": "user", "content": PROMPT.format(window=window_text, notes=body)}]}
        for _ in range(2):
            try:
                c = requests.post(self.url, json=payload, timeout=self.timeout).json()["choices"][0]["message"]["content"]
                v = json.loads(re.search(r"\{.*\}", c, re.S).group(0))
                return {int(i) - 1 for i in v.get("wrong", []) if str(i).isdigit() and 1 <= int(i) <= len(notes)}
            except Exception:                      # noqa: BLE001
                continue
        return None
