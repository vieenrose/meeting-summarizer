"""Where do the students' contradictions come from? Classify each contradicted minutes statement
(eval/judge_prose_tx.py report: the statement, and the judge's reason) into one error type, and
relate the error rate to the type of note the statement comes from.

Types:
  number      a figure, amount, ratio or date is itself wrong
  binding     the right figure or fact, attached to the wrong object (year, scope, article, item)
  attribution the wrong speaker, body or side (who proposed, who is responsible, whose position)
  outcome     the result or status is inverted (passed vs held or rejected, a proposal written as a
              decision, an open issue as settled, a negation dropped)
  citation    the content holds elsewhere in the meeting; the cited time is wrong
  asr         the transcript's speech-recognition error was copied (a misheard word or number)
  invented    a detail the transcript does not give, which makes the statement false
  other
The classifier reads only the statement and the judge's reason, so it is cheap and repeatable; its
labels are the judge's account of the error, not a second judgment.
"""
import argparse
import collections
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.pipeline import ChatClient  # noqa: E402

TYPES = ["number", "binding", "attribution", "outcome", "citation", "asr", "invented", "other"]
PROMPT = """以下是一句會議紀錄，以及審查者判定它與逐字稿矛盾的理由。

## 句子
{sentence}

## 矛盾理由
{why}

把錯誤歸為一類：
- number：數字、金額、比例、日期本身錯誤
- binding：數字或事實本身在逐字稿中出現，但套到錯誤的對象、年度、範圍、條文或項目（張冠李戴）
- attribution：說話者、機關或立場的歸屬錯誤（誰提議、誰負責、誰的主張）
- outcome：結果或狀態顛倒（通過與保留／否決互換、把建議寫成決定、把未決寫成已決、漏掉否定）
- citation：內容在會議別處成立，只是引用的時間錯誤
- asr：照抄了逐字稿的語音辨識錯誤（聽錯的字詞或數字）
- invented：加入逐字稿沒有的細節，導致句子不實
- other：以上皆非
只輸出 JSON：{{"type": "..."}}"""

CITE = re.compile(r"\[(\d+:\d{2}(?::\d{2})?)\]")


def note_tags(run_dir):
    """(session, ts, text prefix) -> note tag, to find which note a minutes statement comes from."""
    out = {}
    for f in os.listdir(run_dir):
        if f.endswith(".json"):
            for n in json.load(open(os.path.join(run_dir, f), encoding="utf-8")).get("notes", []):
                out[(f[:-5], n["ts"], n["text"][:8])] = (n.get("tag") or "-").upper()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list of run names: reports/judge_prose_tx_<run>.json, runs/student/<run>")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8700/v1")
    ap.add_argument("--per-run", type=int, default=250, help="contradicted statements classified per run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    chat = ChatClient(a.judge_url, "judge", max_tokens=60)
    rep = {}

    def classify(row):
        for _ in range(3):
            m = re.findall(r'"type"\s*:\s*"(\w+)"', chat([{"role": "user", "content": PROMPT.format(sentence=row["sentence"], why=row.get("why", ""))}]))
            if m and m[-1] in TYPES:
                return m[-1]
        return "other"

    pool = ThreadPoolExecutor(16)
    for run in a.runs.split(","):
        rows = json.load(open(f"reports/judge_prose_tx_{run}.json", encoding="utf-8"))["rows"]
        judged = [r for r in rows if r.get("verdict") in ("supported", "unsupported", "contradicted")]
        contra = [r for r in judged if r["verdict"] == "contradicted"]
        sample = random.Random(0).sample(contra, min(a.per_run, len(contra)))
        types = list(pool.map(classify, sample))
        # error rate by the type of the note the statement comes from
        tags = note_tags(f"runs/student/{run}") if os.path.isdir(f"runs/student/{run}") else {}
        by_tag = collections.defaultdict(lambda: [0, 0])
        for r in judged:
            ts = CITE.findall(r["sentence"])
            text = CITE.sub("", r["sentence"]).strip()
            tag = tags.get((r["id"], ts[0], text[:8])) if ts else None
            if tag:
                by_tag[tag][0] += 1
                by_tag[tag][1] += r["verdict"] == "contradicted"
        c = collections.Counter(types)
        rep[run] = {"statements": len(judged), "contradicted": len(contra),
                    "contradicted_share": round(len(contra) / max(1, len(judged)), 3),
                    "types": {t: round(c[t] / max(1, len(sample)), 3) for t in TYPES},
                    "by_note_type": {t: {"n": n, "contradicted": round(k / n, 3)} for t, (n, k) in sorted(by_tag.items()) if n >= 20},
                    "examples": {t: [s["sentence"] + " ⟶ " + s.get("why", "")[:160] for s, ty in zip(sample, types) if ty == t][:3] for t in TYPES}}
        print(run, json.dumps({k: rep[run][k] for k in ("contradicted_share", "types", "by_note_type")}, ensure_ascii=False), flush=True)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
