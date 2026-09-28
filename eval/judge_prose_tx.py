"""Transcript-grounded check of a prose abstract: every cited sentence against the transcript
around its own citations.

eval/judge_prose.py checks prose against the gold NOTES. A fact the student found and the gold's
note-taker left out then reads as a fabrication: of 8 sampled "fabrications" on MiniCPM5, at least
5 were said in the transcript (殯儀館 at 39:47, 巴拿馬 300萬 at 58:22, C-PEM at 1:01:18, ...). So
that judge's error count measures agreement with the gold's selection, not faithfulness.

Here each sentence is judged against the transcript lines from SPAN_BEFORE seconds before its
earliest citation to SPAN_AFTER after its latest (capped), which is where a cited claim lives. A
sentence with no citation is not judged; its share is reported. Coverage still comes from
eval/judge_prose.py: it asks whether the gold's key points were conveyed, which the notes answer.

Verdicts per sentence: supported / contradicted (wrong figure, status, person, pairing) /
unsupported (nothing in the excerpt backs it).
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.ingest import parse_line, resolve_citation  # noqa: E402
from summarizer.pipeline import ChatClient  # noqa: E402

SPAN_BEFORE, SPAN_AFTER, MAX_LINES = 30, 150, 120
CITE = re.compile(r"[\[［]([\d:,\s]+)[\]］]")

PROMPT = """你是嚴格的事實查核員。以下是會議逐字稿的片段（語音辨識結果，可能有錯字、同音字），以及一句根據該會議寫的紀要。

## 逐字稿片段
{excerpt}

## 待查核句子
{sentence}

判斷這句話的每個事實（數字、條號、日期、人名或機關、誰主張什麼、通過或保留等結果）是否被逐字稿片段支持：
- supported：全部有依據（同音字、語音辨識錯字視為相同）；
- contradicted：至少一處與片段相矛盾（數字或結果錯誤、張冠李戴、把建議寫成決議、把保留寫成通過）；
- unsupported：沒有矛盾，但至少一個具體事實在片段中找不到依據。

只輸出 JSON：{{"verdict": "supported|contradicted|unsupported", "why": "簡短理由"}}"""


def sentences(prose):
    """Split on 。！？ keeping each sentence's citations with it."""
    parts = re.split(r"(?<=[。！？])", prose.replace("\n", ""))
    return [p.strip() for p in parts if len(re.sub(CITE, "", p).strip()) >= 8]


def excerpt(sentence, lines):
    idx = []
    for group in CITE.findall(sentence):
        for ts in group.split(","):
            i = resolve_citation(ts.strip(), lines)
            if i is not None:
                idx.append(i)
    if not idx:
        return None
    t0, t1 = lines[min(idx)].start_s - SPAN_BEFORE, lines[max(idx)].start_s + SPAN_AFTER
    keep = [l.render() for l in lines if t0 <= l.start_s <= t1]
    return "\n".join(keep[:MAX_LINES])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--split", default="data/split_v2.json")
    ap.add_argument("--transcripts", default="data/v2/transcripts")
    ap.add_argument("--judge-url", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()
    chat = ChatClient(args.judge_url, args.judge_model, max_tokens=300)
    jobs = []
    for sid in json.load(open(args.split))["heldout"]:
        p = os.path.join(args.candidate, sid + ".json")
        if not os.path.exists(p):
            continue
        prose = json.load(open(p, encoding="utf-8")).get("prose", "")
        lines = [parse_line(l) for l in open(os.path.join(args.transcripts, sid + ".txt"),
                                              encoding="utf-8").read().splitlines() if l.strip()]
        for s in sentences(prose):
            jobs.append((sid, s, excerpt(s, lines)))

    def one(job):
        sid, s, ex = job
        if ex is None:
            return {"id": sid, "sentence": s, "verdict": "uncited"}
        for _ in range(3):
            m = re.search(r"\{.*\}", chat([{"role": "user", "content": PROMPT.format(excerpt=ex, sentence=s)}]), re.S)
            try:
                v = json.loads(m.group(0))
                if v.get("verdict") in ("supported", "contradicted", "unsupported"):
                    return {"id": sid, "sentence": s, **v}
            except (AttributeError, json.JSONDecodeError):
                pass
        return {"id": sid, "sentence": s, "verdict": "unparsed"}

    rows = list(ThreadPoolExecutor(args.parallel).map(one, jobs))
    sess = sorted({r["id"] for r in rows})
    judged = [r for r in rows if r["verdict"] in ("supported", "contradicted", "unsupported")]
    c = {k: sum(r["verdict"] == k for r in judged) for k in ("supported", "contradicted", "unsupported")}
    n = max(1, len(judged))
    print(f"{args.candidate}: n={len(sess)} sentences {len(rows)} judged {len(judged)} "
          f"supported {c['supported']/n:.0%} contradicted {c['contradicted']/n:.0%} unsupported {c['unsupported']/n:.0%} "
          f"| contradicted/session {c['contradicted']/len(sess):.2f} unsupported/session {c['unsupported']/len(sess):.2f} "
          f"| uncited {sum(r['verdict'] == 'uncited' for r in rows)/max(1, len(rows)):.0%}")
    json.dump({"candidate": args.candidate, "rows": rows}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
