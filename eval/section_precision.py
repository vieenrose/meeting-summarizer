"""Section-aware check of structured minutes: is an item filed under 決議事項 really a decision the
meeting took, and is an item under 待辦與負責人 really an assigned task?

judge_prose_tx.py judges each statement alone, without its section, so "建議設定餐補" passes there
even when it is filed as a decision -- a proposal written as a decision is only wrong because of
where it sits. Here the judge sees the section's claim and the transcript around the item's
citation, and answers one of:
  decision section: decided / proposed (only proposed or discussed) / unsupported
  action section:   assigned (someone is to do it, or a deadline is set) / idea / unsupported
Precision = decided (or assigned) / judged.
"""
import argparse
import collections
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.judge_prose_tx import excerpt  # noqa: E402
from summarizer.ingest import parse_line  # noqa: E402
from summarizer.pipeline import ChatClient  # noqa: E402

SECTION = re.compile(r"【(.+?)】")
ASK = {
    "決議事項": ("這一項被列為「會議作成的決議」。根據逐字稿片段判斷：\n"
             "- decided：會中確實作成這個決定（宣布通過、決定、定案、同意照辦）；\n"
             "- proposed：只是有人提出或討論，沒有作成決定；\n"
             "- unsupported：片段中找不到這件事。",
             ("decided", "proposed", "unsupported")),
    "待辦與負責人": ("這一項被列為「已指派的待辦事項」。根據逐字稿片段判斷：\n"
               "- assigned：會中確實要求某人或某單位去做，或訂了期限；\n"
               "- idea：只是提出的想法或建議，沒有人被指派；\n"
               "- unsupported：片段中找不到這件事。",
               ("assigned", "idea", "unsupported")),
}
PROMPT = """以下是會議逐字稿的片段（語音辨識結果，可能有錯字），以及會議紀錄中的一項。

## 逐字稿片段
{excerpt}

## 紀錄項目
{item}

{ask}

只輸出 JSON：{{"verdict": "{choices}", "why": "簡短理由"}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--transcripts", default="data/v2/transcripts")
    ap.add_argument("--judge-url", required=True)
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=8)
    a = ap.parse_args()
    chat = ChatClient(a.judge_url, a.judge_model, max_tokens=300)
    jobs = []
    for sid in json.load(open(a.split))["heldout"]:
        p = os.path.join(a.candidate, sid + ".json")
        if not os.path.exists(p):
            continue
        lines = [parse_line(l) for l in open(os.path.join(a.transcripts, sid + ".txt"), encoding="utf-8")
                 .read().splitlines() if l.strip()]
        sec = None
        for line in json.load(open(p, encoding="utf-8"))["minutes"].splitlines():
            if m := SECTION.search(line):
                sec = m.group(1)
                continue
            if sec in ASK and line.startswith("- ") and line[2:].strip() != "無":
                ex = excerpt(line, lines)
                if ex:
                    jobs.append((sid, sec, line[2:].strip(), ex))

    def one(job):
        sid, sec, item, ex = job
        ask, choices = ASK[sec]
        for _ in range(3):
            reply = chat([{"role": "user", "content": PROMPT.format(excerpt=ex, item=item, ask=ask, choices="|".join(choices))}])
            found = re.findall(r"\{[^{}]*\"verdict\"[^{}]*\}", reply)
            try:
                v = json.loads(found[-1])
                if v.get("verdict") in choices:
                    return {"id": sid, "section": sec, "item": item, **v}
            except (IndexError, json.JSONDecodeError):
                pass
        return {"id": sid, "section": sec, "item": item, "verdict": "unparsed"}

    rows = list(ThreadPoolExecutor(a.parallel).map(one, jobs))
    json.dump({"candidate": a.candidate, "rows": rows}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for sec, (_, choices) in ASK.items():
        c = collections.Counter(r["verdict"] for r in rows if r["section"] == sec)
        n = sum(c[x] for x in choices)
        print(f"{a.candidate} {sec}: items {n}, precision {c[choices[0]] / max(1, n):.0%} "
              f"({choices[1]} {c[choices[1]] / max(1, n):.0%}, unsupported {c['unsupported'] / max(1, n):.0%})")


if __name__ == "__main__":
    main()
