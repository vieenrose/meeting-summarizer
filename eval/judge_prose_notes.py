"""Is a prose summary faithful to the notes it was written from? (the style-conversion check)

judge_prose_tx.py checks statements against the transcript. Here the source of truth is the
journal itself: prose written from notes must say nothing the notes do not say. Every sentence is
judged against the whole journal: supported / contradicted (says something the notes contradict:
wrong figure, owner, outcome; a proposal turned into a decision) / unsupported (a fact absent from
the notes). Form is measured too (eval/conversion_prompts.form): bullets, headers, citations.

Input: a dir of JSON files with "notes" and "prose_raw" (distill/convert_targets.py writes them;
so does eval/convert_eval.sh for students).
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.conversion_prompts import CITE, clean_prose, form  # noqa: E402
from eval.realtime_agent import render  # noqa: E402
from summarizer.pipeline import ChatClient  # noqa: E402

PROMPT = """以下是一場會議的筆記，以及根據筆記寫成的摘要中的一句。

## 筆記
{notes}

## 摘要中的一句
{sentence}

這句話是否忠於筆記？
- supported：每個事實都能在筆記中找到（改寫、合併可以）；
- contradicted：與筆記矛盾（數字、對象、誰主張、結果錯誤，或把建議寫成決定、把未決寫成已決）；
- unsupported：沒有矛盾，但含有筆記中沒有的事實。

只輸出 JSON：{{"verdict": "supported|contradicted|unsupported", "why": "簡短理由"}}"""


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[。！？])", text.replace("\n", "")) if len(CITE.sub("", s).strip()) >= 6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--judge-url", required=True)
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=16)
    a = ap.parse_args()
    chat = ChatClient(a.judge_url, a.judge_model, max_tokens=300)
    recs = {os.path.basename(f)[:-5]: json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(f"{a.dir}/*.json"))}
    jobs = []
    for sid, r in recs.items():
        notes_txt = "\n".join(map(render, r["notes"]))
        for s in sentences(clean_prose(r["prose_raw"], r["notes"])):
            jobs.append((sid, s, notes_txt))

    def one(job):
        sid, s, notes_txt = job
        for _ in range(3):
            found = re.findall(r"\{[^{}]*\"verdict\"[^{}]*\}", chat([{"role": "user", "content": PROMPT.format(notes=notes_txt, sentence=s)}]))
            try:
                v = json.loads(found[-1])
                if v.get("verdict") in ("supported", "contradicted", "unsupported"):
                    return {"id": sid, "sentence": s, **v}
            except (IndexError, json.JSONDecodeError):
                pass
        return {"id": sid, "sentence": s, "verdict": "unparsed"}

    rows = list(ThreadPoolExecutor(a.parallel).map(one, jobs))
    per = collections.defaultdict(collections.Counter)
    for r in rows:
        per[r["id"]][r["verdict"]] += 1
    forms = {sid: form(r["prose_raw"], r["notes"]) for sid, r in recs.items()}
    json.dump({"dir": a.dir, "rows": rows, "sessions": {s: dict(c) for s, c in per.items()}, "form": forms},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    c = collections.Counter(r["verdict"] for r in rows)
    n = max(1, c["supported"] + c["contradicted"] + c["unsupported"])
    F = forms.values()
    tot = lambda k: sum(f[k] for f in F)  # noqa: E731
    print(f"{a.dir}: n={len(recs)} sentences {n} supported {c['supported'] / n:.0%} contradicted "
          f"{c['contradicted'] / n:.0%} unsupported {c['unsupported'] / n:.0%} | bullets {tot('bullets')} headers "
          f"{tot('headers')} | cited sentences {tot('cited_sentences') / max(1, tot('sentences')):.0%}, "
          f"bad citations {tot('bad_citations')} | mean chars {tot('chars') / max(1, len(recs)):.0f}")


if __name__ == "__main__":
    main()
