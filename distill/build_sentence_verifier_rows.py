"""Sentence-verifier rows for the reduce step, from the student's own judged prose.

Best-of-4 prose on the held-out set: a random sample is 38% contradicted, greedy 30%, and picking
by the transcript judge 21%, so a good selector is worth 9 points; the deterministic guard picks at
31%. The labels to learn one exist: the student's prose on the training sessions, every sentence
judged against the transcript (eval/judge_prose_tx.py). The verifier sees what the phone has --
the sentence and the notes it cites -- and answers 一致 / 矛盾 / 無依據.

Classes are balanced by downsampling 一致 to twice the 矛盾 count, so the rare class is learned.
"""
import argparse
import collections
import json
import os
import random
import re

CITE = re.compile(r"[\[［]([\d:,\s]+)[\]］]")
LABEL = {"supported": "一致", "contradicted": "矛盾", "unsupported": "無依據"}

PROMPT = """以下是會議筆記中的幾則，以及一句根據筆記寫成的會議紀要。請判斷這句紀要是否忠於這些筆記：
一致：每個事實都有筆記依據；矛盾：至少一處與筆記不符（數字、對象、誰主張、建議或決議、通過或保留）；無依據：沒有矛盾，但有筆記中找不到的具體事實。

## 筆記
{notes}

## 紀要句子
{sentence}

只回答：一致、矛盾或無依據。"""


def cited_notes(sentence, notes):
    by = collections.defaultdict(list)
    for n in notes:
        by[n["ts"]].append(n["text"])
    ts = [t.strip() for g in CITE.findall(sentence) for t in g.split(",")]
    return [f"[{t}] {x}" for t in dict.fromkeys(ts) for x in by.get(t, [])][:4]


def rows(prefix, n, tags):
    out = []
    for i in range(n):
        for tag in tags:
            p = f"reports/judge_prose_tx_{tag}{os.path.basename(prefix)}{i}.json"
            if not os.path.exists(p):
                continue
            for r in json.load(open(p, encoding="utf-8"))["rows"]:
                if r["verdict"] not in LABEL:
                    continue
                run = json.load(open(f"{prefix}{i}/{r['id']}.json", encoding="utf-8"))
                cn = cited_notes(r["sentence"], run["notes"])
                if not cn:
                    continue
                out.append({"kind": "verify_sentence", "session": r["id"], "window": None, "teacher": "judge-tx",
                            "messages": [{"role": "user", "content": PROMPT.format(notes="\n".join(cn), sentence=r["sentence"])},
                                         {"role": "assistant", "content": LABEL[r["verdict"]]}]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="runs/student/rft-prose-s")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="data/train/sentence_verifier_rows.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data = rows(args.prefix, args.n, ("ivod_", "ali_"))
    by = collections.defaultdict(list)
    for d in data:
        by[d["messages"][1]["content"]].append(d)
    rng = random.Random(args.seed)
    keep = by["矛盾"] + by["無依據"] + rng.sample(by["一致"], min(len(by["一致"]), 2 * len(by["矛盾"])))
    rng.shuffle(keep)
    with open(args.out, "w", encoding="utf-8") as f:
        for d in keep:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print({k: len(v) for k, v in by.items()}, "->", len(keep), "rows", args.out)


if __name__ == "__main__":
    main()
