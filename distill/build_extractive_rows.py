"""Extractive map rows, and reduce rows over them.

The 2B map student is contradicted on ~15-17% of its notes against the gold's 6%, whatever the
training (SFT, evidence-first, DPO on judged notes, RFT); its errors are misreadings made while
paraphrasing. The untrained base, which copies the transcript, is contradicted least. So the map
task is changed from paraphrase to selection: for each gold note, the student writes the note's
speaker label and tag, then the transcript span that carries it, verbatim -- 「...」. Abstraction
moves to reduce (Gemma-4-E4B), which reads the spans.

Span choice reuses distill/build_evidence_rows.best_quote with a wider quote and a lower overlap
floor; a gold note with no matching span is dropped from the extractive target (it has no span to
copy), and counted.
"""
import argparse
import json
import re

import distill.build_evidence_rows as ev
from distill.write_prose import prompt_for
from summarizer.pipeline import parse_turn

ACTS = [("決議", ("決議", "決定", "通過", "照案", "照列", "准予")), ("保留", ("保留", "協商", "擇期", "另定")),
        ("提議", ("建議", "主張", "提議", "提案", "要求", "希望")), ("詢問", ("詢問", "質詢", "質疑", "請教")),
        ("答覆", ("回應", "說明", "答覆", "報告", "表示"))]


def act(note):
    """Speech-act label of a gold note, by the first matching keyword class."""
    for label, words in ACTS:
        if any(w in note for w in words):
            return label
    return "陳述"


EXTRACT_RULE = ("本任務為摘錄：每則筆記寫出講者身分（無法確定寫「發言者」）後，"
                "以「…」照抄逐字稿中承載該重點的原文（不超過 80 字），不要改寫。\n\n")
ACT_RULE = "並在原文前以〈〉標出發言性質：決議、保留、提議、詢問、答覆或陳述。\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=["data/train/v2_gold_rows_all.jsonl", "data/train/alimeeting_gold_rows.jsonl"])
    ap.add_argument("--map-out", default="data/train/extract_map_rows.jsonl")
    ap.add_argument("--reduce-out", default="data/train/extract_reduce_rows.jsonl")
    ap.add_argument("--acts", action="store_true", help="annotate each span with its speech act")
    args = ap.parse_args()
    ev.QUOTE, ev.MIN_OVERLAP = 80, 0.2
    kept = total = 0
    extracted = {}                                  # session -> list of (ts, note text) in order
    with open(args.map_out, "w", encoding="utf-8") as fm:
        for path in args.rows:
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                if r["kind"] != "notes":
                    continue
                user = r["messages"][1]["content"]
                lines = [(m.group(1), m.group(2)) for m in map(ev.LINE.match, user.split("TRANSCRIPT\n", 1)[-1].splitlines()) if m]
                out = []
                for tag, ts, t in parse_turn(r["messages"][-1]["content"]).notes:
                    total += 1
                    q = ev.best_quote(t, lines, ts)
                    if not q:
                        continue
                    m = re.match(r"^([^:：]{1,15})[:：]", t)
                    who = m.group(1).strip() if m else "發言者"
                    text = f"{who}: 〈{act(t)}〉「{q}」" if args.acts else f"{who}: 「{q}」"
                    out.append((f"- ({tag}) " if tag else "- ") + f"[{ts}] {text}")
                    extracted.setdefault(r["session"], []).append({"ts": ts, "text": text})
                    kept += 1
                if not out:
                    continue
                msgs = [dict(m) for m in r["messages"]]
                msgs[1]["content"] = user.replace("TRANSCRIPT\n", EXTRACT_RULE + (ACT_RULE if args.acts else "") + "TRANSCRIPT\n", 1)
                msgs[-1]["content"] = "\n".join(out)
                fm.write(json.dumps({**r, "messages": msgs}, ensure_ascii=False) + "\n")
    n = 0
    with open(args.reduce_out, "w", encoding="utf-8") as fr:
        for path in args.rows:
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                if r["kind"] != "prose" or r["session"] not in extracted:
                    continue
                run = {"notes": extracted[r["session"]]}
                fr.write(json.dumps({**r, "messages": [{"role": "user", "content": prompt_for(run)}, r["messages"][-1]]},
                                    ensure_ascii=False) + "\n")
                n += 1
    print(f"map: kept {kept}/{total} gold notes as spans ({kept / max(1, total):.0%}); reduce rows {n}")


if __name__ == "__main__":
    main()
