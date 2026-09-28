"""Evidence-first notes rows: each gold note preceded by the transcript span that supports it.

The student's contradictions are misreadings -- the wrong speaker or object for a figure, a sense
inverted, a proposal written as a decision -- and the untrained base, which copies the transcript,
is contradicted least. Extract-then-generate and rationale distillation (Distilling step-by-step)
both report large faithfulness gains from making the model state its evidence first. Here the
rationale is free: for each gold note, the transcript span from its anchor to SPAN seconds on that
shares most character bigrams with the note is quoted verbatim, in 〔據「...」〕 before the note.
The pipeline strips the quote before reduce (summarizer.pipeline.EVIDENCE).

A note whose best span shares too little with it (below MIN_OVERLAP) gets no quote rather than a
wrong one; the model then learns that a quote is only written when there is one to copy.
"""
import argparse
import json
import re

from distill.write_prose import PROSE_EVIDENCE_RULE
from summarizer.pipeline import EVIDENCE_RULE, parse_turn

SPAN, QUOTE, MIN_OVERLAP = 120, 60, 0.25
LINE = re.compile(r"^\[(\d+:\d{2}(?::\d{2})?)\]\s*(?:S\d+:\s*)?(.*)$")


def secs(ts):
    p = [int(x) for x in ts.split(":")]
    return p[0] * 3600 + p[1] * 60 + p[2] if len(p) == 3 else p[0] * 60 + p[1]


def bigrams(t):
    t = re.sub(r"[\s，。、：:「」（）()？！,.?!]", "", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def best_quote(note, lines, ts):
    body = re.sub(r"^[^:：]{1,15}[:：]\s*", "", note)
    want = bigrams(body)
    if not want:
        return None
    t0 = secs(ts)
    best, score = None, 0.0
    for lts, text in lines:
        if not 0 <= secs(lts) - t0 <= SPAN:
            continue
        for i in range(0, max(1, len(text) - QUOTE + 1), 5):
            span = text[i:i + QUOTE]
            s = len(bigrams(span) & want) / len(want)
            if s > score:
                best, score = span, s
    return best.strip("，。 ") if best and score >= MIN_OVERLAP else None


CITE = re.compile(r"[\[［]([\d:,\s]+)[\]］]")


def prose_row(r):
    """Prefix each gold prose sentence with the notes its citations point to (at most two, 40
    characters each), in the same 〔據「...」〕 form; the prose's own citations give the mapping."""
    prompt = r["messages"][0]["content"]
    notes = {}
    for m in re.finditer(r"^\[(\d+:\d{2}(?::\d{2})?)\] (.+)$", prompt, re.M):
        notes.setdefault(m.group(1), re.sub(r"^[^:：]{1,15}[:：]\s*", "", m.group(2))[:40])
    out = []
    for para in r["messages"][-1]["content"].split("\n"):
        sents = re.split(r"(?<=[。！？])", para)
        buf = ""
        for sent in sents:
            if not sent.strip():
                continue
            ts = [t.strip() for g in CITE.findall(sent) for t in g.split(",")]
            quotes = [notes[t] for t in dict.fromkeys(ts) if t in notes][:2]
            buf += "".join(f"〔據「{q}」〕" for q in quotes) + (" " if quotes else "") + sent
        out.append(buf)
    msgs = [dict(m) for m in r["messages"]]
    msgs[0]["content"] = prompt.replace("\n只輸出紀要本文。", PROSE_EVIDENCE_RULE + "\n只輸出紀要本文。")
    msgs[-1]["content"] = "\n".join(out)
    return {**r, "messages": msgs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=["data/train/v2_gold_rows_all.jsonl", "data/train/alimeeting_gold_rows.jsonl"])
    ap.add_argument("--out", default="data/train/evidence_rows.jsonl")
    args = ap.parse_args()
    n = quoted = total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for path in args.rows:
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                if r["kind"] == "prose":
                    f.write(json.dumps(prose_row(r), ensure_ascii=False) + "\n")
                    continue
                if r["kind"] != "notes":
                    f.write(line)
                    continue
                user = r["messages"][1]["content"]
                lines = [(m.group(1), m.group(2)) for m in map(LINE.match, user.split("TRANSCRIPT\n", 1)[-1].splitlines()) if m]
                out = []
                for tag, ts, t in parse_turn(r["messages"][-1]["content"]).notes:
                    q = best_quote(t, lines, ts)
                    total += 1
                    quoted += q is not None
                    ev = f"〔據「{q}」〕 " if q else ""
                    out.append((f"- ({tag}) " if tag else "- ") + f"[{ts}] {ev}{t}")
                msgs = [dict(m) for m in r["messages"]]
                msgs[1]["content"] = user.replace("TRANSCRIPT\n", EVIDENCE_RULE + "TRANSCRIPT\n", 1)
                msgs[-1]["content"] = "\n".join(out)
                f.write(json.dumps({**r, "messages": msgs}, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} notes rows, {quoted}/{total} notes quoted ({quoted / max(1, total):.0%}) -> {args.out}")


if __name__ == "__main__":
    main()
