"""Abstention on uncertainty: drop the notes whose figures or names the model generated with a low
probability, and see what it buys. Needs a run made with realtime_agent.py --logprobs and its
transcript judgment (eval/judge_prose_tx.py).

Per note, from the reply's token log-probs: the lowest probability among its "content" tokens
(digits, and the tokens of bodies and roles), or among all its tokens when it has none. For each
share of notes kept, it reports the contradicted share of the kept minutes statements (the judge's
verdicts are reused: dropping statements does not change the others'), and writes the filtered run
for a coverage judgment (eval/judge_prose.py).
"""
import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.build_contrast_pairs import ACTOR  # noqa: E402
from eval.realtime_agent import as_prose  # noqa: E402

CITE = re.compile(r"\[(\d+:\d{2}(?::\d{2})?)\]")


def note_conf(reply_lp):
    """[(ts, text prefix, confidence)] for the NOTE lines of one reply."""
    text, spans = "", []
    for tok, lp in reply_lp:
        spans.append((len(text), len(text) + len(tok), lp))
        text += tok
    out = []
    for m in re.finditer(r"^\s*NOTE\s*\[?(\d+:\d{2}(?::\d{2})?)\]?\s*(?:\(([\w-]+)\)\s*)?(.+)$", text, re.M):
        a, b = m.start(3), m.end(3)
        body = m.group(3)
        content = set()
        for k in re.finditer(r"\d+", body):
            content.update(range(a + k.start(), a + k.end()))
        for k in ACTOR.finditer(body):
            content.update(range(a + k.start(), a + k.end()))
        lps = [lp for s, e, lp in spans if e > a and s < b and (not content or any(i in content for i in range(s, e)))]
        out.append((m.group(1), body.strip()[:8], min(lps) if lps else 0.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name: runs/student/<run>, reports/judge_prose_tx_<run>.json")
    ap.add_argument("--keep", default="1.0,0.95,0.9,0.85,0.8,0.7")
    ap.add_argument("--write", default="0.9,0.8", help="kept shares to write as filtered runs")
    a = ap.parse_args()
    d = f"runs/student/{a.run}"
    conf = {}
    for f in os.listdir(d):
        if f.endswith(".json"):
            rec = json.load(open(os.path.join(d, f), encoding="utf-8"))
            for t in rec["trace"]:
                for ts, pre, c in note_conf(t.get("lp") or []):
                    conf[(f[:-5], ts, pre)] = c
    rows = [r for r in json.load(open(f"reports/judge_prose_tx_{a.run}.json", encoding="utf-8"))["rows"]
            if r.get("verdict") in ("supported", "unsupported", "contradicted")]
    scored = []
    for r in rows:
        ts = CITE.findall(r["sentence"])
        key = (r["id"], ts[0], CITE.sub("", r["sentence"]).strip()[:8]) if ts else None
        if key in conf:
            scored.append((conf[key], r["verdict"] == "contradicted"))
    scored.sort(key=lambda x: -x[0])
    print(f"{a.run}: {len(scored)}/{len(rows)} statements matched to a note confidence")
    cuts = {}
    for q in map(float, a.keep.split(",")):
        k = max(1, int(len(scored) * q))
        kept = scored[:k]
        cuts[q] = kept[-1][0]
        print(f"  keep {q:.0%} (p >= {math.exp(kept[-1][0]):.3f}): contradicted {sum(b for _, b in kept) / k:.1%}")
    for q in map(float, a.write.split(",")):
        out = f"runs/student/{a.run}-keep{int(q * 100)}"
        os.makedirs(out, exist_ok=True)
        for f in os.listdir(d):
            if not f.endswith(".json"):
                continue
            rec = json.load(open(os.path.join(d, f), encoding="utf-8"))
            drop = {n["id"] for n in rec["notes"] if conf.get((f[:-5], n["ts"], n["text"].strip()[:8]), 0.0) < cuts[q]}
            gone = [(n["ts"], n["text"].strip()[:8]) for n in rec["notes"] if n["id"] in drop]
            rec["notes"] = [n for n in rec["notes"] if n["id"] not in drop]
            # the minutes lose the dropped notes' lines; the judged prose is rebuilt from them
            rec["minutes"] = "\n".join(l for l in rec["minutes"].split("\n")
                                       if not any(f"[{ts}]" in l and pre in l for ts, pre in gone))
            rec["prose"] = as_prose(rec["minutes"])
            json.dump(rec, open(os.path.join(out, f), "w", encoding="utf-8"), ensure_ascii=False)
        print("  wrote", out)


if __name__ == "__main__":
    main()
