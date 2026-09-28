"""One table for the teacher choice: validity, speed and cost side by side.

Validity is the deciding column -- teacher output that fails the gates cannot be used for training
whatever the model costs -- but speed and cost decide between models that pass at similar rates.
"""
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.validate_teacher import check  # noqa: E402
from summarizer.ingest import parse_line  # noqa: E402

CORPUS_PROMPT_M, CORPUS_OUTPUT_M = 1.084, 0.119   # measured over the 39-session corpus


def rate(run_dir: str) -> dict:
    files = [p for p in glob.glob(f"{run_dir}/*.json") if not p.endswith(".score.json")]
    if not files:
        return {}
    ok, secs, calls, toks = 0, [], [], []
    for p in files:
        run = json.load(open(p, encoding="utf-8"))
        stem = os.path.basename(p)[:-5]
        tpath = f"data/transcripts/{stem}.txt"
        if not os.path.exists(tpath):
            continue
        text = open(tpath, encoding="utf-8").read()
        lines = [parse_line(l) for l in text.splitlines() if l.strip()]
        if not check(run, lines, 0, 650, 0):
            ok += 1
        if run.get("elapsed_s"):
            secs.append(run["elapsed_s"])
            toks.append(len(text) * 0.62 / run["elapsed_s"])
        calls.append(len([l for l in run["log"] if l.get("reply") is not None]))
    return {"n": len(files), "valid": ok,
            "secs": st.median(secs) if secs else None,
            "toks": st.median(toks) if toks else None,
            "calls": st.median(calls) if calls else None}


def main() -> None:
    prices = {r["model"]: r for r in json.load(open("reports/zen_cheap_probe.json", encoding="utf-8"))}
    print(f"{'model':<28}{'notes valid':>12}{'1-pass valid':>13}{'s/sess':>8}{'calls':>7}"
          f"{'corpus$':>9}")
    rows = []
    for model_dir in sorted(glob.glob("runs/bakeoff/*")):
        model = os.path.basename(model_dir)
        notes, single = rate(f"{model_dir}/w4000"), rate(f"{model_dir}/singlepass")
        if not notes and not single:
            continue
        p = prices.get(model, {})
        cost = CORPUS_PROMPT_M * (p.get("in") or 0) + CORPUS_OUTPUT_M * (p.get("out") or 0)
        rows.append((notes.get("valid", 0) / max(1, notes.get("n", 1)), model, notes, single, cost))
    for share, model, notes, single, cost in sorted(rows, reverse=True):
        nv = f"{notes.get('valid', 0)}/{notes.get('n', 0)}" if notes else "-"
        sv = f"{single.get('valid', 0)}/{single.get('n', 0)}" if single else "-"
        secs = notes.get("secs") or single.get("secs")
        calls = notes.get("calls") or single.get("calls")
        print(f"{model:<28}{nv:>12}{sv:>13}"
              f"{(f'{secs:.0f}' if secs else '-'):>8}{(f'{calls:.0f}' if calls else '-'):>7}"
              f"{cost:>9.2f}")
    print("\nBaseline, local Qwen3.6-35B-A3B on 39 sessions: notes 69% valid, single-pass 37%.")


if __name__ == "__main__":
    main()
