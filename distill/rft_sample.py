"""Rejection-sampling data: N sampled note sets per training window, scored, best kept.

Online GRPO with the window judge ran ~7 min a step, because every step waits on 16 judge calls.
Sampling first and judging in bulk decouples the two, and keeps the policy update a plain SFT.
Score = 0.5 lexical notes reward (recall of the gold's figures and names, citations, structure,
grounding) + 0.5 window-judge faithfulness. The lexical half matters: the judge alone rates the
untrained base best (20% wrong vs gold 25%), because notes that copy the transcript are rarely
"wrong"; the base's gold-fact recall is 0.30.
"""
import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def sample(args):
    from vllm import LLM, SamplingParams
    from eval.run_student_vllm import merged_model_dir
    rows = [json.loads(l) for l in open(args.rows, encoding="utf-8")]
    rows = [r for r in rows if r["kind"] == "notes"]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.windows]
    llm = LLM(model=merged_model_dir(args.base, args.adapter), max_model_len=8192, gpu_memory_utilization=0.8)
    outs = llm.chat([r["messages"][:-1] for r in rows],
                    SamplingParams(n=args.n, temperature=0.8, top_p=0.95, max_tokens=1024),
                    chat_template_kwargs={"enable_thinking": False})
    with open(args.samples, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({**r, "samples": [c.text for c in o.outputs]}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} windows x {args.n} -> {args.samples}", flush=True)


def score(args):
    from distill.rewards import score_notes
    from distill.window_judge import WindowJudge
    from summarizer.pipeline import parse_turn
    judges = [WindowJudge(u) for u in args.judges.split(",")]
    rows = [json.loads(l) for l in open(args.samples, encoding="utf-8")]
    done = {}
    if os.path.exists(args.scored):
        for l in open(args.scored, encoding="utf-8"):
            d = json.loads(l)
            done[(d["session"], d["window"])] = d

    def one(r):
        judge = judges[hash((r["session"], r["window"])) % len(judges)]
        prompt, gold = r["messages"][1]["content"], r["messages"][-1]["content"]
        window = prompt.split("TRANSCRIPT\n", 1)[-1]
        scored = []
        for text in r["samples"]:
            lex, detail = score_notes(text, prompt, gold, detail=True)
            if "reason" in detail:
                scored.append({"text": text, "lex": 0.0, "faith": None, "score": 0.0})
                continue
            notes = [f"[{ts}] {t}" for _, ts, t in parse_turn(text).notes]
            wrong = judge.wrong(window, notes)
            faith = None if wrong is None else 1 - len(wrong) / len(notes)
            scored.append({"text": text, "lex": lex, "faith": faith,
                           "score": None if faith is None else 0.5 * lex + 0.5 * faith})
        return {**{k: r[k] for k in ("kind", "session", "window", "teacher", "messages")}, "scored": scored}

    todo = [r for r in rows if (r["session"], r["window"]) not in done]
    with open(args.scored, "a", encoding="utf-8") as f, ThreadPoolExecutor(4 * len(judges)) as ex:
        for i, d in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"scored {i}/{len(todo)}", flush=True)


def select(args):
    """Best sample per window, kept only if it beats the window's median sample -- a window where
    all samples tie teaches nothing -- and is judged at least 0.8 faithful."""
    kept = []
    for l in open(args.scored, encoding="utf-8"):
        d = json.loads(l)
        ok = sorted((s for s in d["scored"] if s["score"] is not None), key=lambda s: s["score"])
        if len(ok) < 2:
            continue
        best, median = ok[-1], ok[len(ok) // 2 - (len(ok) % 2 == 0)]
        if best["score"] > median["score"] and best["faith"] >= 0.8:
            kept.append({"kind": "notes", "session": d["session"], "window": d["window"],
                         "teacher": "rft-minicpm5", "messages": d["messages"][:-1]
                         + [{"role": "assistant", "content": best["text"]}]})
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)} windows -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sample", "score", "select"])
    ap.add_argument("--base", default="openbmb/MiniCPM5-2B")
    ap.add_argument("--adapter", default="runs/sft/minicpm5-v2-ivod-ali/final")
    ap.add_argument("--rows", default="data/train/train_ivod_ali.jsonl")
    ap.add_argument("--windows", type=int, default=1000)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", default="data/train/rft_samples.jsonl")
    ap.add_argument("--scored", default="data/train/rft_scored.jsonl")
    ap.add_argument("--out", default="data/train/rft_best.jsonl")
    ap.add_argument("--judges", default="http://127.0.0.1:1919/v1,http://127.0.0.1:2919/v1")
    a = ap.parse_args()
    {"sample": sample, "score": score, "select": select}[a.step](a)
