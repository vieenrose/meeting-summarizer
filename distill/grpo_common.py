"""Data and reward shared by the GRPO trainers.

Kept apart from the trainer scripts because each trainer patches TRL for its own environment on
import -- grpo_synth.py hides vLLM from TRL 0.24 -- and those patches must not leak into the other.
"""
import json
import os

from distill.rewards import score_summary
from summarizer.ingest import parse_line


def build(rows_path, split_path, gold_dir, transcripts):
    heldout = set(json.load(open(split_path))["heldout"])
    out = []
    for line in open(rows_path, encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] != "synthesis" or r["session"] in heldout:
            continue
        out.append({"prompt": r["messages"][:-1], "session": r["session"]})
    return out


def make_reward(gold_dir, transcripts):
    cache = {}

    def load(sid):
        if sid not in cache:
            lines = [parse_line(l) for l in open(os.path.join(transcripts, sid + ".txt"),
                                                  encoding="utf-8").read().splitlines() if l.strip()]
            gold = json.load(open(os.path.join(gold_dir, sid + ".json"), encoding="utf-8"))["summary"]
            cache[sid] = (lines, gold)
        return cache[sid]

    def summary_reward(completions, session, **_):
        scores = []
        for comp, sid in zip(completions, session):
            text = comp[0]["content"] if isinstance(comp, list) else comp
            lines, gold = load(sid)
            value, detail = score_summary(text, lines, gold, detail=True)
            scores.append(float(value))
            if os.environ.get("REWARD_DEBUG"):
                print("REWARD_DEBUG", sid, round(value, 3), detail.get("reason"), repr(text[:160]), flush=True)
        return scores

    return summary_reward


def build_notes(rows_path, split_path):
    """One prompt per training-session window, with the teacher's notes for that window."""
    heldout = set(json.load(open(split_path))["heldout"])
    out = []
    for line in open(rows_path, encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] != "notes" or r["session"] in heldout:
            continue
        out.append({"prompt": r["messages"][:-1], "gold_notes": r["messages"][-1]["content"],
                    "window_prompt": r["messages"][1]["content"]})
    return out


def make_notes_reward():
    from distill.rewards import score_notes

    def notes_reward(completions, gold_notes, window_prompt, **_):
        scores = []
        for comp, gold, prompt in zip(completions, gold_notes, window_prompt):
            text = comp[0]["content"] if isinstance(comp, list) else comp
            value, detail = score_notes(text, prompt, gold, detail=True)
            scores.append(float(value))
            if os.environ.get("REWARD_DEBUG"):
                print("REWARD_DEBUG notes", round(value, 3), detail.get("reason"), repr(text[:120]), flush=True)
        return scores

    return notes_reward


def make_judged_notes_reward(weight=0.5):
    """Lexical notes reward blended with window-judge faithfulness: the share of a sample's notes
    that the judge finds consistent with the window. A sample the judge cannot read keeps the
    lexical score alone rather than being scored as if it were all right or all wrong."""
    from concurrent.futures import ThreadPoolExecutor
    from distill.rewards import score_notes
    from distill.window_judge import WindowJudge
    from summarizer.pipeline import parse_turn
    judge = WindowJudge()

    def one(args):
        text, gold, prompt = args
        value, detail = score_notes(text, prompt, gold, detail=True)
        if "reason" in detail:                      # no notes, copied the prompt: already zero
            return float(value), None
        notes = [f"[{ts}] {t}" for _, ts, t in parse_turn(text).notes]
        wrong = judge.wrong(prompt.split("TRANSCRIPT\n", 1)[-1], notes)
        if wrong is None:
            return float(value), None
        faith = 1 - len(wrong) / len(notes)
        return (1 - weight) * float(value) + weight * faith, faith

    def judged_notes_reward(completions, gold_notes, window_prompt, **_):
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        with ThreadPoolExecutor(8) as ex:
            out = list(ex.map(one, zip(texts, gold_notes, window_prompt)))
        if os.environ.get("REWARD_DEBUG"):
            print("REWARD_DEBUG judged", [(round(v, 2), f if f is None else round(f, 2)) for v, f in out], flush=True)
        return [v for v, _ in out]

    return judged_notes_reward
