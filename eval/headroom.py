"""How much room does RL have on a student? For N reading windows (the GRPO prompt pool, training
sessions), draw G samples from the student and judge every note of every sample against its window
(distill/rl_rewards.py's READ_JUDGE). GRPO can only move the policy toward samples it already
draws: if even the best of G is unfaithful, RL has little to work with.

Reports, over windows: contradicted / unsupported share of notes in all samples, in the greedy
sample, and in the best sample of each group (fewest contradicted, then most notes); and the
recall of the teacher's notes, so "best" cannot mean "wrote nothing".
"""
import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))

from distill.rl_rewards import READ_JUDGE, _secs, ask, parse_json  # noqa: E402
from eval.realtime_agent import ACT, NOTE, similar  # noqa: E402


def sample(url, msgs, temp, seed):
    r = requests.post(url + "/chat/completions", timeout=900, json={
        "model": "rt", "messages": msgs, "max_tokens": 400, "temperature": temp, "top_p": 0.95, "seed": seed,
        "stop": ["\nNEXT"], "chat_template_kwargs": {"enable_thinking": False}}).json()
    text = r["choices"][0]["message"]["content"] or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def notes_of(text, cap=6):
    out = []
    for l in text.split("\n"):
        m = ACT.match(l)
        if m and m.group(1) == "NOTE":
            n = NOTE.match(m.group(2).strip())
            if n and len(out) < cap:
                out.append({"ts": n.group(1), "tag": (n.group(2) or "").upper(), "text": n.group(3)})
    return out


def judge(url, row, notes):
    if not notes:
        return []
    listing = "\n".join(f"{i}. [{n['ts']}] ({n['tag'] or '-'}) {n['text']}" for i, n in enumerate(notes, 1))
    v = parse_json(ask(url, READ_JUDGE.format(window=row["window_text"], notes=listing)), "[") or []
    by = {x.get("id"): x.get("faith") for x in v if isinstance(x, dict)}
    return [by.get(i, "unsupported") for i in range(1, len(notes) + 1)]


def recall(row, notes):
    ref = row["ref"]
    if not ref:
        return None
    return sum(any(abs(_secs(r["ts"]) - _secs(n["ts"])) <= 60 and similar(r["text"], n["text"]) >= 0.25 for n in notes)
               for r in ref) / len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="http://127.0.0.1:8139/v1")
    ap.add_argument("--judge", default="http://127.0.0.1:8700/v1")
    ap.add_argument("--prompts", default="data/train/rl_prompts.jsonl")
    ap.add_argument("--windows", type=int, default=50)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8")]
    rows = [r for r in rows if r["task"] == "read" and r["ref"]]
    rows = random.Random(0).sample(rows, a.windows)
    pool = ThreadPoolExecutor(8)

    def one(row):
        texts = [sample(a.student, row["messages"], 0.0, 0)] + \
                [sample(a.student, row["messages"], a.temp, s) for s in range(1, a.group + 1)]
        res = []
        for t in texts:
            ns = notes_of(t)
            res.append({"notes": ns, "faith": judge(a.judge, row, ns), "recall": recall(row, ns)})
        return res

    groups = list(pool.map(one, rows))

    def share(samples, key):
        tot = sum(len(s["faith"]) for s in samples)
        return sum(f == key for s in samples for f in s["faith"]) / max(1, tot)

    def best(g):   # fewest contradicted notes, then most notes
        return min(g[1:], key=lambda s: (sum(f == "contradicted" for f in s["faith"]), -len(s["notes"])))

    greedy = [g[0] for g in groups]
    sampled = [s for g in groups for s in g[1:]]
    bests = [best(g) for g in groups]
    mean = lambda xs: round(sum(xs) / max(1, len(xs)), 3)  # noqa: E731
    rep = {"tag": a.tag, "windows": len(groups), "group": a.group}
    for name, ss in (("greedy", greedy), ("sampled", sampled), ("best_of_group", bests)):
        rep[name] = {"contradicted": round(share(ss, "contradicted"), 3), "unsupported": round(share(ss, "unsupported"), 3),
                     "notes_per_window": mean([len(s["notes"]) for s in ss]),
                     "recall": mean([s["recall"] for s in ss if s["recall"] is not None])}
    json.dump({"report": rep, "groups": groups}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
