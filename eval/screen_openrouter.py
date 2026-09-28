"""Quick screen of free OpenRouter models as note-taking teachers.

Each candidate writes notes for the same few real transcript windows from the gold corpus, one from
the densest budget session. Measured per model: does it answer, how fast (latency, tokens/s), how
good (the notes reward against the teacher's gold notes for that window -- the scorer GRPO uses),
and Simplified-character leakage, which disqualifies a zh-TW teacher. The survivors go on to the
full four-session benchmark; this stage only spends a few calls per model.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.rewards import score_notes  # noqa: E402
from eval.zen_probe import script_mix  # noqa: E402

KEY = open(os.path.expanduser("~/.secrets/openrouter_key")).read().strip()


def pick_windows(rows_path, sessions):
    rows = [json.loads(l) for l in open(rows_path, encoding="utf-8")]
    out = []
    for sid in sessions:
        cands = [r for r in rows if r["kind"] == "notes" and r["session"] == sid]
        cands.sort(key=lambda r: -len(r["messages"][-1]["content"]))   # the densest window
        out.append(cands[0])
    return out


def call(model, messages, timeout):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": 8000, "temperature": 0}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                                          "X-Title": "meeting-summarizer"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            d = json.load(resp)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code} {e.read()[:120]!r}", "secs": time.time() - t0}
    except Exception as e:                                            # noqa: BLE001
        return {"error": type(e).__name__, "secs": time.time() - t0}
    if "error" in d:
        return {"error": str(d["error"])[:120], "secs": time.time() - t0}
    msg = d["choices"][0]["message"]
    usage = d.get("usage") or {}
    return {"text": msg.get("content") or "", "secs": time.time() - t0,
            "out_tokens": usage.get("completion_tokens") or 0,
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0}


def screen(model, windows, timeout):
    res = []
    for w in windows:
        r = call(model, w["messages"][:-1], timeout)
        if "error" not in r:
            r["score"] = score_notes(r["text"], w["messages"][1]["content"], w["messages"][-1]["content"])
            r["simplified"] = script_mix(r["text"])[0]
        res.append(r)
    ok = [r for r in res if "error" not in r]
    summary = {"model": model, "answered": f"{len(ok)}/{len(res)}"}
    if ok:
        summary.update({
            "score": sum(r["score"] for r in ok) / len(ok),
            "secs": sum(r["secs"] for r in ok) / len(ok),
            "tok_s": sum(r["out_tokens"] for r in ok) / max(1e-9, sum(r["secs"] for r in ok)),
            "reasoning": sum(r["reasoning_tokens"] for r in ok) / len(ok),
            "simplified": sum(r["simplified"] for r in ok),
            "empty": sum(1 for r in ok if not r["text"].strip()),
        })
    summary["errors"] = [r["error"] for r in res if "error" in r][:2]
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--sessions", nargs="+", default=["ivod_17409", "ivod_17652", "ivod_14064"])
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()
    windows = pick_windows("data/train/gold_rows.jsonl", args.sessions)
    with ThreadPoolExecutor(max_workers=len(args.models)) as pool:
        results = list(pool.map(lambda m: screen(m, windows, args.timeout), args.models))
    results.sort(key=lambda r: -(r.get("score") or -1))
    print(f"{'model':52} {'ok':>4} {'score':>6} {'s/call':>7} {'tok/s':>6} {'reason':>7} {'simp':>5} {'empty':>5}")
    for r in results:
        if "score" in r:
            print(f"{r['model']:52} {r['answered']:>4} {r['score']:6.3f} {r['secs']:7.0f} {r['tok_s']:6.1f} "
                  f"{r['reasoning']:7.0f} {r['simplified']:5d} {r['empty']:5d}")
        else:
            print(f"{r['model']:52} {r['answered']:>4}  --  {'; '.join(r['errors'])[:90]}")
    json.dump(results, open("reports/openrouter_screen.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
