"""Live run of the reading agent on the phone: replay a meeting at real speed, feed each ASR
segment into the model as it is "spoken" (incremental prefill), and measure the real lag.

The phone runs llama-server (upstream llama.cpp, CPU). This driver keeps the whole conversation as
token IDs itself -- the chat template rendered once around markers (/apply-template) -- so every
request extends the cached sequence exactly:

  segment arrives        -> /completion {prompt: tokens + segment, n_predict: 0}   (prefill only)
  window closes          -> + end-of-user/model-header, generate (stop "\\nNEXT", cap 400)
  context near 32k       -> restart from the compacted journal (decisions, open issues, actions
                            first, then recent notes), prefilled at once

Protocol = the deployment harness (realtime_agent.py --harness v3 --no-check --overview none
--number-section --read-max-tokens 400 --restart-journal-tokens 2500). Lag = wall time from a
window's last line being spoken to its notes being parsed. With --speed > 1 the meeting is
replayed faster than real time (a stress test, not a measurement of the live case).
"""
import argparse
import json
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.realtime_agent import ACT, MAX_LOOKBACKS, NOTE, SYSTEM_V3, render, similar, windows_of  # noqa: E402,F401
from summarizer.ingest import parse_line, resolve_citation  # noqa: E402

CTX, READ_MAX, RESTART_BUDGET, MAX_NOTES = 32768, 400, 2500, 6
MARKS = ["⁣S⁣", "⁣U1⁣", "⁣A1⁣", "⁣U2⁣"]


class Phone:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.log = []

    def post(self, path, body):
        r = requests.post(self.url + path, json=body, timeout=3600)
        r.raise_for_status()
        return r.json()

    def tok(self, text, special=False):
        return self.post("/tokenize", {"content": text, "add_special": False, "parse_special": special})["tokens"]

    def prefill(self, tokens):
        t = time.time()
        r = self.post("/completion", {"prompt": tokens, "n_predict": 0, "cache_prompt": True, "id_slot": 0})
        tm = r["timings"]
        self.log.append({"kind": "prefill", "n": tm["prompt_n"], "ms": tm["prompt_ms"], "wall": time.time() - t})
        return tm

    def generate(self, tokens):
        t = time.time()
        r = self.post("/completion", {"prompt": tokens, "n_predict": READ_MAX, "cache_prompt": True, "id_slot": 0,
                                      "temperature": 0.2, "stop": ["\nNEXT"]})
        tm = r["timings"]
        self.log.append({"kind": "generate", "pp": tm["prompt_n"], "pp_ms": tm["prompt_ms"], "tg": tm["predicted_n"],
                         "tg_ms": tm["predicted_ms"], "wall": time.time() - t})
        return r["content"], r.get("stop_type") == "word" or r.get("stopping_word") == "\nNEXT"


def template_parts(phone):
    """Split the chat template into the strings around system, user and assistant contents."""
    msgs = [{"role": "system", "content": MARKS[0]}, {"role": "user", "content": MARKS[1]},
            {"role": "assistant", "content": MARKS[2]}, {"role": "user", "content": MARKS[3]}]
    s = phone.post("/apply-template", {"messages": msgs})["prompt"]
    p0, rest = s.split(MARKS[0])
    p1, rest = rest.split(MARKS[1])
    p2, rest = rest.split(MARKS[2])
    p3, p4 = rest.split(MARKS[3])
    assert p4 == p2, "the generation prompt differs from the in-history model header"
    return p0, p1, p2, p3        # before system, system->user, user->model, model->next user


def compact(journal, count):
    key = {"DECISION": 0, "OPEN-ISSUE": 1, "ACTION": 2}
    order = sorted(range(len(journal)), key=lambda i: (key.get((journal[i]["tag"] or "").upper(), 3), -i))
    chosen, used = set(), 0
    for i in order:
        t = count(render(journal[i]))
        if used + t <= RESTART_BUDGET:
            chosen.add(i)
            used += t
    rest = len(journal) - len(chosen)
    text = "\n".join(render(journal[i]) for i in sorted(chosen)) + (f"\n（另有 {rest} 則較早的筆記未列出）" if rest else "")
    return text or "（尚無筆記）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8200")
    ap.add_argument("--session", required=True)
    ap.add_argument("--transcripts", default="data/v2/transcripts")
    ap.add_argument("--speed", type=float, default=1.0, help="meeting seconds per wall second")
    ap.add_argument("--segment-s", type=float, default=20, help="ASR segment length (meeting seconds)")
    ap.add_argument("--max-windows", type=int, default=0, help="stop after N windows (smoke test)")
    ap.add_argument("--ctx", type=int, default=CTX,
                    help="restart budget: on the phone CPU attention cost grows with depth (prefill 35 tok/s "
                         "at 0, 12 at 4k, 8 at 8k, 4.6 at 16k), so keep the conversation short")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    wtok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B-FP8")   # window sizing, as in the agent
    count = lambda t: len(wtok.encode(t, add_special_tokens=False))  # noqa: E731
    lines = [parse_line(l) for l in open(os.path.join(a.transcripts, a.session + ".txt"), encoding="utf-8")
             .read().splitlines() if l.strip()]
    wins = windows_of(lines, count)
    if a.max_windows:
        wins = wins[:a.max_windows]
    phone = Phone(a.url)
    p0, p1, p2, p3 = template_parts(phone)
    T = {k: phone.tok(v, special=True) for k, v in {"p0": p0, "p1": p1, "p2": p2, "p3": p3}.items()}
    journal, events, restarts = [], [], 0

    def fresh(journal_text):
        return (T["p0"] + phone.tok(SYSTEM_V3) + T["p1"] + phone.tok("## 筆記本（至今）\n" + journal_text)
                + T["p2"] + phone.tok("NEXT") + T["p3"])

    tokens = fresh("（尚無筆記）")
    phone.prefill(tokens)
    t0, m0 = time.time(), wins[0][0].start_s

    def meeting_now():
        return m0 + (time.time() - t0) * a.speed

    def wait_until(meeting_t):
        d = (meeting_t - meeting_now()) / a.speed
        if d > 0:
            time.sleep(d)

    for k, win in enumerate(wins, 1):
        if len(tokens) + 2 * 2000 + READ_MAX + 600 > a.ctx:
            tokens = fresh(compact([e for e in journal], count))
            tm = phone.prefill(tokens)
            restarts += 1
            events.append({"window": k, "restart_prefill": tm["prompt_n"], "ms": tm["prompt_ms"]})
        tokens = tokens + phone.tok(f"## 逐字稿片段 {k}\n")
        seg, seg_end = [], win[0].start_s + a.segment_s
        for j, l in enumerate(win):
            wait_until(l.start_s)                      # the line has just been spoken
            seg.append(l.render())
            last = j == len(win) - 1
            if l.start_s >= seg_end or last:
                tokens = tokens + phone.tok("\n".join(seg) + ("" if last else "\n"))
                if not last:
                    phone.prefill(tokens)               # incremental: prefill while people talk
                seg, seg_end = [], l.start_s + a.segment_s
        close_wall = time.time()
        close_m = win[-1].start_s
        content, stopped = phone.generate(tokens + T["p2"])
        reply = content + ("\nNEXT" if stopped else "")
        n_notes = 0
        for line in reply.splitlines():
            m = ACT.match(line)
            if not m or m.group(1) != "NOTE":
                continue
            n = NOTE.match(m.group(2).strip())
            if not n or resolve_citation(n.group(1), lines) is None or n_notes >= MAX_NOTES:
                continue
            if any(similar(n.group(3), e["text"]) > 0.6 for e in journal[-30:]):
                continue
            journal.append({"id": len(journal) + 1, "window": k, "ts": n.group(1), "tag": n.group(2), "text": n.group(3)})
            n_notes += 1
        tokens = tokens + T["p2"] + phone.tok(reply) + T["p3"]
        lag = time.time() - close_wall
        events.append({"window": k, "meeting_s": close_m, "lag_wall_s": round(lag, 1), "notes": n_notes,
                       "behind_s": round((meeting_now() - close_m) / a.speed, 1), "ctx": len(tokens)})
        print(f"window {k}/{len(wins)} @ {close_m // 60:.0f} min: lag {lag:.0f}s, {n_notes} notes, "
              f"ctx {len(tokens)}", flush=True)
    sections = {"決議事項": ["DECISION"], "待辦與負責人": ["ACTION"], "保留與未決": ["OPEN-ISSUE"], "重要數字": ["NUMBER"]}
    out = []
    for title, tags in sections.items():
        items = [e for e in journal if (e["tag"] or "").upper() in tags]
        out += [f"【{title}】"] + ([f"- {e['text'].rstrip('。')} [{e['ts']}]" for e in items] or ["- 無"])
    items = [l[2:].strip() for l in out if l.startswith("- ") and l[2:].strip() != "無"]
    rec = {"session": a.session, "speed": a.speed, "ctx": a.ctx, "segment_s": a.segment_s, "notes": journal, "minutes": "\n".join(out),
           "prose": "".join(x if x.endswith("。") else x + "。" for x in items), "events": events,
           "calls": phone.log, "restarts": restarts, "wall_s": round(time.time() - t0)}
    os.makedirs(a.out, exist_ok=True)
    json.dump(rec, open(os.path.join(a.out, a.session + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
    lags = [e["lag_wall_s"] for e in events if "lag_wall_s" in e]
    print(f"done: {len(journal)} notes, restarts {restarts}, lag median {sorted(lags)[len(lags) // 2]:.0f}s "
          f"max {max(lags):.0f}s, wall {rec['wall_s']}s")


if __name__ == "__main__":
    main()
