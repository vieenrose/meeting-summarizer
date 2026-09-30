"""SFT data for the realtime reading agent: replay each teacher session (runs/teacher/rt-q38-v3x,
Qwen3.8-27B on the tuned v3x protocol) as the conversation the student will live through.

A session is one growing conversation (eval/realtime_agent.py): [system] [journal so far]
([window k] [actions])*. The teacher's own restarts were decided on its 32k budget; here the
conversation restarts from the journal whenever it would pass --max-tokens, which yields training
sequences that fit one GPU and have exactly the shape of a post-restart state.

Each sequence becomes one row: the rendered conversation, and the character spans of the teacher
turns that carry loss. A turn is kept for loss only if none of its notes was judged contradicted
by the transcript (reports/judge_prose_tx_notes-rt-q38-v3x.json); a rejected turn stays in the
history as context, since the student will see its own imperfect notes there too.
"""
import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.realtime_agent import SYSTEM_V3, SYSTEM_V5, render, windows_of  # noqa: E402
from summarizer.ingest import parse_line  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/teacher/rt-q38-v3x")
    ap.add_argument("--judged", default="reports/judge_prose_tx_notes-rt-q38-v3x.json", help="one or more, comma-separated")
    ap.add_argument("--transcripts", default="data/v2/transcripts,data/alimeeting/transcripts",
                    help="comma-separated dirs; each session is read from the one that has it")
    ap.add_argument("--system", default="v3", choices=["v3", "v5"], help="the protocol the teacher ran")
    ap.add_argument("--base", default="google/gemma-4-E2B-it-qat-q4_0-unquantized")
    ap.add_argument("--max-tokens", type=int, default=14000)
    ap.add_argument("--out", default="data/train/agent_sft_rows.jsonl")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    # The agent sizes windows with the Qwen tokenizer; use the same one so windows match the run.
    wtok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B-FP8")
    wcount = lambda t: len(wtok.encode(t, add_special_tokens=False))  # noqa: E731

    def n_tokens(msgs):
        return len(tok.apply_chat_template(msgs, tokenize=True, return_dict=False, enable_thinking=False))

    system = SYSTEM_V5 if a.system == "v5" else SYSTEM_V3
    bad = collections.defaultdict(set)
    for r in (r for j in a.judged.split(",") for r in json.load(open(j, encoding="utf-8"))["rows"]):
        if r["verdict"] == "contradicted":
            bad[r["id"]].add(r["sentence"])

    stats = collections.Counter()
    with open(a.out, "w", encoding="utf-8") as fo:
        for f in sorted(glob.glob(f"{a.teacher}/ivod_*.json") + glob.glob(f"{a.teacher}/alimeeting_*.json")):
            sid = os.path.basename(f)[:-5]
            rec = json.load(open(f, encoding="utf-8"))
            path = next(os.path.join(d, sid + ".txt") for d in a.transcripts.split(",")
                        if os.path.exists(os.path.join(d, sid + ".txt")))
            lines = [parse_line(l) for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]
            wins = windows_of(lines, wcount)
            replies = {t["window"]: t["reply"] for t in rec["trace"] if "reply" in t and t["window"] != "overview"}
            if len(replies) != len(wins):
                stats["session_window_mismatch"] += 1
                continue
            notes = rec["notes"]
            badwin = {n["window"] for n in notes if (n["text"].rstrip("。") + f" [{n['ts']}]。") in bad[sid]}

            def restart(k):
                journal = [n for n in notes if n["window"] < k]
                return [{"role": "system", "content": system},
                        {"role": "user", "content": "## 筆記本（至今）\n" + ("\n".join(map(render, journal)) or "（尚無筆記）")},
                        {"role": "assistant", "content": "NEXT"}], []

            def flush(msgs, loss_turns):
                if loss_turns:
                    fo.write(json.dumps({"session": sid, "messages": msgs, "loss_turns": loss_turns},
                                        ensure_ascii=False) + "\n")
                    stats["rows"] += 1
                    stats["loss_turns"] += len(loss_turns)

            msgs, loss_turns = restart(1)
            for k, win in enumerate(wins, 1):
                block = "\n".join(l.render() for l in win)
                turn = [{"role": "user", "content": f"## 逐字稿片段 {k}\n{block}"},
                        {"role": "assistant", "content": replies[k]}]
                if n_tokens(msgs + turn) > a.max_tokens and len(msgs) > 3:
                    flush(msgs, loss_turns)
                    msgs, loss_turns = restart(k)
                    stats["restarts"] += 1
                msgs = msgs + turn
                if k in badwin:
                    stats["turns_rejected"] += 1
                else:
                    loss_turns.append(len(msgs) - 1)   # index of the assistant message
                stats["turns"] += 1
            flush(msgs, loss_turns)
            stats["sessions"] += 1
    print(dict(stats))


if __name__ == "__main__":
    main()
