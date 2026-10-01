"""Check the type of every DECISION and ACTION note in teacher traces, and relabel the wrong ones.

The v5 teacher still types as DECISION the reading of the previous minutes, facts and existing
plans, and as ACTION the reading of rules or reports of work under way. A student distilled from
those replies learns the same confusion. Here the judge (eval/section_precision.py's questions)
checks each such note against the transcript around its time:
  DECISION: decided -> keep; proposed -> PROPOSAL; unsupported -> -
  ACTION:   assigned -> keep; idea -> PROPOSAL;     unsupported -> -
The note's line in the teacher's reply and the journal entry are rewritten together, so the
replayed conversation (distill/build_agent_sft.py) stays consistent. Output: a new teacher dir.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.judge_prose_tx import excerpt  # noqa: E402
from eval.section_precision import ASK, PROMPT  # noqa: E402
from summarizer.ingest import parse_line  # noqa: E402
from summarizer.pipeline import ChatClient  # noqa: E402

SEC = {"DECISION": "決議事項", "ACTION": "待辦與負責人"}
NEW = {"decided": None, "assigned": None, "proposed": "PROPOSAL", "idea": "PROPOSAL", "unsupported": "-"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/teacher/rt-q38-v5")
    ap.add_argument("--out", default="runs/teacher/rt-q38-v7")
    ap.add_argument("--transcripts", default="data/v2/transcripts,data/alimeeting/transcripts")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8700/v1")
    ap.add_argument("--parallel", type=int, default=16)
    a = ap.parse_args()
    chat = ChatClient(a.judge_url, "judge", max_tokens=300)
    os.makedirs(a.out, exist_ok=True)
    recs, jobs = {}, []
    for f in sorted(glob.glob(f"{a.teacher}/*.json")):
        sid = os.path.basename(f)[:-5]
        path = next(os.path.join(d, sid + ".txt") for d in a.transcripts.split(",") if os.path.exists(os.path.join(d, sid + ".txt")))
        lines = [parse_line(l) for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]
        rec = json.load(open(f, encoding="utf-8"))
        recs[sid] = rec
        for n in rec["notes"]:
            tag = (n.get("tag") or "").upper()
            if tag in SEC:
                ex = excerpt(f"{n['text']} [{n['ts']}]", lines)
                if ex:
                    jobs.append((sid, n["id"], tag, n["text"], n["ts"], ex))

    def one(job):
        sid, nid, tag, text, ts, ex = job
        ask, choices = ASK[SEC[tag]]
        for _ in range(3):
            reply = chat([{"role": "user", "content": PROMPT.format(excerpt=ex, item=f"{text} [{ts}]", ask=ask, choices="|".join(choices))}])
            found = re.findall(r"\{[^{}]*\"verdict\"[^{}]*\}", reply)
            try:
                v = json.loads(found[-1]).get("verdict")
                if v in choices:
                    return sid, nid, tag, v
            except (IndexError, json.JSONDecodeError):
                pass
        return sid, nid, tag, None

    verdicts = list(ThreadPoolExecutor(a.parallel).map(one, jobs))
    stats = collections.Counter()
    change = collections.defaultdict(dict)
    for sid, nid, tag, v in verdicts:
        stats[(tag, v)] += 1
        if v and NEW[v]:
            change[sid][nid] = NEW[v]
    for sid, rec in recs.items():
        by_id = {n["id"]: n for n in rec["notes"]}
        for nid, new in change[sid].items():
            n = by_id[nid]
            old = n["tag"]
            n["tag"] = new
            # rewrite the same note's line in the reply of its window
            for t in rec["trace"]:
                if t.get("window") == n["window"] and "reply" in t:
                    pat = re.compile(r"^(\s*NOTE\s*\[?" + re.escape(n["ts"]) + r"\]?\s*)\((?:" + re.escape(old) + r")\)(\s*" + re.escape(n["text"][:12]) + ")", re.M)
                    t["reply"], k = pat.subn(lambda m: f"{m.group(1)}({new}){m.group(2)}", t["reply"], count=1)
                    stats["reply_rewritten" if k else "reply_not_found"] += 1
        json.dump(rec, open(os.path.join(a.out, sid + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
    print({f"{k[0]}/{k[1]}" if isinstance(k, tuple) else k: v for k, v in stats.items()})


if __name__ == "__main__":
    main()
