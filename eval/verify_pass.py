"""Run the trained VERIFY pass (distill/build_verify_rows.py) over a student run's notes, window by
window, and write a copy of the run with the fixes applied. Reduce then runs on the result with
eval/prose_from_gold.py --notes-dir.

A FIX line replaces its note only when it parses as a note with a real timestamp from the same
window; anything else leaves the note as it was, so a garbled verifier can only fail to help.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.build_verify_rows import PROMPT  # noqa: E402
from eval.run_student_vllm import merged_model_dir  # noqa: E402
from summarizer.pipeline import parse_turn  # noqa: E402

LINE = re.compile(r"^\s*(\d+)\.\s*(OK|FIX[:：]\s*(.+))\s*$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--base", default="openbmb/MiniCPM5-2B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    runs, jobs = {}, []
    for p in sorted(glob.glob(os.path.join(args.run_dir, "*.json"))):
        sid = os.path.basename(p)[:-5]
        run = json.load(open(p, encoding="utf-8"))
        runs[sid] = run
        wins = {e["window"]: e["messages"][-1]["content"].split("TRANSCRIPT\n", 1)[-1]
                for e in run["log"] if e.get("window") and not e.get("retry")}
        for k, w in wins.items():
            idx = [i for i, n in enumerate(run["notes"]) if n["window"] == k]
            if not idx:
                continue
            body = "\n".join(f"{j + 1}. " + (f"({run['notes'][i]['tag']}) " if run["notes"][i].get("tag") else "")
                             + f"[{run['notes'][i]['ts']}] {run['notes'][i]['text']}" for j, i in enumerate(idx))
            jobs.append((sid, k, idx, w, [{"role": "user", "content": PROMPT.format(window=w, notes=body)}]))
    llm = LLM(model=merged_model_dir(args.base, args.adapter), max_model_len=12288, gpu_memory_utilization=0.8)
    outs = llm.chat([j[-1] for j in jobs], SamplingParams(max_tokens=1500, temperature=0),
                    chat_template_kwargs={"enable_thinking": False})
    fixed = total = 0
    for (sid, k, idx, w, _), o in zip(jobs, outs):
        run = runs[sid]
        stamps = set(re.findall(r"^\[([\d:]+)\]", w, re.M))
        for line in o.outputs[0].text.splitlines():
            m = LINE.match(line)
            if not m or not m.group(3):
                continue
            j = int(m.group(1)) - 1
            if not 0 <= j < len(idx):
                continue
            t = parse_turn("- " + m.group(3)).notes
            if t and t[0][1] in stamps:
                n = run["notes"][idx[j]]
                n.setdefault("verify_old", n["text"])
                n["tag"], n["ts"], n["text"] = t[0][0], t[0][1], t[0][2]
                fixed += 1
        total += len(idx)
    os.makedirs(args.out, exist_ok=True)
    for sid, run in runs.items():
        json.dump(run, open(os.path.join(args.out, sid + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"verify: fixed {fixed}/{total} notes ({fixed / max(1, total):.1%}) -> {args.out}", flush=True)
    import psutil
    for c in psutil.Process().children(recursive=True):
        c.kill()
    os._exit(0)


if __name__ == "__main__":
    main()
