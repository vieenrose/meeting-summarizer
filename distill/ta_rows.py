"""Teacher-assistant rows: map rows whose targets are an intermediate teacher's notes.

Small students learn poorly from much stronger teachers (the learnability / capacity gap); a
teacher closer in size can transfer better. Here Gemma-4-E4B, fine-tuned on the gold map rows,
writes notes for every training window (eval/run_student_vllm.py over the training split), and
those window replies become the 2B student's targets. Window prompts are the run's own, which use
the same windowing as the gold rows.
"""
import glob
import json
import sys

out = sys.argv[-1]
n = 0
with open(out, "w", encoding="utf-8") as f:
    for d in sys.argv[1:-1]:
        for p in glob.glob(f"{d}/*.json"):
            run = json.load(open(p, encoding="utf-8"))
            sid = p.split("/")[-1][:-5]
            for e in run["log"]:
                if e.get("window") and not e.get("retry") and e.get("reply", "").strip().startswith("-"):
                    f.write(json.dumps({"kind": "notes", "session": sid, "window": e["window"], "teacher": "e4b-ta",
                                        "messages": e["messages"][:2] + [{"role": "assistant", "content": e["reply"].strip()}]},
                                       ensure_ascii=False) + "\n")
                    n += 1
print(n, "rows ->", out)
