"""N extra temperature samples of the map step for every window of a student run (for the
consistency filter in eval/notes_filter.py). Same prompts the run used, read from its log."""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.run_student_vllm import merged_model_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--base", default="openbmb/MiniCPM5-2B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    jobs = []
    for p in sorted(glob.glob(os.path.join(args.run_dir, "*.json"))):
        run = json.load(open(p, encoding="utf-8"))
        for e in run["log"]:
            if e.get("window") and not e.get("retry"):
                jobs.append((os.path.basename(p)[:-5], e["window"], e["messages"][:2]))
    llm = LLM(model=merged_model_dir(args.base, args.adapter), max_model_len=12288, gpu_memory_utilization=0.8)
    outs = llm.chat([m for _, _, m in jobs], SamplingParams(n=args.n, temperature=0.7, top_p=0.95, max_tokens=1024),
                    chat_template_kwargs={"enable_thinking": False})
    res = {}
    for (sid, k, _), o in zip(jobs, outs):
        res.setdefault(sid, {})[str(k)] = [c.text for c in o.outputs]
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(jobs)} windows x {args.n} -> {args.out}", flush=True)
    import psutil
    for c in psutil.Process().children(recursive=True):
        c.kill()
    os._exit(0)


if __name__ == "__main__":
    main()
