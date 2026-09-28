"""N sampled prose abstracts per session from a run's own notes; writes one run dir per sample so
eval/judge_prose_tx.py can score each unchanged."""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.write_prose import EVIDENCE, prompt_for  # noqa: E402
from eval.run_student_vllm import merged_model_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--base", default="openbmb/MiniCPM5-2B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    runs = {}
    for d in args.run_dirs:
        for p in glob.glob(os.path.join(d, "*.json")):
            runs[os.path.basename(p)[:-5]] = json.load(open(p, encoding="utf-8"))
    sids = sorted(runs)
    llm = LLM(model=merged_model_dir(args.base, args.adapter), max_model_len=16384, gpu_memory_utilization=0.8)
    outs = llm.chat([[{"role": "user", "content": prompt_for(runs[s])}] for s in sids],
                    SamplingParams(n=args.n, temperature=0.8, top_p=0.95, max_tokens=1500),
                    chat_template_kwargs={"enable_thinking": False})
    for i in range(args.n):
        os.makedirs(f"{args.out_prefix}{i}", exist_ok=True)
    for s, o in zip(sids, outs):
        for i, c in enumerate(o.outputs):
            json.dump({"notes": runs[s]["notes"], "prose": EVIDENCE.sub("", c.text).strip()},
                      open(f"{args.out_prefix}{i}/{s}.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(sids)} sessions x {args.n} proses -> {args.out_prefix}*", flush=True)
    import psutil
    for c in psutil.Process().children(recursive=True):
        c.kill()
    os._exit(0)


if __name__ == "__main__":
    main()
