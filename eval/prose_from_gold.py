"""Diagnostic: the student writes only the prose, from the QA'd gold notes instead of its own.

If the prose from gold notes is still unfaithful, the errors come from the reduce step; if it is
close to the gold's, they come from the student's notes (the map step). The output directory holds
gold notes and summary plus the student's prose, so eval/judge_prose.py and eval/score_v2.py read it
unchanged.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.run_student_vllm import BASE, merged_model_dir  # noqa: E402
from distill.write_prose import EVIDENCE, prompt_for  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--notes-dir", default=None, help="take notes from this run dir instead of the gold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gold", default="runs/v2/w4000")
    ap.add_argument("--split", default="data/split_v2.json")
    ap.add_argument("--evidence", action="store_true")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    from distill.grpo_vllm import gemma4_overrides

    sessions = json.load(open(args.split))["heldout"]
    gemma = "gemma" in args.base.lower()
    llm = LLM(model=merged_model_dir(args.base, args.adapter), max_model_len=16384, gpu_memory_utilization=0.6,
              hf_overrides=gemma4_overrides(args.base),
              **({"limit_mm_per_prompt": {"image": 0, "audio": 0}} if gemma else {}))
    src = args.notes_dir or args.gold
    golds = {s: json.load(open(os.path.join(src, s + ".json"), encoding="utf-8")) for s in sessions
             if os.path.exists(os.path.join(src, s + ".json"))}
    sessions = [s for s in sessions if s in golds]
    prompts = [[{"role": "user", "content": prompt_for(golds[s], args.evidence)}] for s in sessions]
    outs = llm.chat(prompts, SamplingParams(max_tokens=1500, temperature=0),
                    chat_template_kwargs={"enable_thinking": False})
    os.makedirs(args.out, exist_ok=True)
    for s, o in zip(sessions, outs):
        g = golds[s]
        run = {"notes": g["notes"], "summary": g.get("summary", []), "windows": g["windows"],
               "windows_without_notes": [], "prose": EVIDENCE.sub("", o.outputs[0].text).strip()}
        json.dump(run, open(os.path.join(args.out, s + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(sessions)} proses -> {args.out}", flush=True)
    import psutil
    for child in psutil.Process().children(recursive=True):
        child.kill()
    os._exit(0)


if __name__ == "__main__":
    main()
