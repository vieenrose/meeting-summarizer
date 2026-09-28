"""Best-of-N prose selection with the trained sentence verifier, scored with existing judge labels.

Candidates per held-out session: the greedy prose (runs/student/v2-minicpm5) and N samples
(runs/student/bo4-s*), every sentence already judged against the transcript. The verifier labels
each cited sentence from the sentence and its cited notes only -- what the phone has. Reported:
verifier accuracy on the held-out sentences, and the transcript-judged contradiction rate of
  greedy / pick-min-predicted-contradictions / pick-then-drop-predicted-contradicted-sentences.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.build_sentence_verifier_rows import PROMPT, cited_notes  # noqa: E402
from eval.run_student_vllm import merged_model_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default="openbmb/MiniCPM5-2B")
    ap.add_argument("--cands", nargs="+", default=["v2-minicpm5", "bo4-s0", "bo4-s1", "bo4-s2", "bo4-s3"])
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    items = []                                          # (cand, sid, sentence, judge verdict, prompt)
    for c in args.cands:
        for r in json.load(open(f"reports/judge_prose_tx_{c}.json", encoding="utf-8"))["rows"]:
            run = json.load(open(f"runs/student/{c}/{r['id']}.json", encoding="utf-8"))
            cn = cited_notes(r["sentence"], run["notes"])
            p = PROMPT.format(notes="\n".join(cn), sentence=r["sentence"]) if cn else None
            items.append((c, r["id"], r["sentence"], r["verdict"], p))
    llm = LLM(model=merged_model_dir(args.base, args.adapter), max_model_len=8192, gpu_memory_utilization=0.8)
    todo = [i for i, x in enumerate(items) if x[4]]
    outs = llm.chat([[{"role": "user", "content": items[i][4]}] for i in todo], SamplingParams(max_tokens=4, temperature=0),
                    chat_template_kwargs={"enable_thinking": False})
    pred = {i: o.outputs[0].text.strip()[:3] for i, o in zip(todo, outs)}
    gold = {"supported": "一致", "contradicted": "矛盾", "unsupported": "無依據"}
    ok = [pred[i].startswith(gold[items[i][3]][:2]) for i in todo if items[i][3] in gold]
    tp = sum(1 for i in todo if items[i][3] == "contradicted" and pred[i].startswith("矛盾"))
    pp = sum(1 for i in todo if pred[i].startswith("矛盾"))
    pos = sum(1 for i in todo if items[i][3] == "contradicted")
    print(f"verifier accuracy {sum(ok)/len(ok):.1%}; 矛盾 precision {tp/max(1,pp):.1%} recall {tp/max(1,pos):.1%}")

    by = {}
    for i, (c, sid, s, v, _) in enumerate(items):
        by.setdefault(sid, {}).setdefault(c, []).append((v, pred.get(i, "")))
    judged = ("supported", "contradicted", "unsupported")

    def rate(choice, drop=False):
        c = n = kept = tot = 0
        for sid, cands in by.items():
            for v, p in cands[choice(sid, cands)]:
                if v not in judged:
                    continue
                tot += 1
                if drop and p.startswith("矛盾"):
                    continue
                kept += 1
                c += v == "contradicted"
                n += 1
        return f"{c/n:.1%} contradicted ({c/len(by):.2f}/session), {kept/tot:.0%} sentences kept"

    greedy = lambda sid, cands: args.cands[0]                              # noqa: E731
    score = lambda xs: sum(p.startswith("矛盾") + 0.5 * p.startswith("無依據") for _, p in xs) / max(1, len(xs))  # noqa: E731
    pick = lambda sid, cands: min(cands, key=lambda k: score(cands[k]))   # noqa: E731
    print("greedy            ", rate(greedy))
    print("greedy + drop     ", rate(greedy, True))
    print("best-of-5         ", rate(pick))
    print("best-of-5 + drop  ", rate(pick, True))
    sys.stdout.flush()
    import psutil
    for ch in psutil.Process().children(recursive=True):
        ch.kill()
    os._exit(0)


if __name__ == "__main__":
    main()
