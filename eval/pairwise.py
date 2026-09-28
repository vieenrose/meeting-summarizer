"""Head-to-head judging against the reference summary.

The requirement is that the shipped summary be comparable to the one the teacher writes in a single
pass over the whole transcript. Absolute scores answer that badly: coverage against a fact list is
noisy at n=39 and its scale is arbitrary. A direct comparison is both more sensitive and closer to
the question actually being asked.

Two precautions, because LLM judges are known to favour whichever answer they see first and
whichever is longer:
  * every pair is judged twice, once in each order, and a verdict only counts as a win if it
    survives both;
  * a disagreement between the two orders is reported as a tie, not silently resolved.
"""
import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.pipeline import ChatClient  # noqa: E402

PROMPT = """以下是同一場會議的兩份摘要，以及該會議必須記載的關鍵事實。

關鍵事實：
{facts}

摘要 A：
{a}

摘要 B：
{b}

請比較兩份摘要，判斷哪一份更好。判斷標準依序為：
1. 涵蓋多少關鍵事實（決議、待辦、數字、爭點）；
2. 是否忠實，有沒有與事實相反或憑空捏造的內容；
3. 是否具體（寫出誰主張什麼、結果如何），而非只列主題。
長度不是優點，內容正確且完整才是。

只回答一個詞：A（A 明顯較好）、B（B 明顯較好）、TIE（兩者相當）。"""


def verdict(chat: ChatClient, facts: str, a: str, b: str) -> str:
    reply = chat([{"role": "user", "content": PROMPT.format(facts=facts, a=a, b=b)}]).strip().upper()
    for token in ("TIE", "A", "B"):
        if re.search(rf"\b{token}\b", reply) or reply.startswith(token):
            return token
    return "UNPARSED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="run dir for the deployable method")
    ap.add_argument("--reference", required=True, help="run dir for the teacher single-pass reference")
    ap.add_argument("--keyfacts", default="data/keyfacts_silver_core")
    ap.add_argument("--judge-url", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()
    chat = ChatClient(args.judge_url, args.judge_model, max_tokens=8)

    def one(path: str):
        stem = os.path.splitext(os.path.basename(path))[0]
        ref_path = os.path.join(args.reference, f"{stem}.json")
        facts_path = os.path.join(args.keyfacts, f"{stem}.json")
        if stem.endswith(".score") or not os.path.exists(ref_path) or not os.path.exists(facts_path):
            return None
        cand = "\n".join(json.load(open(path, encoding="utf-8"))["summary"])
        ref = "\n".join(json.load(open(ref_path, encoding="utf-8"))["summary"])
        if not cand or not ref:
            return None
        facts = "\n".join(f"- {f['fact']}" for f in json.load(open(facts_path, encoding="utf-8"))["facts"])

        # Candidate first, then reference first: only a verdict that survives both counts.
        first = verdict(chat, facts, cand, ref)
        second = verdict(chat, facts, ref, cand)
        if first == "A" and second == "B":
            result = "candidate"
        elif first == "B" and second == "A":
            result = "reference"
        else:
            result = "tie"
        return {"id": stem, "result": result, "raw": [first, second],
                "cand_chars": len(re.sub(r"\[[^\]]*\]|\s", "", cand)),
                "ref_chars": len(re.sub(r"\[[^\]]*\]|\s", "", ref))}

    rows = [r for r in ThreadPoolExecutor(max_workers=args.parallel).map(
        one, sorted(glob.glob(os.path.join(args.candidate, "*.json")))) if r]
    if not rows:
        print("no comparable sessions")
        return

    n = len(rows)
    wins = sum(r["result"] == "candidate" for r in rows)
    ties = sum(r["result"] == "tie" for r in rows)
    losses = sum(r["result"] == "reference" for r in rows)
    consistent = sum(r["raw"][0] != r["raw"][1] or r["raw"][0] == "TIE" for r in rows)
    print(f"\n{args.candidate} vs {args.reference}  n={n}")
    print(f"  candidate better {wins} ({wins/n:.0%}), tie {ties} ({ties/n:.0%}), "
          f"reference better {losses} ({losses/n:.0%})")
    print(f"  not worse than reference: {(wins + ties)/n:.0%}")
    print(f"  judge self-consistent across both orders: {consistent/n:.0%}")
    print(f"  mean length: candidate {sum(r['cand_chars'] for r in rows)/n:.0f} chars, "
          f"reference {sum(r['ref_chars'] for r in rows)/n:.0f}")
    if args.out:
        json.dump({"candidate": args.candidate, "reference": args.reference, "n": n,
                   "candidate_better": wins, "tie": ties, "reference_better": losses,
                   "rows": rows}, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
