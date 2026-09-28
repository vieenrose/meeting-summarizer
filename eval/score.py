"""Score summaries for key-fact coverage and per-point faithfulness with an LLM judge.

Inputs
  runs/<...>/<id>.json          agent or baseline output (has "summary")
  data/transcripts/<id>.txt     the transcript the summary was written from
  data/keyfacts/<id>.json       HUMAN-written key facts: {"facts": [{"id": 1, "fact": "...", "ts": "M:SS"}]}

Judge rules
  * Use a judge from a different model family than the student being evaluated
    (this project measured large same-family bias). The judge is a parameter, not a default.
  * 3-point scales only: coverage covered/partial/missing = 1/0.5/0; faithfulness
    supported/unsupported/contradicted, where "contradicted" is an inversion.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.pipeline import ChatClient  # noqa: E402
from summarizer.ingest import parse_line, resolve_citation  # noqa: E402

COVER_PROMPT = """會議摘要：
{summary}

關鍵事實：{fact}

摘要是否表達了這個關鍵事實？只回答一個詞：COVERED（完整表達）、PARTIAL（部分或模糊）、MISSING（沒有提到或說錯）。"""

FAITH_PROMPT = """逐字稿片段（語音辨識結果，可能有錯字）：
{context}

摘要中的一句：{point}

這句話和逐字稿的關係？只回答一個詞：SUPPORTED（有依據）、UNSUPPORTED（逐字稿沒有依據）、CONTRADICTED（與逐字稿相反，例如把否決說成通過）。"""


def judge_word(chat: ChatClient, prompt: str, allowed: tuple) -> str:
    try:
        reply = chat([{"role": "user", "content": prompt}]).upper()
    except Exception as e:  # noqa: BLE001 - a judge hiccup should cost one label, not the run
        print(f"  judge call failed ({type(e).__name__}); counting as UNPARSED", flush=True)
        return "UNPARSED"
    for word in allowed:
        if word in reply:
            return word
    return "UNPARSED"


# The judge is served at 16k context. A summary point can cite many moments, and each pulls in a
# neighbourhood of transcript, so an uncapped context overflowed the window and every request came
# back HTTP 400. Cap the cited moments and the total characters instead of failing.
MAX_CITES_PER_POINT = 6
MAX_CONTEXT_CHARS = 9000


def context_for(point: str, lines, before: int = 3, after: int = 14) -> str:
    """Asymmetric window: a citation marks where a topic starts, and its outcome comes later.

    A symmetric +-4 lines produced a false inversion: the point correctly said a tied motion passed
    on the chair's vote, but the window centred on the motion's start contained only the *previous*
    motion's rejection, and the judge called the point contradicted. Reaching further forward than
    back matches how the transcript actually unfolds.
    """
    picks = set()
    for c in re.findall(r"\[(\d{1,2}(?::\d{2}){1,2})\]", point)[:MAX_CITES_PER_POINT]:
        idx = resolve_citation(c, lines)
        if idx is not None:
            picks.update(range(max(0, idx - before), min(len(lines), idx + after + 1)))
    out, used = [], 0
    for i in sorted(picks):
        rendered = lines[i].render()
        if used + len(rendered) > MAX_CONTEXT_CHARS:
            out.append("…")
            break
        out.append(rendered)
        used += len(rendered)
    return "\n".join(out) or "（沒有有效出處）"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--keyfacts", default="data/keyfacts")
    ap.add_argument("--judge-url", required=True)
    ap.add_argument("--judge-model", required=True)
    args = ap.parse_args()
    chat = ChatClient(args.judge_url, args.judge_model, max_tokens=8)

    rows = []
    for path in sorted(glob.glob(os.path.join(args.run_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        facts_path = os.path.join(args.keyfacts, f"{stem}.json")
        if stem.endswith(".score") or not os.path.exists(facts_path):
            continue
        run = json.load(open(path, encoding="utf-8"))
        lines = [parse_line(l) for l in open(os.path.join(args.transcripts, f"{stem}.txt"),
                                               encoding="utf-8").read().splitlines() if l.strip()]
        summary = "\n".join(run["summary"])
        facts = json.load(open(facts_path, encoding="utf-8"))["facts"]

        cover = [judge_word(chat, COVER_PROMPT.format(summary=summary, fact=f["fact"]),
                            ("COVERED", "PARTIAL", "MISSING")) for f in facts]
        faith = [judge_word(chat, FAITH_PROMPT.format(context=context_for(p, lines), point=p),
                            ("CONTRADICTED", "UNSUPPORTED", "SUPPORTED")) for p in run["summary"]]
        score = {
            "id": stem,
            "coverage": sum({"COVERED": 1, "PARTIAL": 0.5}.get(c, 0) for c in cover) / max(1, len(facts)),
            "faithful_share": sum(f == "SUPPORTED" for f in faith) / max(1, len(faith)),
            "inversion": any(f == "CONTRADICTED" for f in faith),
            "unparsed": cover.count("UNPARSED") + faith.count("UNPARSED"),
            "chars": len(re.sub(r"\[[^\]]*\]|\s", "", summary)),
            "cover_labels": cover,
            "faith_labels": faith,
        }
        json.dump(score, open(os.path.join(args.run_dir, f"{stem}.score.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        rows.append(score)
        print(f"{stem}: coverage {score['coverage']:.2f}, faithful {score['faithful_share']:.2f}, "
              f"inversion {score['inversion']}, {score['chars']} chars", flush=True)

    if rows:
        n = len(rows)
        print(f"\nn={n}  coverage {sum(r['coverage'] for r in rows) / n:.3f}  "
              f"faithful {sum(r['faithful_share'] for r in rows) / n:.3f}  "
              f"inversion rate {sum(r['inversion'] for r in rows) / n:.1%}  "
              f"unparsed {sum(r['unparsed'] for r in rows)}")


if __name__ == "__main__":
    main()
