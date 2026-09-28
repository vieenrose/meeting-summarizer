"""Write each session a prose abstract, alongside its numbered summary.

The numbered summary is built for checking: one point per topic, each carrying its own
timestamp, so coverage and citations can be verified mechanically. That makes it a good
verification object and a poor thing to read -- six stacked points state what happened
without ever saying what the meeting was.

The prose abstract is written from the same notes as the numbered points, so it adds no
new facts, and it is checked the same way: citations must resolve against the transcript,
all three thirds must be represented, and Simplified characters are disqualifying. Prose
loses the one-topic-per-line structure, so the third check runs over the whole paragraph
rather than per point.

It is written last and stored beside the numbered summary, never in place of it. The
numbered form stays the training target and the audit surface.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.zen_client import ZenChat  # noqa: E402
from eval.zen_probe import script_mix  # noqa: E402
from summarizer.ingest import parse_line, resolve_citation  # noqa: E402
from summarizer.pipeline import uncoverable_thirds  # noqa: E402

CITE = re.compile(r"[\[［](\d{1,2}(?::\d{2}){1,2})[\]］]")
MIN_CHARS, MAX_CHARS = 280, 560

PROMPT = (
    "以下是整場會議依序寫下的筆記：\n\n{notes}\n\n"
    "請根據這些筆記，寫一段繁體中文的會議紀要（散文，不要編號、不要條列）：\n"
    "0. 全文 {lo} 到 {hi} 字（不含時間標記），分成二到三個段落。\n"
    "1. 開頭一句先交代這場會議在做什麼、最後的結果是什麼，讓人讀完第一句就知道重點。\n"
    "2. 接著說明過程：誰主張什麼、爭點在哪、關鍵數字，最後寫決議、待辦與負責人。\n"
    "3. 只寫筆記中有的內容，不要推測立場或動機，不要加入筆記以外的背景知識。\n"
    "4. 重要事實後面附上出處時間，格式 [時間]，時間必須來自筆記；全文至少六處。\n"
    "5. 會議前、中、後三段都要寫到。\n只輸出紀要本文。"
)


def body_len(text):
    return len(re.sub(r"[\[［][^\]］]*[\]］]|\s", "", text))


def check(text, lines):
    problems = []
    if not text.strip():
        return ["沒有內容"]
    if re.search(r"^\s*(\d+[.、]|[-*])", text, re.M):
        problems.append("出現條列或編號，請改寫成散文")
    n = body_len(text)
    if n < MIN_CHARS:
        problems.append(f"全文 {n} 字，太短，請補足到 {MIN_CHARS} 字以上")
    if n > MAX_CHARS:
        problems.append(f"全文 {n} 字，超過 {MAX_CHARS} 字，請精簡")
    simp, trad = script_mix(text)
    if simp:
        problems.append(f"出現 {simp} 個簡體字，請全部改為繁體")
    cites = CITE.findall(text)
    indices, bad = [], []
    for c in cites:
        idx = resolve_citation(c, lines)
        (indices if idx is not None else bad).append(idx if idx is not None else c)
    if bad:
        problems.append("以下時間不存在於逐字稿：" + "、".join(f"[{b}]" for b in bad))
    if len(indices) < 6:
        problems.append(f"只有 {len(indices)} 處有效出處時間，請補到六處以上")
    total = max(1, len(lines))
    covered = {min(2, 3 * i // total) for i in indices}
    missing = {0, 1, 2} - covered - uncoverable_thirds(lines)
    if missing:
        names = {0: "前段", 1: "中段", 2: "後段"}
        problems.append("缺少會議" + "、".join(names[m] for m in sorted(missing)) + "的內容")
    return problems


# Evidence-first reduce (distill/build_evidence_rows.py): each sentence is preceded by the notes it
# condenses, quoted, and the quotes are stripped before the prose is checked or shown. Measured
# against the transcript, reduce doubles the contradiction rate even from gold notes (7% -> 22%):
# the errors are made while merging notes, so the merge is made explicit.
PROSE_EVIDENCE_RULE = "\n6. 每一句之前，先以〔據「…」〕照抄該句所根據的筆記原文（最多兩則，每則不超過 40 字），再寫該句。"
EVIDENCE = re.compile(r"〔據「.*?」〕\s*")


def prompt_for(run, evidence=False):
    notes = "\n".join(f"[{n['ts']}] {n['text']}" for n in run["notes"])
    p = PROMPT.format(notes=notes, lo=MIN_CHARS, hi=MAX_CHARS)
    return p.replace("\n只輸出紀要本文。", PROSE_EVIDENCE_RULE + "\n只輸出紀要本文。") if evidence else p


def write_one(chat, run, lines, turns=3, evidence=False):
    messages = [{"role": "user", "content": prompt_for(run, evidence)}]
    best, log = ("", ["未產生"]), []
    for _ in range(turns):
        reply = chat(messages).strip()
        text = "\n\n".join(p.strip() for p in EVIDENCE.sub("", reply).split("\n") if p.strip())
        problems = check(text, lines)
        log.append({"step": "prose", "messages": list(messages), "reply": reply,
                    "problems": problems})
        if text and (not best[0] or len(problems) < len(best[1])):
            best = (text, problems)
        if not problems:
            break
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content":
                      "請修正以下問題後重新輸出完整紀要：\n" + "\n".join(problems)}]
    return best[0], best[1], log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/gold/w4000")
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--model", default="minimax-m3")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--max-output-tokens", type=int, default=24000)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--turns", type=int, default=3, help="repair turns; dense sessions need more")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--force", action="store_true", help="rewrite sessions that already have prose")
    args = ap.parse_args()

    ids = args.only or sorted(f[:-5] for f in os.listdir(args.run_dir) if f.endswith(".json"))

    def one(sid):
        path = os.path.join(args.run_dir, sid + ".json")
        run = json.load(open(path, encoding="utf-8"))
        if run.get("prose") and not args.force:
            return sid, None, ["已存在，略過"]
        lines = [parse_line(l) for l in
                 open(os.path.join(args.transcripts, sid + ".txt"), encoding="utf-8")
                 .read().splitlines() if l.strip()]
        chat = ZenChat(args.model, max_tokens=args.max_output_tokens, timeout=900, retries=3)
        text, problems, log = write_one(chat, run, lines, args.turns)
        if args.apply and text:
            run["prose"] = text
            run["prose_problems"] = problems
            run.setdefault("log", []).extend(log)
            json.dump(run, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return sid, text, problems

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for sid, text, problems in pool.map(one, ids):
            status = "OK" if not problems else "; ".join(problems)
            print(f"{sid}: {body_len(text or '')} 字 -- {status}")
            if text and args.only:
                print(text)


if __name__ == "__main__":
    main()
