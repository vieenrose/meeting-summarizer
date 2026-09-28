"""Control experiment: does the per-window note step help, or is it just scaffolding?

Feeds the whole transcript in one prompt and asks for the same summary, with the same citation and
coverage requirements the notes pipeline is held to. Only the intermediate representation differs,
so any gap is attributable to the notes.

Single-pass is not deployable -- a 2-hour meeting is ~23k tokens and the phone has 2.5 GB and a
40-minute budget -- but the teacher can do it at 64k, and the answer decides what the student
should be trained to imitate.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.ingest import parse_line  # noqa: E402
from summarizer.pipeline import ChatClient, NotesPipeline, PipelineConfig  # noqa: E402

PROMPT = """以下是一場會議的完整逐字稿（語音辨識結果，可能有錯字）。

{transcript}

請寫出 {n} 點繁體中文會議摘要（總長不超過 550 字，只寫逐字稿中有的內容，不要推測立場或動機）：
1. 每點一行，以「1.」「2.」編號開頭，句尾附上出處時間，格式 [時間]，時間必須照抄逐字稿中真實存在的一行。
2. 優先寫決議、待辦與負責人、關鍵數字、爭議與未決事項，保留誰主張什麼。
3. 會議前、中、後三段都要有內容。
注意：講者標籤（S1、S2）由語音辨識自動判斷，常把不同的人標成同一位；請從發言內容判斷身分。
只輸出編號清單。"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--out", default="runs/singlepass")
    ap.add_argument("--max-input-tokens", type=int, default=62000)
    ap.add_argument("--points", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--backend", choices=["vllm", "zen"], default="vllm")
    ap.add_argument("--only", nargs="*", default=None, help="session ids to run")
    ap.add_argument("--max-output-tokens", type=int, default=900,
                    help="reasoning models need room to think before answering: 900 truncated "
                         "nemotron-3-ultra mid-thought and it never reached the summary")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    if args.backend == "zen":
        from eval.zen_client import ZenChat
        chat = ZenChat(args.model, max_tokens=args.max_output_tokens, timeout=900)
    else:
        chat = ChatClient(args.base_url, args.model, max_tokens=args.max_output_tokens)
    # Reuse the pipeline's own summary checker so both arms are held to the same standard.
    checker = NotesPipeline(chat, lambda t: len(tok.encode(t, add_special_tokens=False)),
                            PipelineConfig(summary_points=args.points))

    def one(path: str) -> str:
        stem = os.path.splitext(os.path.basename(path))[0]
        dest = os.path.join(args.out, f"{stem}.json")
        if os.path.exists(dest):
            return f"skip {stem}"
        text = open(path, encoding="utf-8").read()
        lines = [parse_line(l) for l in text.splitlines() if l.strip()]
        n_tokens = len(tok.encode(text, add_special_tokens=False))
        if n_tokens > args.max_input_tokens:
            return f"{stem}: SKIPPED, {n_tokens} tokens exceeds the single-pass window"

        t0 = time.time()
        messages = [{"role": "user", "content": PROMPT.format(transcript=text, n=args.points)}]
        log, best = [], ([], ["沒有編號清單"])
        for _ in range(2):
            reply = chat(messages)
            points = [p.strip() for p in reply.splitlines()
                      if re.match(r"^\s*\d+[.、]", p) and re.search(r"[\u4e00-\u9fff]", p)]
            problems = checker._check_summary(points, lines)
            log.append({"step": "singlepass", "reply": reply, "problems": problems})
            if points and (not best[0] or len(problems) <= len(best[1])):
                best = (points, problems)
            if not problems:
                break
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": "請修正以下問題後重新輸出完整清單：\n" + "\n".join(problems)}]

        result = {"mode": "singlepass", "input_tokens": n_tokens, "windows": 1, "notes": [],
                  "windows_without_notes": [], "summary": best[0], "log": log,
                  "elapsed_s": time.time() - t0}
        json.dump(result, open(dest, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return (f"{stem}: {n_tokens} tokens, {len(best[0])} points, "
                f"{result['elapsed_s']:.0f}s, problems={best[1]}")

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        files = sorted(glob.glob(os.path.join(args.transcripts, "*.txt")))
        if args.only:
            files = [f for f in files if os.path.basename(f)[:-4] in set(args.only)]
        for line in pool.map(one, files):
            print(line, flush=True)


if __name__ == "__main__":
    main()
