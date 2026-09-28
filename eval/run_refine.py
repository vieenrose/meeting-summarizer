"""The third deployable arm: iterative refine, with no notes at all.

Read each window in order and rewrite a single running summary after every one. This is the other
shape that fits a phone -- state stays one summary instead of a growing note list -- and it is the
only deployable alternative to the notes pipeline that has not been measured here.

Two reasons to expect it to lose, both worth testing rather than assuming:
  * rewriting rather than appending means an earlier fact can be dropped at any step and never
    recovered, whereas a note, once written, survives to synthesis;
  * carried state is what made the teacher answer NOTHING-NEW while describing the votes it had
    just read, and refine carries state by construction.
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
from summarizer.pipeline import (ChatClient, NotesPipeline, PipelineConfig,  # noqa: E402
                                 make_windows)

FIRST = """以下是一場會議逐字稿的第 1 段（共 {total} 段，語音辨識結果，可能有錯字）。

{block}

請寫出目前為止的會議摘要，最多 {n} 點，每點一行以「1.」編號開頭，句尾附上出處時間 [時間]，
時間必須照抄逐字稿中真實存在的一行。只寫逐字稿中有的內容。只輸出編號清單。"""

REFINE = """這是目前為止的會議摘要：

{summary}

以下是逐字稿的第 {k} 段（共 {total} 段）：

{block}

請根據新內容更新摘要：保留仍然重要的既有內容（連同其出處時間），加入本段新的決議、待辦、數字與爭點，
必要時修正先前寫錯的部分。最多 {n} 點，每點一行以「1.」編號開頭，句尾附上出處時間 [時間]，
時間必須照抄逐字稿中真實存在的一行。只輸出編號清單。"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--out", default="runs/refine/w4000")
    ap.add_argument("--window-tokens", type=int, default=4000)
    ap.add_argument("--points", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    def count_tokens(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    chat = ChatClient(args.base_url, args.model, max_tokens=900)
    checker = NotesPipeline(chat, count_tokens, PipelineConfig(summary_points=args.points))

    def one(path: str) -> str:
        stem = os.path.splitext(os.path.basename(path))[0]
        dest = os.path.join(args.out, f"{stem}.json")
        if os.path.exists(dest):
            return f"skip {stem}"
        lines = [parse_line(l) for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]
        windows = make_windows(lines, args.window_tokens, count_tokens)
        t0, log, points = time.time(), [], []

        for k, window in enumerate(windows, 1):
            block = "\n".join(l.render() for l in window)
            prompt = (FIRST.format(total=len(windows), block=block, n=args.points) if k == 1 else
                      REFINE.format(summary="\n".join(points) or "（尚無摘要）", k=k,
                                    total=len(windows), block=block, n=args.points))
            reply = chat([{"role": "user", "content": prompt}])
            new_points = [p.strip() for p in reply.splitlines() if re.match(r"^\s*\d+[.、]", p)]
            log.append({"window": k, "reply": reply, "kept": len(new_points), "rejected": []})
            if new_points:                      # an empty reply must not wipe the summary
                points = new_points

        problems = checker._check_summary(points, lines)
        log.append({"step": "write", "reply": "\n".join(points), "problems": problems})
        result = {"mode": "refine", "windows": len(windows), "notes": [],
                  "windows_without_notes": [], "summary": points, "log": log,
                  "elapsed_s": time.time() - t0}
        json.dump(result, open(dest, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return f"{stem}: {len(windows)} windows, {len(points)} points, problems={problems}"

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for line in pool.map(one, sorted(glob.glob(os.path.join(args.transcripts, "*.txt")))):
            print(line, flush=True)


if __name__ == "__main__":
    main()
