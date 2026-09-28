"""Build SILVER key-fact lists from a transcript with an independent model.

These are NOT the gold lists described in data/keyfacts/README.md. A human still has to write
those, and only they can settle the coverage gate. Silver lists exist so coverage can be compared
between pipelines overnight, before any human annotation exists.

Guard rails that keep the comparison meaningful:
  * The extractor must not be the model being evaluated, nor its family (default: Gemma, while the
    summarizer under test is Qwen). Circularity is the whole risk here.
  * Extraction reads the transcript in plain windows with no agent protocol, so it shares no
    machinery with the reading agent it will score.
Every file records the extractor so a silver score can never be mistaken for a gold one.
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
from summarizer.ingest import parse_line  # noqa: E402

PROMPT = """以下是一場立法院委員會會議逐字稿的一部分（語音辨識結果，可能有錯字）。

{block}

請列出這段逐字稿中「會議紀錄必須記載」的事實，每行一則，格式：
[時間]|類型|事實
類型只能是 DECISION（決議、通過、保留、退回）、ACTION（待辦事項與負責單位）、NUMBER（金額、日期、票數、條號）、POSITION（某人主張或反對什麼）、OPEN（未解決的爭點）。
規則：
1. [時間] 必須照抄逐字稿中真實存在的一行時間。
2. 一行一則事實，用繁體中文寫成可查證的完整句子。
3. 只寫逐字稿明確說出的內容，不要推測。程序性發言（點名、休息）不必列出。
4. 這段若沒有值得記載的事實，輸出一行：NONE
最多 6 則。"""

LINE_RE = re.compile(r"^\[(\d{1,2}(?::\d{2}){1,2})\]\s*\|\s*(DECISION|ACTION|NUMBER|POSITION|OPEN)\s*\|\s*(.+)$")


def extract(chat: ChatClient, transcript_path: str, window_chars: int) -> list:
    lines = [parse_line(l) for l in open(transcript_path, encoding="utf-8").read().splitlines() if l.strip()]
    starts = {l.start_s for l in lines}
    facts, block, size = [], [], 0
    blocks = []
    for line in lines:
        block.append(line)
        size += len(line.text)
        if size >= window_chars:
            blocks.append(block)
            block, size = [], 0
    if block:
        blocks.append(block)

    for chunk in blocks:
        text = "\n".join(l.render() for l in chunk)
        reply = chat([{"role": "user", "content": PROMPT.format(block=text)}])
        for raw in reply.splitlines():
            m = LINE_RE.match(raw.strip())
            if not m:
                continue
            ts, kind, fact = m.group(1), m.group(2), m.group(3).strip()
            secs = parse_line(f"[{ts}] x").start_s
            if secs in starts:  # drop invented timestamps, same rule the agent is held to
                facts.append({"id": len(facts) + 1, "type": kind, "fact": fact, "ts": ts})
    return facts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--out", default="data/keyfacts_silver")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True, help="must not be the model or family under evaluation")
    ap.add_argument("--window-chars", type=int, default=6000)
    ap.add_argument("--parallel", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    chat = ChatClient(args.base_url, args.model, max_tokens=900)

    def one(path: str) -> str:
        stem = os.path.splitext(os.path.basename(path))[0]
        dest = os.path.join(args.out, f"{stem}.json")
        if os.path.exists(dest):
            return f"skip {stem}"
        facts = extract(chat, path, args.window_chars)
        json.dump({"session": stem, "annotator": f"SILVER:{args.model}",
                   "source": "auto-extracted from transcript; NOT a human gold list",
                   "facts": facts}, open(dest, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        kinds = {}
        for f in facts:
            kinds[f["type"]] = kinds.get(f["type"], 0) + 1
        return f"{stem}: {len(facts)} facts {kinds}"

    files = sorted(glob.glob(os.path.join(args.transcripts, "*.txt")))
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for line in pool.map(one, files):
            print(line, flush=True)


if __name__ == "__main__":
    main()
