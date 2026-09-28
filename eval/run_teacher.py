"""Run the summarizer pipeline over evaluation transcripts against an OpenAI-compatible endpoint.

Example (teacher window sweep):
  python eval/run_teacher.py --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.6-35B-A3B \
      --transcripts data/transcripts --window-tokens 2000 4000 8000 --out runs/teacher
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.pipeline import ChatClient, NotesPipeline, PipelineConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None, help="HF id for token counting (default: --model)")
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--window-tokens", type=int, nargs="+", default=[4000])
    ap.add_argument("--out", default="runs/teacher")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--backend", choices=["vllm", "zen"], default="vllm")
    ap.add_argument("--max-output-tokens", type=int, default=900)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--prompt-version", choices=["v1", "v2"], default="v1")
    ap.add_argument("--no-chat-template-kwargs", action="store_true",
                    help="Mistral tokenizer-mode servers reject chat_template_kwargs entirely")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    def count_tokens(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    files = sorted(glob.glob(os.path.join(args.transcripts, "*.txt")))
    if args.only:
        files = [f for f in files if os.path.basename(f)[:-4] in set(args.only)]
    if args.backend == "zen":
        from eval.zen_client import ZenChat
        chat = ZenChat(args.model, max_tokens=args.max_output_tokens, timeout=900)
    else:
        chat = ChatClient(args.base_url, args.model, max_tokens=args.max_output_tokens,
                          no_chat_template_kwargs=args.no_chat_template_kwargs)

    for window in args.window_tokens:
        out_dir = os.path.join(args.out, f"w{window}")
        os.makedirs(out_dir, exist_ok=True)
        pipeline = NotesPipeline(chat, count_tokens, PipelineConfig(window_tokens=window, prompt_version=args.prompt_version))

        def one(path: str) -> str:
            stem = os.path.splitext(os.path.basename(path))[0]
            dest = os.path.join(out_dir, f"{stem}.json")
            if os.path.exists(dest):
                return f"skip {stem}"
            t0 = time.time()
            result = pipeline.run(open(path, encoding="utf-8").read())
            result["elapsed_s"] = time.time() - t0
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=1)
            return (f"w{window} {stem}: {result['windows']} windows, {len(result['notes'])} notes, "
                    f"{len(result['windows_without_notes'])} empty windows, "
                    f"{len(result['summary'])} points, {result['elapsed_s']:.0f}s")

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            for line in pool.map(one, files):
                print(line, flush=True)


if __name__ == "__main__":
    main()
