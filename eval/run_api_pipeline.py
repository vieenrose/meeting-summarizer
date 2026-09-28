"""The full map-reduce pipeline (notes -> points -> prose) against any OpenAI-compatible endpoint,
e.g. Bonsai 2 27B on the PrismML llama-server. Same prompts and output shape as
eval/run_student_vllm.py, so eval/score_v2.py and the judges read it unchanged.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.write_prose import write_one as write_prose  # noqa: E402
from summarizer.ingest import parse_line  # noqa: E402
from summarizer.pipeline import ChatClient, NotesPipeline, PipelineConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True, help="comma-separated base URLs")
    ap.add_argument("--model", default="bonsai")
    ap.add_argument("--split", default="data/split_v2.json")
    ap.add_argument("--transcripts", default="data/v2/transcripts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--evidence", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--acts", action="store_true")
    ap.add_argument("--window-tokenizer", default="Qwen/Qwen3.6-35B-A3B-FP8")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    wtok = AutoTokenizer.from_pretrained(a.window_tokenizer)
    count = lambda t: len(wtok.encode(t, add_special_tokens=False))  # noqa: E731
    urls = a.urls.split(",")
    os.makedirs(a.out, exist_ok=True)

    def one(i_sid):
        i, sid = i_sid
        dest = os.path.join(a.out, sid + ".json")
        if os.path.exists(dest):
            return f"skip {sid}"
        chat = ChatClient(urls[i % len(urls)], a.model, max_tokens=1500)
        text = open(os.path.join(a.transcripts, sid + ".txt"), encoding="utf-8").read()
        t0 = time.time()
        cfg = PipelineConfig(prompt_version="v2", evidence=a.evidence, extract=a.extract, acts=a.acts)
        result = NotesPipeline(chat, count, cfg).run(text)
        lines = [parse_line(l) for l in text.splitlines() if l.strip()]
        prose, problems, log = write_prose(chat, result, lines, turns=1, evidence=a.evidence)
        result.update(prose=prose, prose_problems=problems, elapsed_s=time.time() - t0)
        result["log"].extend(log)
        json.dump(result, open(dest, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return f"{sid}: {len(result['notes'])} notes, prose {len(prose)} chars, {result['elapsed_s']:.0f}s"

    sids = json.load(open(a.split))["heldout"]
    with ThreadPoolExecutor(a.parallel) as ex:
        for line in ex.map(one, enumerate(sids)):
            print(line, flush=True)


if __name__ == "__main__":
    main()
