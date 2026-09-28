"""Evaluate a student with vLLM: the same pipeline as run_student_hf.py, about an order faster.

Checking GRPO checkpoints overnight needs evaluation in minutes, not the ~25 of transformers
generate(). The pipeline is unchanged; each session runs in its own thread, and their chat calls are
gathered into shared vLLM batches by a small batcher, so seven sessions decode together.

An adapter is merged into the base and saved first, because vLLM loads Gemma 4 only through the
config overrides in distill/grpo_vllm.py, which read config.json from the model directory.
"""
import argparse
import json
import os
import queue
import shutil
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.pipeline import NotesPipeline, PipelineConfig  # noqa: E402
from summarizer.ingest import parse_line  # noqa: E402
from distill.write_prose import write_one as write_prose  # noqa: E402

BASE = "google/gemma-4-E2B-it"


def merged_model_dir(base, adapter, out_root="runs/merged"):
    if not adapter:
        return base
    dest = os.path.join(out_root, adapter.strip("/").replace("/", "__"))
    if os.path.exists(os.path.join(dest, "config.json")):
        return dest
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m.save_pretrained(dest, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(dest)
    # The merged model is saved without the KV-shared layers' k_proj/v_proj/k_norm (transformers
    # does not build them), which vLLM needs. LoRA never touches them, so copy them from the base.
    _restore_shared_kv(base, dest)
    _copy_processor_files(base, dest)
    del m
    torch.cuda.empty_cache()
    return dest


def _copy_processor_files(base, dest):
    """vLLM builds Gemma 4's multimodal processor even for text, from files save_pretrained omits."""
    import glob
    snap = base if os.path.isdir(base) else os.path.dirname(glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{base.replace('/', '--')}/snapshots/*/config.json"))[0])
    for f in os.listdir(snap):
        if f.endswith((".json", ".jinja")) and f not in ("config.json", "model.safetensors.index.json"):
            if not os.path.exists(os.path.join(dest, f)):
                shutil.copy(os.path.join(snap, f), os.path.join(dest, f))


def _restore_shared_kv(base, dest):
    import glob
    from safetensors import safe_open
    from safetensors.torch import save_file
    snap = base if os.path.isdir(base) else os.path.dirname(glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{base.replace('/', '--')}/snapshots/*/config.json"))[0])
    have = set()
    for f in glob.glob(os.path.join(dest, "*.safetensors")):
        with safe_open(f, "pt") as h:
            have.update(h.keys())
    extra = {}
    for f in glob.glob(os.path.join(snap, "*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                if k not in have and ".self_attn." in k:
                    extra[k] = h.get_tensor(k)
    if extra:
        save_file(extra, os.path.join(dest, "model-shared-kv.safetensors"), metadata={"format": "pt"})
        index = os.path.join(dest, "model.safetensors.index.json")
        if os.path.exists(index):
            idx = json.load(open(index))
            idx["weight_map"].update({k: "model-shared-kv.safetensors" for k in extra})
            json.dump(idx, open(index, "w"), indent=1)
    print(f"restored {len(extra)} KV-shared attention tensors from the base", flush=True)


class Batcher:
    """Collect chat calls from session threads and run them as one vLLM batch."""

    def __init__(self, llm, params, wait=0.05, max_batch=64):
        self.llm, self.params, self.wait, self.max_batch = llm, params, wait, max_batch
        self.q = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def __call__(self, messages):
        fut = Future()
        self.q.put((messages, fut))
        return fut.result()

    def _loop(self):
        while True:
            batch = [self.q.get()]
            deadline = time.time() + self.wait
            while len(batch) < self.max_batch:
                left = deadline - time.time()
                if left <= 0:
                    break
                try:
                    batch.append(self.q.get(timeout=left))
                except queue.Empty:
                    break
            try:
                outs = self.llm.chat([m for m, _ in batch], self.params, use_tqdm=False,
                                     chat_template_kwargs={"enable_thinking": False})
                for (_, fut), o in zip(batch, outs):
                    fut.set_result(o.outputs[0].text)
            except Exception as e:                       # noqa: BLE001
                for _, fut in batch:
                    fut.set_exception(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="data/split_v2.json")
    ap.add_argument("--transcripts", default="data/v2/transcripts")
    ap.add_argument("--window-tokenizer", default="Qwen/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--mem", type=float, default=0.6)
    ap.add_argument("--prompt-version", choices=["v1", "v2"], default="v2")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--window-tokens", type=int, default=4000)
    ap.add_argument("--evidence", action="store_true", help="evidence-first notes prompt")
    ap.add_argument("--extract", action="store_true", help="extractive notes prompt")
    ap.add_argument("--keep-evidence", action="store_true")
    ap.add_argument("--acts", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from distill.grpo_vllm import gemma4_overrides

    sessions = args.only or json.load(open(args.split))["heldout"]
    model_dir = merged_model_dir(args.base, args.adapter)
    llm = LLM(model=model_dir, max_model_len=args.max_model_len, gpu_memory_utilization=args.mem,
              hf_overrides=gemma4_overrides(args.base),
              **({"limit_mm_per_prompt": {"image": 0, "audio": 0}} if "gemma" in args.base.lower() else {}))
    chat = Batcher(llm, SamplingParams(max_tokens=1500, temperature=0))
    wtok = AutoTokenizer.from_pretrained(args.window_tokenizer)
    count = lambda t: len(wtok.encode(t, add_special_tokens=False))  # noqa: E731
    os.makedirs(args.out, exist_ok=True)

    def one(sid):
        dest = os.path.join(args.out, sid + ".json")
        if os.path.exists(dest):
            return f"skip {sid}"
        t0 = time.time()
        text = open(os.path.join(args.transcripts, sid + ".txt"), encoding="utf-8").read()
        result = NotesPipeline(chat, count, PipelineConfig(prompt_version=args.prompt_version, window_tokens=args.window_tokens, evidence=args.evidence, extract=args.extract, keep_evidence=args.keep_evidence, acts=args.acts)).run(text)
        lines = [parse_line(l) for l in text.splitlines() if l.strip()]
        prose, prose_problems, prose_log = write_prose(chat, result, lines, turns=1, evidence=args.evidence)
        result["prose"] = prose
        result["prose_problems"] = prose_problems
        result.setdefault("log", []).extend(prose_log)
        result["elapsed_s"] = time.time() - t0
        json.dump(result, open(dest, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return f"{sid}: {len(result['notes'])} notes, {len(result['summary'])} points, prose {len(prose)} chars, {result['elapsed_s']:.0f}s"

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(sessions)) as pool:
        for line in pool.map(one, sessions):
            print(line, flush=True)
    print(f"total {time.time() - t0:.0f}s", flush=True)
    # vLLM's engine core runs in a child process that outlived this one after a normal exit and
    # kept 21 GB of the GPU, which would starve every later checkpoint evaluation. Kill it.
    import psutil
    for child in psutil.Process().children(recursive=True):
        child.kill()
    os._exit(0)


if __name__ == "__main__":
    main()
