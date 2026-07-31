"""Extend Falcon-H1's tokenizer with zh-TW multi-character tokens.

Measured problem: Falcon-H1 spends 0.98 chars/token on Traditional Chinese — roughly one
token per character, with no multi-character units. Qwen3.5 gets 1.67. That 70% token
inflation multiplies KV cache, prefill time and context pressure on exactly the long
zh-TW meetings we target (3h zh: 51.8k tokens on Falcon vs 30.4k on Qwen).

Fix: train a zh-TW BPE on in-domain text, keep the merges that are genuinely multi-char
and absent from the base vocab, add them, and initialize each new embedding as the mean
of the pieces the base tokenizer would have produced — so the model starts out
representing the new token roughly where it already represented its parts. A short
continued-pretraining pass then settles them.

Usage:
  python train/extend_tokenizer.py --n-new 16000 --out train/out/falcon-zhtw-tok
"""
import argparse
import json
from pathlib import Path

import torch
from tokenizers import SentencePieceBPETokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


def zh_corpus():
    """In-domain zh-TW text: the meeting transcripts we actually summarize."""
    out = []
    p = ROOT / "data/normalized/vcsum.jsonl"
    for line in p.open():
        d = json.loads(line)
        for u in d["utterances"]:
            t = u["text"].strip()
            if t:
                out.append(t)
    # plus every zh-TW target the teacher produced (summaries, notes, action items)
    j = ROOT / "distill/targets_judged.jsonl"
    if j.exists():
        for line in j.open():
            d = json.loads(line)
            eff = d["meta"].get("tgt") or d["lang"]
            if eff == "zh-TW":
                out.append(d["completion"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="tiiuae/Falcon-H1-1.5B-Instruct")
    ap.add_argument("--n-new", type=int, default=16000)
    ap.add_argument("--out", default=str(ROOT / "train/out/falcon-zhtw-base"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    corpus = zh_corpus()
    chars = sum(len(t) for t in corpus)
    print(f"corpus: {len(corpus):,} lines / {chars:,} chars")

    # Train a zh BPE and keep only multi-character candidates the base lacks.
    bpe = SentencePieceBPETokenizer()
    bpe.train_from_iterator(corpus, vocab_size=args.n_new * 2, min_frequency=8,
                            special_tokens=[], show_progress=False)
    have = set(tok.get_vocab())
    cand = []
    for piece in bpe.get_vocab():
        s = piece.replace("▁", "")
        if len(s) < 2 or s in have:
            continue
        if not all("一" <= c <= "鿿" for c in s):
            continue          # pure-Han tokens only; punctuation/latin already fine
        cand.append(s)
    # prefer the tokens that save the most tokens: longest first
    cand.sort(key=lambda s: (-len(s), s))
    new_tokens = cand[: args.n_new]
    print(f"adding {len(new_tokens):,} zh tokens (e.g. {new_tokens[:8]})")

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32)
    emb_in = model.get_input_embeddings()
    old_vocab = emb_in.weight.shape[0]

    # capture piece decompositions BEFORE the vocab changes
    decomp = [tok(t, add_special_tokens=False)["input_ids"] for t in new_tokens]

    n_added = tok.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tok))
    emb = model.get_input_embeddings().weight.data
    # add_tokens dedupes, so never assume row = old_vocab + i: ask the tokenizer.
    rows = [tok.convert_tokens_to_ids(t) for t in new_tokens]
    with torch.no_grad():
        for row, ids in zip(rows, decomp):
            if ids and row is not None and old_vocab <= row < emb.shape[0]:
                emb[row] = emb[torch.tensor(ids)].mean(dim=0)
    if not model.config.tie_word_embeddings:
        out_w = model.get_output_embeddings().weight.data
        with torch.no_grad():
            for row, ids in zip(rows, decomp):
                if ids and row is not None and old_vocab <= row < out_w.shape[0]:
                    out_w[row] = out_w[torch.tensor(ids)].mean(dim=0)
    print(f"added {n_added:,} tokens; embedding rows {old_vocab:,} -> {emb.shape[0]:,}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(args.out)
    tok.save_pretrained(args.out)

    # report the win on held-out zh text
    sample = "".join(corpus[:400])
    base = AutoTokenizer.from_pretrained(args.base)
    print(f"zh chars/token: {len(sample)/len(base.encode(sample)):.2f} -> "
          f"{len(sample)/len(tok.encode(sample)):.2f}")


if __name__ == "__main__":
    main()
