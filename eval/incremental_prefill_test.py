"""Does a model, on llama-server, support incremental (chunked) prefill as the realtime agent
would use it? The transcript window is fed as ASR-sized chunks while it "arrives", then the turn
is closed and the model generates.

  1. Render the chat template around a placeholder (/apply-template) -> prefix, suffix.
  2. Tokenize prefix and each chunk separately; send the growing token list to /completion with
     n_predict 0 (prefill only, cache kept). Pass: each call computes only its new tokens.
  3. Close the turn (suffix tokens) and generate greedily. Pass: the close computes only the
     suffix, and the output agrees with a one-shot prefill of the same tokens on a fresh slot as
     far as two one-shot runs agree with each other (chunked and one-shot prefill take different
     kernel paths, so float rounding can flip a late token without any cache fault).
  4. Report how often chunk-wise tokenization differs from tokenizing the whole text (the model
     then sees a slightly unusual split at chunk joins).
"""
import argparse
import json
import sys

import requests

MARK = "⁣PLACEHOLDER⁣"


def post(url, path, body):
    r = requests.post(url + path, json=body, timeout=600)
    r.raise_for_status()
    return r.json()


def tokenize(url, text, special=False):
    return post(url, "/tokenize", {"content": text, "add_special": False, "parse_special": special})["tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8130")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--transcript", default="data/v2/transcripts/ivod_16776.txt")
    ap.add_argument("--chunk-lines", type=int, default=6, help="transcript lines per ASR chunk (~150 tokens)")
    ap.add_argument("--lines", type=int, default=90)
    a = ap.parse_args()
    u = a.url
    system = "你是會議閱讀助理。讀完逐字稿片段後，輸出最多 5 行 NOTE [時間] 內容，最後一行 NEXT。"
    lines = [l for l in open(a.transcript, encoding="utf-8").read().splitlines() if l.strip()][200:200 + a.lines]
    tmpl = post(u, "/apply-template", {"messages": [{"role": "system", "content": system},
                                                     {"role": "user", "content": MARK}],
                                        "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    prefix, suffix = tmpl.split(MARK)
    chunks = ["\n".join(lines[i:i + a.chunk_lines]) + "\n" for i in range(0, len(lines), a.chunk_lines)]
    chunks[-1] = chunks[-1].rstrip("\n")

    toks = tokenize(u, prefix, special=True)
    calls, ok_reuse, first = [], True, True
    for c in [None] + chunks:
        new = [] if c is None else tokenize(u, c)
        toks = toks + new
        r = post(u, "/completion", {"prompt": toks, "n_predict": 0, "cache_prompt": True, "id_slot": 0})
        pn = r["timings"]["prompt_n"]
        expect = len(toks) if first else len(new)
        calls.append((len(new), pn))
        # llama.cpp may re-evaluate the last token or two to produce logits
        if not first and pn > expect + 4:
            ok_reuse = False
        first = False
    close = tokenize(u, suffix, special=True)
    full = toks + close
    gen = {"prompt": full, "n_predict": 96, "temperature": 0, "n_probs": 10, "return_tokens": True,
           "post_sampling_probs": False}
    r = post(u, "/completion", {**gen, "cache_prompt": True, "id_slot": 0})
    close_pn = r["timings"]["prompt_n"]
    one = post(u, "/completion", {**gen, "cache_prompt": False, "id_slot": 1})
    two = post(u, "/completion", {**gen, "cache_prompt": False, "id_slot": 1})

    def agree(x, y):
        x, y = x["tokens"], y["tokens"]
        n = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        return n

    def first_dist(x):
        return {t["id"]: t["logprob"] for t in x["completion_probabilities"][0]["top_logprobs"]}

    def dist_gap(x, y):                             # KL(p||q) on the shared top-10, renormalized
        import math
        p, q = first_dist(x), first_dist(y)
        k = p.keys() & q.keys()
        zp, zq = sum(math.exp(p[i]) for i in k), sum(math.exp(q[i]) for i in k)
        return round(sum(math.exp(p[i]) / zp * (p[i] - math.log(zp) - q[i] + math.log(zq)) for i in k), 5)

    whole = tokenize(u, prefix, special=True) + tokenize(u, "".join(chunks)) + close
    res = {"tag": a.tag, "chunks": len(chunks), "tokens": len(full),
           "chunk_prefill": sum(p for _, p in calls[1:]), "chunk_new": sum(n for n, _ in calls[1:]),
           "reuse_ok": ok_reuse, "close_prefill": close_pn, "close_tokens": len(close),
           "close_ok": close_pn <= len(close) + 4,
           "agree_incr_vs_oneshot": agree(r, one), "agree_oneshot_vs_oneshot": agree(one, two),
           "kl_incr": dist_gap(r, one), "kl_repeat": dist_gap(one, two), "top1_same": r["tokens"][0] == one["tokens"][0],
           "generated": len(r["tokens"]), "tokenization_differs": whole != full,
           "first_output": r["content"][:80]}
    same = res["agree_incr_vs_oneshot"] >= min(res["agree_oneshot_vs_oneshot"], 32)
    print(json.dumps(res, ensure_ascii=False))
    sys.exit(0 if ok_reuse and res["close_ok"] and same else 1)


if __name__ == "__main__":
    main()
