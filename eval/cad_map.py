"""Context-aware decoding (Shi et al. 2023) for the map step, on a subset of held-out sessions.

Each token is chosen from (1 + beta) * logp(y | prompt with transcript) - beta * logp(y | same
prompt with the transcript removed): what the model would write anyway, without reading the
window, is pushed down. Greedy, two KV caches, plain transformers -- a measurement, not the
production decoder. The window prompts are the ones the greedy run used (read from its log), and
the output directory has the run's shape so the note judge reads it unchanged.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.pipeline import parse_turn  # noqa: E402


def generate(model, tok, full, empty, beta, max_new=1024):
    enc = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt",  # noqa: E731
                                            enable_thinking=False, return_dict=False).to(model.device)
    a, b = enc(full), enc(empty)
    out_a = model(a, use_cache=True)
    out_b = model(b, use_cache=True)
    pa, pb = out_a.past_key_values, out_b.past_key_values
    la, lb = out_a.logits[:, -1], out_b.logits[:, -1]
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    ids = []
    for _ in range(max_new):
        s = (1 + beta) * torch.log_softmax(la.float(), -1) - beta * torch.log_softmax(lb.float(), -1)
        # Restrict to tokens the context-conditioned model finds plausible (the CAD paper's
        # adaptive constraint), so the contrast cannot promote garbage.
        pa_ = torch.softmax(la.float(), -1)
        s[pa_ < 0.1 * pa_.max()] = -float("inf")
        t = int(s.argmax())
        if t in stop:
            break
        ids.append(t)
        x = torch.tensor([[t]], device=model.device)
        o1 = model(x, past_key_values=pa, use_cache=True)
        o2 = model(x, past_key_values=pb, use_cache=True)
        pa, pb, la, lb = o1.past_key_values, o2.past_key_values, o1.logits[:, -1], o2.logits[:, -1]
    return tok.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="merged model dir (base+adapter)")
    ap.add_argument("--run-dir", required=True, help="greedy run of the same model, for prompts")
    ap.add_argument("--sessions", type=int, default=15)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.merged)
    model = AutoModelForCausalLM.from_pretrained(a.merged, dtype=torch.bfloat16, device_map="cuda").eval()
    os.makedirs(a.out, exist_ok=True)
    sids = sorted(f[:-5] for f in os.listdir(a.run_dir) if f.startswith("ivod_"))[:a.sessions]
    with torch.no_grad():
        for sid in sids:
            run = json.load(open(os.path.join(a.run_dir, sid + ".json"), encoding="utf-8"))
            notes = []
            for e in run["log"]:
                if not e.get("window") or e.get("retry"):
                    continue
                full = e["messages"][:2]
                empty = [full[0], {"role": "user", "content": full[1]["content"].split("TRANSCRIPT\n", 1)[0]}]
                reply = generate(model, tok, full, empty, a.beta)
                for tag, ts, t in parse_turn(reply).notes:
                    notes.append({"id": len(notes) + 1, "window": e["window"], "ts": ts, "text": t, "tag": tag})
            json.dump({"notes": notes, "windows": run["windows"]}, open(os.path.join(a.out, sid + ".json"), "w",
                                                                         encoding="utf-8"), ensure_ascii=False)
            print(sid, len(notes), "notes", flush=True)


if __name__ == "__main__":
    main()
