"""LoRA SFT of Gemma-4-E2B on the gold teacher rows, with TRL.

The base is the QAT checkpoint (gemma-4-E2B-it-qat-q4_0-unquantized): the model the phone runs is
a quantized one, and fine-tuning the quantization-aware weights keeps the adapter closest to what
gets deployed.

Rows become prompt/completion pairs, so loss falls only on what the teacher wrote -- a window's
notes, or the final summary -- and never on the transcript the student is given to read. Sessions
listed in data/split.json are held out whole, because a window from a session seen in training
would make the evaluation measure memory instead of summarising.

LoRA touches only the language model. Gemma-4-E2B is multimodal, and its vision and audio towers
carry q_proj/v_proj layers of their own that an unscoped target list would silently adapt too.
"""
import argparse
import json
import os
import random

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# The official full-precision release. The QAT checkpoint omits k_proj/v_proj/k_norm for the 20
# KV-shared layers, and vLLM's Gemma 4 loader requires them, which rules out fast vLLM rollouts
# for GRPO. Quantization happens after fine-tuning instead.
BASE = "google/gemma-4-E2B-it"
class CompletionLogitsTrainer(SFTTrainer):
    """Compute the LM head only where there is loss.

    Gemma's vocabulary has 262,144 entries. At a 6k-token context the full logits tensor is ~3 GB
    in bf16 and twice that when upcast for cross-entropy, which does not fit a 32 GB card beside
    the activations. But loss falls only on the completion -- a few hundred tokens of notes or
    summary -- so logits for the transcript positions are computed and immediately thrown away.
    logits_to_keep takes the positions directly, so the model applies its own final softcap to
    exactly the logits that matter and nothing else.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        assert labels.shape[0] == 1, "positions are gathered per sequence; keep batch size 1"
        # Logit at position t predicts token t+1.
        positions = (labels[0, 1:] != -100).nonzero().squeeze(-1)
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
                    logits_to_keep=positions, use_cache=False)
        logits = out.logits[0].float()
        loss = torch.nn.functional.cross_entropy(logits, labels[0, positions + 1], reduction="sum")
        denom = num_items_in_batch if num_items_in_batch is not None else positions.numel()
        loss = loss / denom
        return (loss, out) if return_outputs else loss


TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
# A text-only decoder (MiniCPM5 is a Llama) has no towers to exclude.
TEXT_TARGETS = r".*model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def parse_weighted_rows(specs):
    """'path.jsonl' or 'path.jsonl:0.3' -> [(path, weight), ...]. Bare paths default to weight 1."""
    out = []
    for spec in specs:
        if ":" in spec and not spec.startswith(("http:", "https:")):
            path, _, w = spec.rpartition(":")
            out.append((path, float(w)))
        else:
            out.append((spec, 1.0))
    return out


def load(rows_specs, split_path, synth_repeat=1, seed=0):
    """rows_specs: [(path, weight), ...]. A source's weight scales how much of it enters
    training: whole-number repeats plus a seeded random subsample for the fractional part (e.g.
    weight 0.3 on a source keeps one random 30% of its rows; weight 2.5 keeps it twice plus a
    random half a third time). Held-out sessions are excluded from every source the same way,
    since session ids are already domain-prefixed (ivod_/alimeeting_) and unambiguous across
    sources. Rows never get downweighted by DROPPING information outright without saying so:
    the subsample is seeded and reported (see main()'s printed per-source counts)."""
    heldout = set(json.load(open(split_path))["heldout"])
    rng = random.Random(seed)
    train, evald = [], []
    per_source_kept = {}
    for path, weight in rows_specs:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        whole, frac = int(weight), weight - int(weight)
        kept = 0
        for r in rows:
            ex = {"prompt": r["messages"][:-1], "completion": r["messages"][-1:],
                  "kind": r["kind"], "session": r["session"], "source": path}
            if r["session"] in heldout:
                evald.append(ex)
                continue
            reps = whole + (1 if rng.random() < frac else 0)
            if reps == 0:
                continue
            kept += reps
            # A session yields one synthesis row and one prose row but ~7 notes rows, so the
            # summary steps are ~10% of the data. The first run learned notes well and summaries
            # not at all (it echoed the prompt's formatting rules back as summary points).
            # Repeating synthesis and prose rows rebalances the tasks without inventing new targets.
            train.extend([ex] * reps * (synth_repeat if r["kind"] in ("synthesis", "prose") else 1))
        per_source_kept[path] = (kept, len(rows), weight)
    for path, (kept, total, weight) in per_source_kept.items():
        print(f"  {path}: {total} rows, weight {weight} -> {kept} kept for training")
    return train, evald


def _patch_apply_chat_template_return_dict_default(tok) -> None:
    """TRL 0.24.0's SFTTrainer builds the prompt-only token count with a bare
    `processing_class.apply_chat_template(..., tokenize=True)` call (no `return_dict`), written
    against transformers versions where that defaulted to `return_dict=False` (a flat token-id
    list). transformers 5.17.0 flipped that default to True, so the call now silently returns a
    BatchEncoding instead -- `len(prompt_ids)` becomes the dict's key count (~2), not the token
    count, and TRL's `completion_mask = [0]*len(prompt_ids) + [1]*(...)` ends up masking only the
    first ~2 tokens as prompt and marking almost the entire sequence -- including the transcript
    the model is supposed to just read -- as loss-bearing completion. That would train the model
    to also predict the input, not just the summary. Pin this call's default back to False."""
    import functools
    original = tok.apply_chat_template.__func__ if hasattr(tok.apply_chat_template, "__func__") \
        else type(tok).apply_chat_template

    @functools.wraps(original)
    def patched(self, *args, **kwargs):
        kwargs.setdefault("return_dict", False)
        # MiniCPM5 thinks by default; the targets are answers. With thinking off its template puts
        # the empty think block in the prompt, so the loss falls on the answer alone. Gemma ignores it.
        kwargs.setdefault("enable_thinking", False)
        return original(self, *args, **kwargs)

    tok.apply_chat_template = patched.__get__(tok, type(tok))


def main():
    # CompletionLogitsTrainer.compute_loss gathers logit positions per sequence and asserts
    # batch size 1; with >1 GPU visible, Trainer silently wraps the model in DataParallel and
    # doubles (etc.) the batch that reaches compute_loss, which used to just crash on the assert
    # -- but only once the batch happened to differ from 1, so an unlucky first few batches could
    # in principle look fine and then fail deep into a run. Fail immediately and explain instead.
    if torch.cuda.device_count() > 1:
        raise RuntimeError(
            f"{torch.cuda.device_count()} GPUs visible. This trainer assumes single-GPU "
            "(DataParallel silently multiplies the batch compute_loss sees). Run with "
            "CUDA_VISIBLE_DEVICES=0 (or whichever single index).")

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--rows", nargs="+", default=["data/train/gold_rows.jsonl"],
                    help="one or more JSONL row files, each optionally suffixed :weight "
                         "(default 1.0), e.g. --rows data/train/v2_rows.jsonl "
                         "data/train/alimeeting_rows.jsonl:0.3")
    ap.add_argument("--mix-seed", type=int, default=0,
                    help="seeds the weighted subsample when a --rows weight is fractional")
    ap.add_argument("--split", default="data/split_v2.json")
    ap.add_argument("--out", default="runs/sft/gemma4-e2b-lora")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=7168)
    ap.add_argument("--max-steps", type=int, default=-1, help="for a smoke test")
    ap.add_argument("--synth-repeat", type=int, default=1, help="repeat synthesis and prose rows N times")
    args = ap.parse_args()

    train, evald = load(parse_weighted_rows(args.rows), args.split, args.synth_repeat, args.mix_seed)
    print(f"train rows {len(train)} ({len({x['session'] for x in train})} sessions), "
          f"eval rows {len(evald)} ({len({x['session'] for x in evald})} sessions)")

    tok = AutoTokenizer.from_pretrained(args.base)
    _patch_apply_chat_template_return_dict_default(tok)
    # A row longer than max_length is truncated from the right, so its whole target can vanish. In
    # the eval set that makes the loss 0/0 = nan for the run (MiniCPM5 tokenizes one held-out
    # window to 8.6k tokens), and a nan eval loss silently disables best-checkpoint selection.
    def fits(x):
        full = tok.apply_chat_template(x["prompt"] + x["completion"], tokenize=True)
        return len(full) <= args.max_length
    long_eval = [x for x in evald if not fits(x)]
    evald = [x for x in evald if fits(x)]
    if long_eval:
        print(f"dropped {len(long_eval)} eval rows longer than {args.max_length} tokens: "
              + ", ".join(f"{x['session']}/{x['kind']}" for x in long_eval))
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, attn_implementation="sdpa")

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        eval_strategy="epoch" if args.max_steps < 0 else "no",
        save_strategy="epoch" if args.max_steps < 0 else "no",
        load_best_model_at_end=args.max_steps < 0,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        completion_only_loss=True,
        dataset_num_proc=4,
    )
    peft = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                      target_modules=TARGETS if "gemma" in args.base.lower() else TEXT_TARGETS,
                      task_type="CAUSAL_LM")
    trainer = CompletionLogitsTrainer(model=model, args=cfg, processing_class=tok, peft_config=peft,
                         train_dataset=Dataset.from_list(train),
                         eval_dataset=Dataset.from_list(evald))
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(os.path.join(args.out, "final"))
    tok.save_pretrained(os.path.join(args.out, "final"))


if __name__ == "__main__":
    main()
