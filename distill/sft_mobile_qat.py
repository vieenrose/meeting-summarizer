"""LoRA SFT of Gemma-4-E2B *mobile* (Google's QAT mobile checkpoint: int2/int4/int8 weights with
per-channel scales, int8 static-range activations) so that the result can be written back on the
same grids and run by LiteRT-LM or llama.cpp with the mobile footprint.

Quantization-aware: every targeted QuantizedLinear computes with
    W = scale * clamp(round_ste((int * scale + s * B A) / scale), qmin, qmax)
i.e. the fine-tuned weight is fake-quantized on Google's own per-channel scale and bit width, and
the gradient goes straight through the rounding. Activations keep Google's static int8 ranges,
also with a straight-through rounding (the stock torch.round has a zero gradient, which would stop
backpropagation at the first quantized layer). After training, distill/export_mobile.py writes the
new integers into the same packed format: same scales, same bits, no requantization error.

Data and loss are distill/sft_agent.py's (same rows, same loss turns, same data-parallel loop).
"""
import argparse
import math
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.integrations import gemma_quant as gq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.sft_agent import encode, loss_of  # noqa: E402

BASE = "unsloth/gemma-4-E2B-it-qat-mobile"
TARGETS = r"language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"


def round_ste(x):
    return x + (torch.round(x) - x).detach()


def srq_ste(x, scale, bits=8):
    """gemma_quant.apply_srq with a straight-through rounding."""
    scale = scale.to(x.dtype)
    hi = 2 ** (bits - 1) - 1
    if float(scale) == 0.0:
        return x
    return torch.clamp(round_ste(x / scale), -hi - 1, hi) * scale


class QATLoRA(nn.Module):
    """Wraps a gemma_quant.QuantizedLinear: frozen integer weights, trainable LoRA, fake-quant."""

    weight_fq = True    # class switch: stage 1 trains without weight rounding, stage 2 with it

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.bits = base.num_bits
        self.qmin, self.qmax = -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1
        self.scaling = alpha / rank
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.randn(rank, base.in_features, device=dev) * (1 / math.sqrt(base.in_features)))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev))

    def int_weights(self):
        b = self.base
        if self.bits == 2:
            return gq._unpack_int2(b.weight, b.in_features)
        if self.bits == 4:
            return gq._unpack_int4(b.weight, b.in_features)
        return b.weight

    def merged_int(self):
        """The integers of the fine-tuned weight on the base's grid (for export)."""
        with torch.no_grad():
            s = self.base.weight_scale.float()
            w = self.int_weights().float() + (self.scaling * self.lora_B.float() @ self.lora_A.float()) / s
            return torch.clamp(torch.round(w), self.qmin, self.qmax).to(torch.int8)

    def forward(self, x):
        b = self.base
        s = b.weight_scale.to(x.dtype)
        w = self.int_weights().to(x.dtype) + (self.scaling * self.lora_B.to(x.dtype) @ self.lora_A.to(x.dtype)) / s
        wq = (torch.clamp(round_ste(w), self.qmin, self.qmax) if QATLoRA.weight_fq else w) * s
        x = srq_ste(x, b.input_activation_scale)
        out = F.linear(x, wq, b.bias)
        return srq_ste(out, b.output_activation_scale)


def kd_loss_of(model, ids, labels, item, w):
    """(1 - w) * CE on the gold tokens + w * CE on the teacher's renormalized top-k distribution."""
    ids, labels = ids.unsqueeze(0).cuda(), labels.cuda()
    pos = (labels[1:] != -100).nonzero().squeeze(-1)
    assert torch.equal(pos.cpu(), item["pos"].long()), "KD positions do not match the rows"
    lq = torch.log_softmax(model(input_ids=ids, logits_to_keep=pos, use_cache=False).logits[0].float(), -1)
    ce = F.nll_loss(lq, labels[pos + 1], reduction="sum")
    tp = torch.softmax(item["lp"].float().cuda(), -1)
    soft = -(tp * lq.gather(1, item["ids"].long().cuda())).sum()
    return (1 - w) * ce + w * soft, pos.numel()


def wrap(model, rank, alpha):
    import re
    pat = re.compile(TARGETS)
    wrapped = {}
    for name, mod in list(model.named_modules()):
        if pat.search(name) and isinstance(mod, gq.QuantizedLinear):
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            q = QATLoRA(mod, rank, alpha)
            setattr(parent, name.rsplit(".", 1)[1], q)
            wrapped[name] = q
    for p in model.parameters():
        p.requires_grad_(False)
    for q in wrapped.values():
        q.lora_A.requires_grad_(True)
        q.lora_B.requires_grad_(True)
    return wrapped


def save(wrapped, path, rank, alpha):
    os.makedirs(path, exist_ok=True)
    t = {}
    for n, q in wrapped.items():
        t[n + ".lora_A"] = q.lora_A.detach().float().cpu().contiguous()
        t[n + ".lora_B"] = q.lora_B.detach().float().cpu().contiguous()
    save_file(t, os.path.join(path, "qat_lora.safetensors"), metadata={"rank": str(rank), "alpha": str(alpha)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="data/train/agent_sft_rows_v6.jsonl")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default="runs/sft/mobile/lora")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--val-sessions", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-rows", type=int, default=-1, help="smoke test")
    ap.add_argument("--weight-fq", choices=["on", "off"], default="on",
                    help="off: stage 1, the LoRA delta is added in float (a delta below half a quantization "
                         "step would otherwise never change an integer, and the loss could not move); "
                         "on: stage 2, fake-quantized on the base's grid")
    ap.add_argument("--init-lora", help="start from a qat_lora.safetensors (stage 2)")
    ap.add_argument("--kd", help="teacher top-k file from distill/kd_teacher_logits.py (same rows)")
    ap.add_argument("--kd-weight", type=float, default=0.7)
    a = ap.parse_args()
    import json
    import torch.distributed as dist
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if world > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    random.seed(0)
    torch.manual_seed(0)
    rows = [json.loads(l) for l in open(a.rows, encoding="utf-8")]
    if a.max_rows > 0:
        rows = rows[:a.max_rows]
    sessions = sorted({r["session"] for r in rows})
    val_s = set(random.Random(0).sample(sessions, min(a.val_sessions, max(1, len(sessions) // 4))))
    tok = AutoTokenizer.from_pretrained(a.base)
    kd = None
    if a.kd:
        kd = torch.load(a.kd)["items"]
        assert len(kd) >= len(rows)
    data = [(r["session"], *encode(tok, r), j) for j, r in enumerate(rows)]
    train = [(i, l, j) for s, i, l, j in data if s not in val_s]
    val = [(i, l) for s, i, l, j in data if s in val_s]
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).cuda()
    wrapped = wrap(model, a.rank, 2 * a.rank)
    QATLoRA.weight_fq = a.weight_fq == "on"
    if a.init_lora:
        from safetensors.torch import load_file
        lo = load_file(a.init_lora)
        for n, q in wrapped.items():
            q.lora_A.data.copy_(lo[n + ".lora_A"].to(q.lora_A.device))
            q.lora_B.data.copy_(lo[n + ".lora_B"].to(q.lora_B.device))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if world > 1:
        dist.init_process_group("nccl")
    params = [p for q in wrapped.values() for p in (q.lora_A, q.lora_B)]
    if rank == 0:
        print(f"train segments {len(train)}, val {len(val)}, wrapped {len(wrapped)} layers, "
              f"trainable {sum(p.numel() for p in params) / 1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(train) / a.accum) * a.epochs if a.max_steps < 0 else a.max_steps
    sched = get_cosine_schedule_with_warmup(opt, min(20, max(1, steps // 10)), steps)

    def evaluate():
        model.eval()
        tot = n = 0
        with torch.no_grad():
            for ids, labels in val[rank::world]:
                l, k = loss_of(model, ids, labels)
                tot, n = tot + l.item(), n + k
        if world > 1:
            t = torch.tensor([tot, n], device="cuda", dtype=torch.float64)
            dist.all_reduce(t)
            tot, n = t[0].item(), t[1].item()
        model.train()
        return tot / max(1, n)

    v0 = evaluate()
    if rank == 0:
        print(f"step 0 val loss {v0:.4f}", flush=True)
    step = 0
    model.train()
    for ep in range(a.epochs):
        order = list(range(len(train)))
        random.shuffle(order)
        for b in range(0, len(order), a.accum):
            batch = [train[j] for j in order[b:b + a.accum]]
            denom = sum((l != -100).sum().item() for _, l, _ in batch)
            run = 0.0
            for ids, labels, j in batch[rank::world]:
                l, _ = kd_loss_of(model, ids, labels, kd[j], a.kd_weight) if kd else loss_of(model, ids, labels)
                (l / denom).backward()
                run += l.item()
            if world > 1:
                for p in params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad)
                t = torch.tensor([run], device="cuda", dtype=torch.float64)
                dist.all_reduce(t)
                run = t.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if rank == 0 and step % 5 == 0:
                changed = sum(int((q.merged_int() != q.int_weights().to(torch.int8)).sum()) for q in list(wrapped.values())[:4])
                print(f"ep {ep} step {step}/{steps} loss {run / denom:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"ints changed (4 layers) {changed}", flush=True)
            if 0 < a.max_steps <= step:
                break
        v = evaluate()
        if rank == 0:
            print(f"epoch {ep} val loss {v:.4f}", flush=True)
            save(wrapped, os.path.join(a.out, f"epoch{ep}"), a.rank, 2 * a.rank)
        if 0 < a.max_steps <= step:
            break
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
