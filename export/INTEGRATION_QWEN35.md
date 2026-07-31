# VoxSumDroid integration note — voxsum-qwen35-0.8b

Successor to `voxsum-qwen3-0.6b-litertlm`. Same input/output contracts; different base
model, chosen on measurements rather than availability.

Model: https://huggingface.co/Luigi/voxsum-qwen35-0.8b

---

## 1. What changed and why it matters to the app

| | qwen3-0.6b (shipped) | **qwen35-0.8b (this)** |
|---|---|---|
| architecture | 28 layers, all full attention | 24 layers, **6 attention + 18 linear** |
| KV @32k | 3,760 MB (fp16) | **197 MB (q4)** — 19× smaller |
| zh-TW tokenizer | 1.55 chars/token | **1.67** |
| prefill @16k, 8 threads | — | **491 tok/s** (fastest of 9 models tested) |
| training data | unfiltered teacher output | **faithfulness-filtered** (16% dropped) |

The KV reduction is the headline: a 3-hour zh-TW meeting now fits in **~1.2 GB total RSS**
instead of being impossible.

## 2. Deployment status — read this first

There is **no `.litertlm` for this model**. `ai-edge-torch` has no converter for the
`qwen3_5` architecture (hybrid Gated-DeltaNet linear attention), and LiteRT-LM's generative
API is documented as transformer-only. Options:

- **llama.cpp / GGUF** — works today; `Qwen3.5-0.8B` GGUFs load and I benchmarked this
  architecture at 491 tok/s prefill. This means reintroducing a llama.cpp path that the app
  removed in 2026-07.
- **Wait for LiteRT support** of `qwen3_5` — not on any published roadmap.
- **Convert in the VoxSum project** using the litert fork, if the DeltaNet scan can be
  expressed as a custom op (comparable effort to the TQ3 port).

If a `.litertlm` is mandatory and cannot wait, the previously shipped
`voxsum-qwen3-0.6b-litertlm` remains the only converted artifact — at the cost of 3.76 GB
KV at 32k and the faithfulness numbers this model improves on.

## 3. Runtime configuration

Quantization: **q4 weights (int4-block32), int8 output head** — the model was QAT-trained
under exactly this, so post-training quantization to int4 should lose little.

```
llama-server -m voxsum-qwen35-0.8b-Q4_K_M.gguf \
  -c 32768 -fa on -ctk q4_0 -ctv q4_0 \
  --temp 0.7 --top-p 0.8 --top-k 20
```

Memory, by target context:

| context | covers | RSS |
|---|---|---|
| 16k | ~1 h zh-TW / ~2 h en | ~1.0 GB |
| **32k** | **~3 h zh-TW** | **~1.2 GB** |
| 64k | ~4 h en | ~1.4 GB |

## 4. Feed it transcript-format v1, unchanged

`TranscriptFormat.format()` output verbatim — `[M:SS] S1: text`, names verbatim,
no-speaker variant. All three speaker styles are in the training data.

## 5. Chunking policy — the important behavioural note

Faithfulness is **length-dependent**, measured:

| transcript tokens | faith (1-5) | inversion rate |
|---|---|---|
| 2k–6k | 4.29 | **1%** |
| 6k–12k | 4.04 | 4% |
| **12k–20k** | **3.34** | **25%** |

So: **chunk at ~10–12k tokens even though 32k fits.** A chunk that fits is not the same as
a chunk the model summarizes faithfully. The map/reduce prompt templates remain in the
training mix, and reduce-stage inputs (short partial summaries) land in the model's best
regime. Single-pass is appropriate for meetings under ~45 min en / ~20 min zh-TW.

## 6. Prompts and output

Unchanged from the v2 spec: the app's existing templates work, plus the single-call
`NOTES` format (`docs/OUTPUT-FORMAT.md`). Verified output on a zh-TW transcript:

```
TITLE: 第三季行銷預算決定
SUMMARY:
- 決定社群廣告佔百分之四十為核心預算。
- 分配三成 KOL 合作，三成用於檔期促銷。
DECISIONS:
- 社群廣告預算百分之四十。
ACTIONS:
- 小陳：下週五前提供細部計畫。
OPEN:
- 財務預算追補進度。
TOPICS:
- 第三季行銷預算分配。
```

`SummaryText.stripThink` is still required (the model emits an empty `<think></think>`),
as are `cleanSummary` and the `tooLong`→shrink backstop.

## 7. Honest limitations to surface in UI

- **Owner attribution slips on long transcripts.** In the sample above the model assigned
  a finance follow-up to 小陳 when the transcript credits S3. Action items are a draft to
  correct.
- Faithfulness past 12k tokens degrades as tabled in §5 — mitigate by chunking.
- Cross-lingual (en→zh-TW) detail fidelity is the weakest axis; the language routing itself
  is reliable.
- Prefill on ARM is the real cost: ~4 min for a 3-hour zh-TW meeting at 8 threads
  (extrapolated from 491 tok/s on desktop, derated 4×). Show progress; allow cancel.
