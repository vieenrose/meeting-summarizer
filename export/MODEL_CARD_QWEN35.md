---
license: apache-2.0
language: [zh, en]
base_model: Qwen/Qwen3.5-0.8B
tags: [on-device, summarization, meeting, zh-tw, voxsum, qat, long-context]
pipeline_tag: summarization
---

# voxsum-qwen35-0.8b — on-device meeting summarizer (zh-TW / en)

Qwen3.5-0.8B (text) fine-tuned for **long meeting-transcript summarization and insight
extraction** in Traditional Chinese and English, for
[VoxSumDroid](https://github.com/vieenrose/VoxSumDroid). Trained with **quantization-aware
training** so it survives int4 weights on device.

## Why this base model

Nine candidates were benchmarked on the three constraints that actually bind on a phone
(prefill speed, Chinese token efficiency, RSS). Qwen3.5-0.8B won all three:

| | Qwen3.5-0.8B | best rival |
|---|---|---|
| prefill @16k, 8 CPU threads | **491 tok/s** | 457 (granite-4.0-h-1b) |
| zh-TW tokenizer | **1.67 chars/token** | 0.98 (Falcon-H1), 0.69 (Granite) |
| 3-h zh meeting, prefill (ARM-derated) | **~4 min** | 9–12 min |
| KV cache @32k, q4 | 197 MB | — |

Its hybrid architecture (only **6 of 24 layers** hold a KV cache; the other 18 are
Gated-DeltaNet linear attention with constant state) is what keeps memory and prefill low
at long context. Rejected: Falcon-H1-1.5B (1.84 GB @64k, slowest prefill),
Granite-4.0-h-1b (0.69 chars/token on zh), Granite-3.1-1b-a400m MoE (−54% prefill decay by
16k), T5Gemma-2 enc-dec (no runtime support, 32k ceiling, degenerate at 270M),
Gemma-3-1B (32k native), Qwen3.5-2B (1.68 GB, over budget).

## Memory (measured components + measured runtime overhead)

| context | weights q4 | KV q4 | total RSS |
|---|---|---|---|
| 32k (≈3-h zh-TW meeting) | ~600 MB | 197 MB | **~1.2 GB** |
| 64k | ~600 MB | 393 MB | ~1.4 GB |

## Input format

VoxSumDroid transcript format v1 — one utterance per line:

```
[0:00] S1: 我們今天要決定第三季的行銷預算。
[1:23:45] Alice: ...
```

## Output format

Single-call structured notes (`docs/OUTPUT-FORMAT.md` v2) — ASCII section keys, content in
the transcript's language:

```
TITLE: 第三季行銷預算決定
SUMMARY:
- 決定社群廣告佔百分之四十為核心預算。
DECISIONS:
- 社群廣告預算百分之四十。
ACTIONS:
- 小陳：下週五前提供細部計畫。
OPEN:
- 財務預算追補進度。
TOPICS:
- 第三季行銷預算分配。
```

The app's per-task prompts (summarize / title / action items / topics / open questions /
risks, bullet-executive-narrative styles, en+zh with cross-lingual clause) are also
supported — see the training repo.

## Training

1. **Distillation** from Qwen3.5-9B over 3,891 meetings (QMSum, MeetingBank, DialogSum,
   VCSum→OpenCC s2twp).
2. **Faithfulness filtering** — every one of 39,731 targets judged for factual support and
   polarity inversion; **6,414 dropped** (16.1% scored faith≤3; 300 were outright
   inversions). This was the single largest quality lever.
3. **QAT SFT** — int4-block32 weight fake-quant (output head int8), 32k packed sequences.
4. **Rejection-sampling fine-tune** on the model's own long-document outputs: 389 samples
   the judge scored faith 4.73 with zero inversions, upweighted 6×. Their rejected
   counterparts averaged faith 1.85 and contained 121 inversions.

## Known limitations — please surface these honestly in UI

- **Faithfulness degrades past ~12k tokens.** Measured on the shipped checkpoint:
  faith 4.05 / 1.8% inversion at 4–12k tokens, versus 3.36 / 15.3% inversion above 12k,
  where 46% of outputs score faith≤2.
  Chunking at ~10–12k tokens with a hierarchical reduce keeps the model in its reliable
  regime; the map/reduce prompts are in the training mix for exactly this.
- **Owner attribution can slip** on long transcripts (assigning an action to the wrong
  speaker). Action items are a draft to correct, never an audit trail.
- Cross-lingual (en→zh-TW) faithfulness is the weakest axis; language *routing* is
  reliable (0.98) but details drift.

**Sampler:** temp 0.7, top_p 0.8, top_k 20, presence_penalty 1.0. Near-greedy
(temp 0–0.3) measurably improves faithfulness for extraction tasks.
