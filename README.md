# VoxSum Meeting Summarizer — sub-1B fine-tune

Fine-tune of a sub-1B decoder SLM for **zh-TW + en meeting transcripts** that produces
summaries **and insights** (action items, decisions, topics, open questions), for on-device
use in [VoxSumDroid](https://github.com/vieenrose/VoxSumDroid).

## Locked decisions

| Aspect | Decision | Why |
|---|---|---|
| Target runtime | LiteRT-LM `.litertlm` bundle | app removed llama.cpp/ONNX 2026-07; `LiteLlmEngine` nCtx=4096, CPU default |
| Base model | `Qwen/Qwen3-0.6B` | app's former default; proven conversion (`litert-community/Qwen3-0.6B`); ai-edge-torch has `examples/qwen/convert_v3_to_tflite.py` (qwen3); Apache-2.0; strong zh-TW/en. Qwen3.5-0.8B is stronger but arch `qwen3_5` has no converter yet |
| Teacher | `lovedheart/Qwen3.5-9B-FP8` via vLLM, **GPU 1 only** | GPU 0 reserved for other work; ~10GB fits one 5090 with batch headroom |
| Training | TRL SFT, full-parameter, bf16, seq 4096, completions-only loss, GPU 1 | 0.6B trains easily on one 5090 |
| Input | **Transcript format v1** (`VoxSumDroid/docs/TRANSCRIPT-FORMAT.md`): `[M:SS] S1: text`, names verbatim, no-diarization variant | single implementation point: `configs/transcript_format.py`, mirrors `TranscriptFormat.kt` |
| Eval | validate_llm.py-style script checks + teacher-as-judge faithfulness | mirrors the app's existing harness |

## Prior art: LiquidAI LFM2-2.6B-Transcript

Metadata header (title/date/duration/participants) + `**Speaker**: text` lines, fixed task
prompts (executive summary, detailed summary, action items, key decisions, participants,
topics), temp 0.3. English-only, 2.6B, 32k ctx. We borrow the fixed-task-prompt design;
we differ: sub-1B, bilingual zh-TW/en, cross-lingual output, timestamps, 4k ctx (hierarchical
map-reduce preserved), title generation, VoxSumDroid prompt-template compatibility.

## Architecture: single-pass (v2, 2026-07-29)

Map-reduce is REMOVED (user decision): the model receives the whole transcript in one
call. Context target = **32k** (Qwen3-0.6B native max; zh-TW hour ≈ 26k tokens, en hour
≈ 13-15k). Training seq len 32768 (liger fused CE), export kv_cache_max_len=32768.
Meetings that exceed the budget are skipped in training; runtime over-length handling is
the app's concern. Run-1 chunked records are kept as short-transcript auxiliaries
("reduce"-phase records are dropped at SFT build). On-device cost note: fp16 KV at 26k
tokens ≈ 3GB — GPU backend / int8 KV / a lower app-side cap may be needed on weak devices.

## Task suite (what the model is trained to do)

Two prompt families, both trained:

**A. VoxSumDroid-compat** (templates verbatim from `Summarizer.kt` / `ActionItemExtractor.kt`):
map summary (bullet / executive / narrative styles), hierarchical reduce, shrink,
title (≤8 words), action items + decisions; en instruction + `*_ZH` variants; strengthened
target-language clause for cross-lingual (en↔zh-TW; "same language as transcript" default).

**B. Insight tasks** (LFM2-style fixed prompts, en + zh-TW): executive summary, detailed
summary, action items (owner + deadline), key decisions, main topics, participants,
open questions / follow-ups, notable disagreements & risks.

## Data

- **en**: MeetingBank, QMSum, AMI, ICSI, DialogSum
- **zh**: VCSum (→ OpenCC s2tw), + synthetic zh-TW meetings from the teacher
- All rendered into the VoxSum transcript format (timestamps synthesized from utterance
  length at realistic speaking rates; speaker labels: generic `Speaker N`/`語者 N` and
  real-name variants) — rendering blocked on the format scheme.
- Distillation: teacher answers the full task suite per (meeting-chunk × task × lang-target).

## Layout

```
configs/    transcript_format.py (TBD), training + gen configs
data/       raw/ (downloaded corpora) → voxsum_format/ (rendered) → sft/ (final ChatML)
distill/    vLLM teacher server + generation scripts
train/      TRL SFT
eval/       validate + faithfulness + report
export/     ai-edge-torch → .litertlm + registry entry
VoxSumDroid/ app source (reference)
ai-edge-torch/ converter source (reference)
```
