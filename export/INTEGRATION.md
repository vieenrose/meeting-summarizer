# VoxSumDroid integration note — voxsum-qwen3-0.6b (fine-tuned Qwen3 0.6B)

Everything the app needs to swap in the fine-tuned summarizer. Written against
VoxSumDroid @ `c03d647` (`LiteLlmEngine`, `LlmRegistry`, `Summarizer`,
`ActionItemExtractor`, `TranscriptFormat`).

---

## 1. What this model is

Qwen3-0.6B fine-tuned on 50.6k distilled examples for **single-pass** meeting
summarization and insight extraction over VoxSum transcript-format-v1 input, in
**zh-TW and en** (plus cross-lingual). Packaged as a LiteRT-LM bundle:

| | |
|---|---|
| file | `voxsum-qwen3-0.6b_q8_ekv32768.litertlm` (664 MB, dynamic int8) |
| context | **32768** (mask-as-input; prefill signatures 8…4096) |
| template | Qwen3 ChatML embedded in the bundle → `ChatTemplate.NONE` |
| stop tokens | `<|im_end|>`, `<|endoftext|>` (embedded) |
| repo | https://huggingface.co/Luigi/voxsum-qwen3-0.6b-litertlm |

## 2. Registry entry

```kotlin
LlmSpec(
    id = "voxsum-qwen3-0.6b-litertlm",
    displayName = "VoxSum Qwen3 0.6B (meeting fine-tune)",
    url = "$HF/Luigi/voxsum-qwen3-0.6b-litertlm/resolve/7774888d3093a0ab07ef518e48284669eeac843a/voxsum-qwen3-0.6b_q8_ekv32768.litertlm",
    sha256 = "660436ac918f61f7fddffc419abf561b396585d115382efb8247814ac9382b9a",
    sizeBytes = 664_413_600L,
    fileName = "voxsum-qwen3-0.6b_q8_ekv32768.litertlm",
    chatTemplate = ChatTemplate.NONE,
    shortName = "VoxSum 0.6B",
    sampler = SamplerProfile(
        topK = 20, topP = 0.8f, temp = 0.7f,
        repeatPenalty = 1.0f, presencePenalty = 1.0f,   // presence penalty is load-bearing
    ),
)
```

`presencePenalty = 1.0` is not cosmetic: without it the model repeats bullets on long
inputs and action-item faithfulness drops ~0.6 points in eval.

## 3. Engine configuration

```kotlin
LiteLlmEngine.create(context, modelPath, sampler, nCtx = 32768, backend = "cpu")
```

`nCtx` is the only required change (`4096` → `32768`); it flows into
`maxNumTokens` in `EngineConfig`.

**RAM caveat — read before shipping.** KV cache is allocated from `nCtx`, and Qwen3-0.6B
has no sliding-window attention: a full 32k context costs roughly **3-3.7 GB** on top of
the ~700 MB of weights. That is fine on a 12 GB flagship and fatal on a 6 GB device.
Recommended: pick `nCtx` from `ActivityManager.MemoryInfo.totalMem` — e.g. 32768 above
8 GB, 16384 between 6-8 GB, 8192 below — and keep the map-reduce path (§5) as the
fallback for transcripts that exceed the chosen window.

## 4. Feed it the v1 transcript format

The model was trained on `TranscriptFormat.format()` output verbatim:

```
[0:00] S1: We need to decide the Q3 marketing budget today.
[0:05] S2: I suggest forty percent on social ads.
[1:23:45] Alice: ...
```

All three speaker variants (S-tags, real names, no-speaker) are in the training data.
Do **not** hand-build a "similar" string, and do not strip the timestamps — they are what
the model saw.

## 5. Single-pass vs map-reduce

The model is trained to summarize a whole meeting in ONE call. Sizing (measured):

| transcript | ≈ tokens | fits 32k? |
|---|---:|---|
| 1 h English | 13-15k | yes |
| 1 h zh-TW (CJK is denser) | 24-28k | yes, with little headroom |
| 2 h+ / dense council meeting | 30k+ | no |

So: count tokens (or budget ~0.6 chars/token for CJK as `Summarizer` already does); if the
prompt fits the configured `nCtx`, make one call; otherwise fall back to the existing
map-reduce path, which still works — the map/reduce prompt templates were in the training
mix too.

## 6. Prompts

Two families are supported. Both are trained; pick per feature.

**(a) Existing app templates — unchanged.** `Summarizer.MAP_TEMPLATE`,
`MAP_TEMPLATE_ZH`, `TITLE_TEMPLATE(_ZH)`, `SHRINK_TEMPLATE(_ZH)`,
`ActionItemExtractor.MAP_TEMPLATE`, all three `SummaryStyle` directives, and the
strengthened target-language clause. No app-side prompt changes needed to benefit.

**(b) NEW — one-call structured notes (`NOTES`).** Returns everything at once:

```
TITLE: Q3 Marketing Budget Decision
SUMMARY:
- Team allocates 40% of Q3 budget to social ads.
- ...
DECISIONS:
- Budget split approved: 40/30/30.
ACTIONS:
- Chen: deliver the detailed plan (due: next Friday)
- S1: chase finance for budget approval
OPEN:
- Finance approval not yet granted.
TOPICS:
- Q3 budget allocation
```

Prompt text: `distill/tasks.py::NOTES_TEMPLATE` / `NOTES_TEMPLATE_ZH` in the training
repo. Section keys are **ASCII wire format**, always all six, always in that order; the
*content* follows the transcript's language (or the target-language clause). An empty
section is exactly `-`. Parse with: new section at `^[A-Z]+:`, content until the next key;
render your own localized headers (摘要 / 決策 / 行動項目 …). Full spec:
`docs/OUTPUT-FORMAT.md`.

Measured compliance: **100 %** in the transcript's own language, 94 % cross-lingual
(n = 96 held-out meetings). Budget ~640 output tokens.

## 7. Post-processing — keep it

`SummaryText.stripThink` is still required: the model emits an empty `<think></think>`
block before its answer. `cleanSummary`, `tooLong`→shrink, and the sentence-dedup backstop
should also stay — the format is high-probability trained behaviour, not a grammar.
For NOTES, validate the six keys before rendering and fall back to the plain-summary path
if parsing fails.

## 8. Behaviour deltas vs the current Gemma default

Improvements (held-out eval, teacher-judged):

- cross-lingual **output-language compliance 0.38 → 0.98** — the base model silently
  answered in the transcript's language 62 % of the time when a target language was set
- action items coverage 1.90 → 3.94 (of 5), open questions 2.90 → 4.10
- executive summary coverage 3.58 → 3.88 (faithfulness 3.65 → 3.48, roughly par)

Known limits, please surface honestly in the UI:

- **cross-lingual faithfulness is the weak axis** (≈2.5/5 on hour-long en→zh-TW): the
  language is right, but details can drift. Native-language summaries score ≈3.2-3.9.
- action items and decisions remain a *draft to correct*, never an audit trail.
- 0.6B on CPU: expect multi-minute prefill for a full hour-long transcript. Show progress
  and allow cancel; consider the GPU backend where available.

## 9. Suggested rollout

1. Add the `LlmSpec` alongside the Gemma entries (do not remove them) — users can A/B.
2. Ship device-tier `nCtx` selection + single-pass-if-it-fits.
3. Add the NOTES call behind the existing "insights" surface; keep the per-task prompts
   as the fallback.
4. Measure on-device prefill/decode and peak RSS per tier before making it the default.
