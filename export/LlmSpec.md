# VoxSumDroid registry entry — voxsum-qwen3-0.6b v1

Add to `LlmRegistry.ALL` (uploaded: https://huggingface.co/Luigi/voxsum-qwen3-0.6b-litertlm, revision pinned):

```kotlin
LlmSpec(
    id = "voxsum-qwen3-0.6b-litertlm",
    displayName = "VoxSum Qwen3 0.6B (meeting fine-tune)",
    url = "$HF/Luigi/voxsum-qwen3-0.6b-litertlm/resolve/c0740a19d736c0cf4a2a9ff6082a5b3718d75ea2/voxsum-qwen3-0.6b_q8_ekv32768.litertlm",
    sha256 = "5fa8e11fd9955c39512a25a24c686f42483ebb7059ba30f1d3c218bacdcd5b98",
    sizeBytes = 664_413_600L,
    fileName = "voxsum-qwen3-0.6b_q8_ekv32768.litertlm",
    chatTemplate = ChatTemplate.NONE,   // bundle carries qwen3 template metadata
    shortName = "VoxSum 0.6B",
    sampler = SamplerProfile(           // eval-validated; presence penalty is load-bearing
        topK = 20, topP = 0.8f, temp = 0.7f,
        repeatPenalty = 1.0f, presencePenalty = 1.0f,
    ),
)
```

App-side notes for the single-pass integration:

- Engine init must use `nCtx = 32768` (`LiteLlmEngine.create(..., nCtx = 32768)`).
  KV at full 32k ≈ 3-3.7GB fp16 — consider capping nCtx lower on low-RAM devices
  (the graph was exported with `mask_as_input`, prefill signatures 8..4096).
- Feed the model the **v1 transcript format** (`TranscriptFormat.format()`) in ONE call —
  the model was trained single-pass; the old map-reduce prompts still work but are no
  longer required. Trained prompt families: Summarizer MAP/TITLE/SHRINK (+ `*_ZH`),
  ActionItemExtractor, and the 7 insight tasks (exec/detailed summary, action items,
  decisions, topics, open questions, risks) in en + zh-TW with the strengthened
  target-language clause.
- Model answers directly (no `<think>` block); `SummaryText.stripThink` stays harmless.
- Known limit: hour-long en→zh-TW cross-lingual summaries score 2.29/5 faithfulness
  (language routing itself is 0.98) — see eval/REPORT.md for mitigations.
