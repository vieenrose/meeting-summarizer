---
license: apache-2.0
language: [zh, en]
base_model: Qwen/Qwen3-0.6B
tags: [litert-lm, litertlm, on-device, summarization, meeting, zh-tw, voxsum]
pipeline_tag: summarization
---

# voxsum-qwen3-0.6b — on-device meeting summarizer (zh-TW / en)

Qwen3-0.6B fine-tuned for **single-pass meeting summarization and insights** over
[VoxSumDroid](https://github.com/vieenrose/VoxSumDroid) transcripts, packaged as a
LiteRT-LM `.litertlm` bundle (dynamic int8, **32k context**, mask-as-input, Qwen3 ChatML
template + stop tokens embedded).

## Input format (transcript format v1)

One utterance per line, start-timestamp + speaker tag:

```
[0:00] S1: 我們今天要決定第三季的行銷預算。
[0:05] S2: 我建議把百分之四十放在社群廣告，效果最好。
[1:23:45] Alice: ...
```

Variants supported: `S1…Sn` tags, real names, or no speaker field (no diarization).

## Tasks

Summaries (bullet / executive / narrative), title (≤8 words), action items + decisions,
detailed summary, topics, open questions, risks/disagreements — in **zh-TW and en**,
including cross-lingual output via the VoxSumDroid language clause ("Write the ENTIRE
output in …"). Prompts must follow the VoxSumDroid templates the model was trained on
(see the [training repo](https://github.com/vieenrose/meeting-summarizer)).

## Training

Distilled from Qwen3.5-9B over 3,891 meetings (QMSum, MeetingBank, DialogSum,
VCSum→OpenCC s2twp; 60k filtered prompt/completion pairs), full-parameter SFT,
seq 32768 packed, 2 epochs. Details + eval: training repo `eval/REPORT.md`.

## Eval vs base (held-out meetings, teacher-as-judge)

| metric | base | fine-tune |
|---|---|---|
| cross-lingual output-language compliance | 0.38 | **0.98** |
| action items coverage (1-5) | 1.90 | **3.44** |
| open questions faith / cover | 3.96 / 2.90 | **4.48 / 3.96** |
| known limit: en→zh-TW hour-long faithfulness | 3.73* | 2.29 |

\* base scores "faithful" largely by *not translating* (wrong language 62% of the time).

**Recommended sampler:** temp 0.7, top_p 0.8, top_k 20, **presence_penalty 1.0**.

## Files

- `voxsum-qwen3-0.6b_q8_ekv32768.litertlm` — 664MB, kv 32768
  sha256 `5fa8e11fd9955c39512a25a24c686f42483ebb7059ba30f1d3c218bacdcd5b98`

RAM note: full 32k KV ≈ 3-3.7GB fp16 on-device — cap `nCtx` lower on small devices.
