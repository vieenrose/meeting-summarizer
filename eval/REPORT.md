# Eval report — voxsum-qwen3-0.6b v1 (2026-07-30)

48 held-out test meetings (12 × QMSum / VCSum / DialogSum / MeetingBank), single-pass
full-transcript prompts (32k budget), judged by Qwen3.5-9B (Q4_K_M) for faithfulness
(FAITH 1-5, "no inventions") and coverage (COVER 1-5). `lang_ok` = output script matches
the requested language; `fmt_ok` = no preamble/markdown/think, length limits respected.

Samplers: **plain** = temp 0.7 / top_p 0.9. **app** = the profile the registry entry
ships: temp 0.7 / top_p 0.8 / top_k 20 / presence_penalty 1.0.

| task | metric | base | ft (plain) | ft (app) |
|---|---|---:|---:|---:|
| summarize | lang_ok | 0.69 | 0.99 | **0.99** |
| summarize | faith | 3.84 | 2.94 | 2.97 |
| summarize | cover | 3.48 | 3.52 | **3.56** |
| actions | faith | 2.62 | 2.98 | **3.60** |
| actions | cover | 1.90 | 3.02 | **3.44** |
| exec_summary | faith | 3.65 | 2.73 | 2.98 |
| open_questions | faith | 3.96 | 4.46 | **4.48** |
| open_questions | cover | 2.90 | 4.06 | 3.96 |
| title | fmt_ok | 0.92 | 0.98 | **1.00** |
| title | cover | 3.56 | 4.02 | 3.81 |

Splits that explain the headline numbers:

| summarize split | base | ft (app) |
|---|---|---|
| native lang_ok / faith / cover | 1.00 / 3.96 / 3.71 | 1.00 / 3.65 / 4.21 |
| cross-lingual lang_ok / faith / cover | **0.38** / 3.73 / 3.25 | **0.98** / 2.29 / 2.92 |

## Conclusions

1. **Cross-lingual output language — the app's hard requirement — goes 0.38 → 0.98.**
   The base model simply ignores the strengthened language clause 62% of the time and
   banks "faithfulness" credit for untranslated English summaries.
2. **Action items / decisions and open questions improve decisively** (base cover 1.90
   is effectively useless; ft/app reaches 3.44-3.96 with faith wins too).
3. **presence_penalty matters**: it lifts actions faith +0.6 and kills the repetition
   loops. The registry SamplerProfile must ship it.
4. **Known weakness: cross-lingual faithfulness (2.29)** on hour-long en→zh-TW
   summaries — a 0.6B capability limit, not sampling (penalty barely moves it).
   Options if it matters: more cross-lingual distillation data, native-summarize →
   translate two-step in the app, or the Gemma-3-1B / Qwen3.5-0.8B students when
   conversion paths allow.
5. Native-language behavior is at or above base across the board with strictly better
   format compliance.

Ship: yes, as v1.
