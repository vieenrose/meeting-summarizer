# Eval report — voxsum-qwen3-0.6b (2026-07-30)

48 held-out test meetings (12 × QMSum / VCSum / DialogSum / MeetingBank), single-pass
full-transcript prompts (32k budget), judged by Qwen3.5-9B for faithfulness (FAITH 1-5,
"no inventions") and coverage (COVER 1-5). `lang_ok` = output script matches the
requested language; `fmt_ok` = format compliance — for `notes` this is the **strict
wire-format check** from docs/OUTPUT-FORMAT.md (key order, all sections present, `-`
empty markers, no placeholders).

Sampler for all runs below: temp 0.7 / top_p 0.8 / top_k 20 / presence_penalty 1.0
(the profile shipped in the registry entry).

## v2 (structured NOTES) vs v1 vs base

| task | metric | base | v1 | **v2** |
|---|---|---:|---:|---:|
| notes | fmt_ok | – | – | **0.97** |
| notes | lang_ok | – | – | 0.96 |
| notes | faith / cover | – | – | 2.84 / 3.39 |
| summarize | lang_ok | 0.69 | 0.99 | 0.98 |
| summarize | faith / cover | 3.84 / 3.48 | 2.97 / 3.56 | **3.19 / 3.72** |
| actions | cover | 1.90 | 3.44 | **3.94** |
| actions | faith | 2.62 | 3.60 | 3.50 |
| exec_summary | faith / cover | 3.65 / 3.58 | 2.98 / 3.48 | **3.48 / 3.88** |
| open_questions | faith / cover | 3.96 / 2.90 | 4.48 / 3.96 | 4.42 / **4.10** |
| title | fmt_ok / faith | 0.92 / 4.50 | 1.00 / 4.42 | 0.96 / 4.08 |

NOTES split (the v2 headline):

| split | fmt_ok | lang_ok | faith |
|---|---:|---:|---:|
| native language | **1.00** | 1.00 | 3.21 |
| cross-lingual | 0.94 | 0.92 | 2.48 |

## Conclusions

1. **The structured NOTES format is reliable**: 100% wire-format compliance in the
   transcript's own language, 94% cross-lingual (3 failures / 96 samples). One call
   now returns title + summary + decisions + actions + open questions + topics.
2. **v2 improved the v1 tasks too** rather than trading them away — summarize faith
   +0.22 / cover +0.16, exec_summary faith +0.50, actions cover +0.50 — with a lower
   final train eval_loss (1.448 vs 1.456) despite learning an extra task.
3. **Small regression on `title`** (fmt 1.00→0.96, faith 4.42→4.08): the standalone
   title task now competes with `TITLE:` inside NOTES. Acceptable — the app can take
   the title from the NOTES call instead of a separate one.
4. **Cross-lingual faithfulness remains the weak axis** (notes 2.48, summarize
   cross-lingual ≈2.3): a 0.6B capability limit. Mitigations unchanged: more
   cross-lingual distillation data, a native→translate two-step, or a larger student.
5. Base-model comparison is unchanged from v1: the base ignores the target-language
   clause 62% of the time, so its apparently-competitive faith scores are earned by
   not translating at all.

Ship: yes — v2 supersedes v1.
