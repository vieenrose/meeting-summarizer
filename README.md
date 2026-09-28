# meeting-summarizer

Reliable minutes for long meetings, produced **on the phone**.

The input is a 1.5–3 h zh-TW meeting transcribed by on-device ASR. The output is structured minutes in which every item cites the transcript line it rests on.

**Device budget:** peak RSS ≤ 6 GB, and every model call fits a context of ≤ 32k tokens.

## Approach

A **reading agent with an external journal**, driven by **Bonsai 2 27B**: the ternary Qwen3.8-27B from PrismML, packed as `PTQ1_0`, 5.95 GB. The model is used as is, with no fine-tuning. Reliability comes from the harness.

```
transcript ──► window 1 ──► window 2 ──► … ──► window N
                  │            │                   │
                  ▼            ▼                   ▼
           ┌──────────────── journal (outside the context) ───────────────┐
           │ #1 [12:04] (DECISION) …   #2 [15:30] …   #7 revised in w5 …  │
           └──────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
            minutes: decisions · actions & owners · open items · overview
                                      │
                                      ▼
                  each item re-checked against its cited lines
```

### One reading turn

Each turn sees three things:

- a **bounded view of the journal**: recent entries, entries sharing keywords with the current window, and a count of the rest;
- the current transcript window (~6k tokens);
- lines re-read with `LOOKBACK`, if the turn asked for any.

The model answers with actions, one per line:

| action | effect |
|---|---|
| `NOTE [ts] (TYPE) text` | add a journal entry (`DECISION`, `ACTION`, `NUMBER`, `OPEN-ISSUE`, `-`) |
| `REVISE #id [ts] text` | rewrite an entry that this window changes, e.g. held → passed |
| `LOOKBACK t1-t2` | re-read earlier transcript lines, at most twice per window |
| `NEXT` | move to the next window |

### Closing the meeting

1. The whole journal becomes minutes in four sections:
   - 決議事項 (decisions)
   - 待辦與負責人 (actions and owners)
   - 保留與未決 (held and open items)
   - 會議概要 (overview)

   Every item carries a timestamp.
2. **Verification.** Each item is checked against the transcript around its citation, one short, isolated call per item. The item is kept, corrected, or dropped.

### Context and thinking management

- **Stateless turns.** No chat history accumulates; state lives only in the journal.
- **Hard fit.** For every call, `prompt + thinking budget + output ≤ 32k`. The journal view shrinks until the call fits.
- **Budgeted thinking.** The server caps reasoning per turn (`--reasoning-budget`) and returns it separately (`--reasoning-format deepseek`). It is never fed back into a later turn.

These principles come from DeerFlow's context engineering: state offloaded to a store, isolated sub-agents, and loading on demand. They are implemented as a small loop that can be ported to Kotlin on top of llama.cpp. DeerFlow's own runtime (LangGraph, sandbox) does not run on a phone.

## Run

Use the PrismML llama.cpp fork, which Bonsai 2 needs. `-c 65536 -np 2` gives two slots of 32k each.

```bash
llama-server -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -c 65536 -np 2 --jinja \
  --reasoning-format deepseek --reasoning-budget 768 --port 8091 --alias bonsai

python3 eval/journal_agent.py --urls http://127.0.0.1:8091/v1 --think 768 \
  --out runs/ja-bonsai2-think          # --think 0 with --reasoning-budget 0 for thinking off
```

Each session writes its journal, the minutes, the verified minutes, call counts, the largest prompt, and thinking tokens.

## Evaluation

The metric is **`eval/judge_prose_tx.py`**. Every cited statement is checked against the transcript from 30 s before its citation to 150 s after, and labelled `supported`, `contradicted` or `unsupported`. Headline: the **contradicted rate**.

The judge is Gemma-4-31B. It is independent of the teacher that produced the reference minutes.

```bash
JUDGE_SCRIPT=judge_prose_tx DIRS="runs/ja-bonsai2-think" bash scripts/v2_judge.sh
```

Reference points on the 38 held-out sessions, as the share of statements contradicted by the transcript:

| system | contradicted |
|---|---|
| teacher minutes (gold) | 11% |
| best earlier on-device pipeline (2B map → 4B reduce → guard) | 25% |
| **Bonsai 2 27B journal agent** | *in progress* |

## Why this direction

Earlier work distilled small students (2–4B) for a map-reduce pipeline. Their contradiction rate plateaued at about 2–3× the teacher's under every training method tried: SFT, DPO, RFT, GRPO, verifiers, and more data. The two errors that dominated were paraphrasing misreadings and merging errors at reduce. A 27B model fits the device only because it is ternary, and it removes that capacity ceiling. The agent design keeps each call short, and lets later evidence correct earlier notes.

## Layout

| path | contents |
|---|---|
| `eval/journal_agent.py` | the reading agent |
| `eval/judge_prose_tx.py`, `scripts/v2_judge.sh` | transcript-grounded judge |
| `summarizer/` | transcript ingest, windowing, citation resolution |
| `distill/`, `eval/` (others) | earlier distillation and evaluation work |

## Status

This is research, not production. Still to do:

- human evaluation;
- measurement on the phone (memory, speed, heat);
- the Kotlin port.
