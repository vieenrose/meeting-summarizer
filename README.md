# meeting-summarizer

This project summarises long meetings on a phone. The input is a 1.5–3 h zh-TW meeting transcribed by on-device ASR (VibeVoice). The output is minutes whose every item cites the transcript line it rests on (`[M:SS]`).

Target device budget: **peak RSS ≤ 6 GB**, and every model turn fits a context of **≤ 32k tokens**.

## Status (2026-09-29)

| | |
|---|---|
| Current direction | Reading agent with an external journal, driven by **Bonsai 2 27B ternary (PTQ1_0, 5.95 GB)**. No fine-tuning. Every turn is ≤ 32k tokens and thinking is budgeted. |
| Dropped | Sub-27B students (MiniCPM5-2B, Gemma-4-E2B/E4B, Qwen3.5-4B). On contradictions, SFT, DPO, RFT and GRPO all plateau at about 2–3× the teacher's error rate (see [Findings](#findings)). Bonsai gen 1 was dropped for weak agency. |
| Not production-ready | No human evaluation yet, no on-device run, no Kotlin port. |

## Data

- **Gold corpus**: `Luigi/voxsum-meeting-gold-zh` on Hugging Face (private). The dataset card describes it.
  - 408 training sessions: 171 IVOD Legislative Yuan committee meetings and 237 AliMeeting meetings.
  - 38 held-out IVOD sessions.
  - Row kinds: `notes` (map, one window → notes), `synthesis` (numbered points) and `prose` (the final abstract).
  - The gold went through QA rounds until a round produced no confirmed fix: Qwen3.8-Flash-Next proposed defects and Claude reviewed each one. The gold follows the noisy ASR and is never normalised to real-world facts.
- Session splits: `data/split_v2.json`, with the held-out sessions under `heldout`.
- IVOD terms are personal / non-commercial. The owner decides training and distribution use.

## Evaluation

The reference metric is **`eval/judge_prose_tx.py`**. It checks every cited sentence against the transcript, from 30 s before its citation to 150 s after. The judge is Gemma-4-31B, which never taught IVOD. Each sentence is labelled supported, contradicted or unsupported, and the **contradicted rate** is the headline.

> `eval/judge_prose.py` judges against the gold *notes*. It counts true facts the gold omitted as fabrications, so use it for **coverage of the gold points** only.

To judge notes, sample 25 notes per session into a prose-shaped run dir; the `nj25-*` dirs are built this way. Mechanical checks: `eval/score_v2.py` (validator and prose-gate pass rates, note-fact recall).

Serving and judging:

```
JUDGE_SCRIPT=judge_prose_tx DIRS="runs/student/<run>" bash scripts/v2_judge.sh
```

The judge takes both GPUs (TP2); results are appended to `reports/v2_sft_night.txt`.

## Findings

All figures below are on the 38 held-out sessions, as the share of statements contradicted by the transcript.

| system | notes (map) | prose |
|---|---|---|
| gold (teacher map-reduce) | 6% | 11% |
| MiniCPM5-2B SFT, paraphrased notes | 17% | 30% |
| MiniCPM5-2B, evidence-first notes | 15% | – |
| MiniCPM5-2B, **extractive** notes (speaker + verbatim span) | **9–10%** | – |
| Gemma-4-E4B SFT notes | 13% | – |
| MiniCPM5 evidence notes → Gemma-4-E4B reduce → prose guard | – | 25% |

- **Where errors come from.** Reduce roughly doubles the error rate: gold notes are 7% contradicted, and student prose written from those same notes is about 20%. Paraphrasing is the other main source: the extractive map reaches 9–10%.
- **No training method moved contradictions below the plateau** on 2–4B students. This covers GRPO, RFT and DPO with lexical, window-judge and transcript-judged rewards; verifiers; self-consistency; CAD; base/SFT interpolation; a teacher-assistant (E4B → 2B); and more data (the learning curve is flat).
- **Best-of-4 prose picked by the judge reaches 21%**, but a 2B verifier cannot select it (45% accuracy).

The detailed log is in the project memory and in `reports/`.

## Reading agent (current work)

`eval/journal_agent.py` reads the transcript once, window by window (~6k tokens). The agent's **journal** lives outside the context. Each turn sees a bounded view of it: the recent entries, the entries that share keywords with the window, and a count of the rest.

Actions:

| action | effect |
|---|---|
| `NOTE [ts] (TYPE) …` | add an entry |
| `REVISE #id [ts] …` | rewrite an entry |
| `LOOKBACK t1-t2` | re-read earlier lines (at most 2 per window) |
| `NEXT` | move to the next window |

After the last window the agent writes structured minutes:

- 決議事項 (decisions)
- 待辦與負責人 (actions and owners)
- 保留與未決 (held and open items)
- 會議概要 (overview)

Every item is then verified against the transcript around its citation (Chain-of-Verification), with one isolated, short-context call per item.

Context and thinking management:

- Turns are stateless.
- `prompt + thinking budget + output ≤ 32k`; the journal view shrinks until a turn fits.
- Thinking is capped server-side (`--reasoning-budget`) and returned separately (`--reasoning-format deepseek`). It is never fed back into a later turn.

These ideas come from DeerFlow's context engineering (state offloaded to a store, isolated sub-agents, on-demand loading), implemented without its runtime: a phone runs neither LangGraph nor a sandbox.

Run the agent against the PrismML llama.cpp fork, whose binaries are in `~/Bonsai-demo/bin/cuda`:

```
llama-server -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -c 65536 -np 2 --jinja \
  --reasoning-format deepseek --reasoning-budget 768 --port 8091 --alias bonsai
python3 eval/journal_agent.py --urls http://127.0.0.1:8091/v1 --think 768 --out runs/student/ja-bonsai2-think
```

`-c 65536 -np 2` gives two slots of 32k.

## Layout

| path | contents |
|---|---|
| `summarizer/` | pipeline (windowing, map/reduce prompts, note parsing), transcript ingest |
| `distill/` | export of training rows, SFT (`sft_gemma.py`, works for Gemma/MiniCPM/Qwen), GRPO/DPO, data builders (evidence, extractive, verify, RFT), prose writer and gate, numerals |
| `eval/` | runners (vLLM students, OpenAI-compatible API pipeline, single pass, journal agent), judges, filters and guard (`prose_guard.py`) |
| `scripts/` | orchestration chains (`v2_*.sh`) |
| `runs/` | teacher gold (`runs/v2/w4000`, `runs/alimeeting/w4000`), student outputs, adapters |
| `deploy/` | merged HF checkpoints and GGUF exports (`deploy/gguf/*-Q4_K_M.gguf`) |
| `site/` | corpus explorer (served on the tailnet, port 8323) |

## Operational notes

- **Disk**: delete `runs/merged/` after evaluation. Merged checkpoints are regenerable, and a full disk once broke two evaluations.
- **Orphaned vLLM engines**: they can hold a GPU after `os._exit`. Free the GPU with `nvidia-smi --query-compute-apps=pid` and kill those pids. Never `pkill -f` a pattern that matches your own shell.
- **Flash-Next** (local, FreeToken): one instance per RTX 5090, on ports 1919 and 2919.
