# meeting-summarizer

Minutes of a long meeting, written **live, on the phone**, while the meeting is going on.

The input is a zh-TW meeting of 1.5–3.5 h, transcribed by on-device ASR. The output is structured minutes. Every item cites the transcript line it rests on.

**Target device:** OPPO Reno7 (Dimensity 900, 8 GB), with CPU-only llama.cpp. Every model call fits a context of ≤ 32k tokens, and the agent must keep pace with the meeting as it happens.

## Architecture

A **realtime reading agent** runs on a ~2B Q4_0 model. The agent protocol is designed and validated with **Qwen3.8-27B**, which serves as the quality reference and, later, as the teacher. The ~2B candidates run the *same* protocol, so the 27B's traces can be used directly for fine-tuning.

```mermaid
flowchart LR
    A[live ASR] --> S[transcript segments]
    S --> W[window closes<br/>~2k tokens · ~4 min of speech]
    W --> R{{reading turn<br/>NOTE · REVISE · LOOKBACK · NEXT}}
    R --> C{{check turn<br/>FIX · DROP · OK}}
    C --> J[(journal)]
    J --> M[minutes assembled from checked notes<br/>決議 · 待辦 · 保留]
    J --> O{{overview, 3–5 cited sentences}}
    O --> M
```

### One growing conversation per session

On a phone CPU, prefill dominates the cost (≈35–40 tok/s against 7–9 tok/s of decode for a 2B). The design therefore never recomputes what it has already read.

The candidate models are hybrids: Qwen3.5 has Gated-DeltaNet layers and LFM2.5 has short-convolution layers, so part of their state is recurrent. llama.cpp can reuse the cache for them only when the new prompt **extends** the previous one. A prompt that diverges after a shared prefix is recomputed from the divergence.

For this reason a session is a single conversation that only grows:

```
[system] [journal so far] ([window k] [actions] [check] [fixes])*
```

- Each window costs only its own tokens, plus a short check turn.
- A reading turn stops at `NEXT`. The stop word is kept in the history, so that the history matches exactly what was generated.
- When the next window would overflow 32k, the conversation restarts from `[system] [whole journal]`, one or two times in a 3.5 h meeting.
- Models that always think (LFM2.5-2.6B) get an empty `<think></think>` prefilled. The block is kept in the history (`preserve_thinking`), which keeps the cache valid. The agent detects these models by itself.

```mermaid
sequenceDiagram
    participant A as ASR
    participant H as harness
    participant L as llama.cpp (one slot)
    loop every window
        A->>H: transcript lines
        H->>L: + window k  (only new tokens prefilled)
        L-->>H: NOTE / REVISE / LOOKBACK … NEXT
        H->>L: + check the notes just written  (window still in context)
        L-->>H: FIX / DROP / OK
        opt context near 32k
            H->>L: restart: system + whole journal
        end
    end
    H->>L: fresh short call: overview from the journal
    H->>H: minutes = checked notes by type + cited overview
```

### Why the model does not write the minutes

Earlier distillation work showed that a small model merges items and invents facts at the reduce step: reducing doubled the error rate. Here the minutes are **assembled** from the checked notes, grouped by type (`DECISION` → 決議, `ACTION` → 待辦, `OPEN-ISSUE` → 保留). The model only writes a short overview, and every overview sentence must cite a time that exists in the journal.

The harness also guards against a small model's loops:
- at most 8 notes or revisions per turn;
- a note or revision that repeats an existing note is rejected (bigram Jaccard > 0.6);
- a citation must resolve to a transcript line.

### Incremental (chunked) prefill — next step

Prefilling a window only when it closes leaves the CPU idle while people talk. The next version prefills each ASR segment as soon as it arrives. The segment is appended as token IDs through `/completion` with `n_predict: 0`, and the user turn stays open until the window closes. When the window closes, only the check and the decode remain.

This changes no output, because the same tokens are computed. It only moves prefill into the wait. On the Reno7, prefill speed barely depends on chunk size:

| chunk (tokens) | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| Qwen3.5-2B Q4_0 prefill (tok/s) | 36.9 | 35.7 | 38.7 | 39.8 | 39.7 |

## Results so far

The protocol was validated with **Qwen3.8-27B** (NVFP4, served by NInfer on one RTX 5090) on 10 held-out IVOD sessions. The judge is Gemma-4-31B: `eval/judge_prose_tx.py` checks each cited statement against the transcript, from 30 s before its citation to 150 s after.

| | Qwen3.8-27B, realtime agent | gold (teacher) |
|---|---|---|
| notes contradicted (25 sampled per session) | 10 % | 7 % |
| notes unsupported | 24 % | 8 % |
| **minutes contradicted** | **11 %** | 17 % (gold prose) |
| coverage of the gold's key points | 0.88 | 0.90 |
| recall of gold decisions | 73 % | — |
| well-formed action lines | 100 % | — |

Per section of the minutes:

| section | contradicted | unsupported |
|---|---|---|
| decisions | 10 % | 22 % |
| actions | 9 % | 29 % |
| held / open | 11 % | 30 % |
| overview | 18 % | 55 % |

Two known weaknesses:
- The model attributes statements to bodies the excerpt does not name, for example "財政部說明…" when the speaker labels only say S1.
- The overview synthesizes passages far from its citations.

### Phone budget

A 3.5 h meeting needs about **55k tokens of prefill** in total, restarts included. The previous stateless agent needed about 180k.

The 27B's token volume was replayed at the measured speed of a 2B on the Reno7:
- with prefill at window close, the agent falls at most 1.2–2.6 min behind the meeting, and the minutes are ready 1.3–3.6 min after it ends;
- with incremental prefill, the modelled delays are 0.4–1.7 min behind and 0.8–3.0 min after the end.

Reno7 CPU speeds (8 threads, `llama-bench` pp512 / tg32, tok/s):

| model | prefill | decode |
|---|---|---|
| LFM2.5-1.2B Q4_0 | 78.1 | 18.2 |
| Qwen3.5-2B Q4_0 | 40.4 | 8.8 |
| LFM2.5-2.6B Q4_0 | 35.0 | 8.0 |
| Gemma-4-E2B Q4_0 (QAT) | 34.6 | 7.0 |
| MiniCPM5-2B Q4_0 | 33.5 | 9.2 |
| LFM2.5-8B-A1B Q4_0 (MoE) | 24.5 | 9.0 |
| Qwen3.5-4B Q4_0 | 15.4 | 4.2 |

Q4_0 beats Q4_K_M (−15 to −23 % prefill) and Q8_0 on this CPU. A 27B, even ternary (Bonsai), is out of reach on this phone: at best 2.6 tok/s of prefill.

## Plan

1. **Bake-off.** Every ~2B candidate runs the same agent on the same sessions and faces the same judge. They are ranked by contradiction rate, under the realtime constraint. This step is running.
2. **Fine-tuning.** Distill the best 2B on the 27B's agent traces from the training sessions, never from held-out ones. The trace format is the same one the student will run, so no conversion is needed.
3. **Incremental prefill.** Implement it, then measure it on the phone with the upstream llama.cpp Android build, the ASR running alongside, and a hot, throttled CPU.

## Run

```bash
# teacher / reference (any OpenAI-compatible server; llama-server reports cache use in `timings`)
python3 eval/realtime_agent.py --url http://127.0.0.1:8120/v1 --model q38 \
  --split data/split_rt_bakeoff.json --out runs/student/rt-q38-27b

# a ~2B candidate on llama.cpp (upstream): -np 4 slots of 32k
llama-server -m Qwen3.5-2B-Q4_0.gguf -ngl 99 -c 131072 -np 4 --jinja --port 8110 --alias rt
python3 eval/realtime_agent.py --url http://127.0.0.1:8110/v1 --model rt --parallel 4 \
  --phone-pp 40.4 --phone-tg 8.8 --out runs/student/rt-q35-2b

bash scripts/rt_bakeoff2.sh                 # all candidates, two at a time
bash scripts/rt_judge_one.sh q38-27b q35-2b  # judge notes + minutes, coverage, per-section report
python3 eval/rt_report.py                   # one table: faithfulness, coverage, protocol, phone lag
```

Each session record stores the notes, the minutes and a trace of every call. Each trace entry holds its arrival time and the tokens prefilled and decoded, so the phone timing can be recomputed for any measured speed.

## Layout

| path | contents |
|---|---|
| `eval/realtime_agent.py` | the realtime reading agent |
| `eval/rt_report.py`, `eval/minutes_report.py` | bake-off table, per-section report, phone timing model |
| `eval/judge_prose_tx.py`, `scripts/v2_judge.sh` | transcript-grounded judge |
| `scripts/rt_bakeoff2.sh`, `scripts/rt_judge_one.sh` | bake-off and judging runners |
| `eval/journal_agent.py` | earlier stateless journal agent (full re-prefill per turn) |
| `summarizer/` | transcript ingest, windowing, citation resolution |
| `distill/`, other `eval/` | earlier map-reduce distillation and evaluation work |

Data (transcripts, gold minutes, runs) is not included.

## Status

Research, not production. Still to do: the bake-off results, the fine-tuning, incremental prefill on the device, human evaluation, and the Kotlin port.
