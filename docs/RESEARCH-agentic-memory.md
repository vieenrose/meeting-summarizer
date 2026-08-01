# Agentic + external memory for long-document summarization with short-context SLMs

**Status:** research proposal, 2026-08-01. Prototype-first; training only if the harness
plateaus.

## 1. Thesis

A sub-1B model with a *short* optimized context (8–16k) plus an external memory and a
lightweight harness should summarize a 128k-token document **better and cheaper** than the
same model given a 128k context.

## 2. Why — our own measurements say the model is good, but only when the window is small

Measured on `voxsum-qwen35-0.8b` (48 held-out meetings, teacher-judged, greedy decoding):

| transcript tokens | faith (1-5) | faith≤2 | inversion rate | KV @q4 |
|---|---|---|---|---|
| <4k | 4.03 | 20% | 8.6% | 13 MB |
| **4k–12k** | **4.05** | **18%** | **1.8%** | 25–50 MB |
| >12k | 3.36 | 46% | **15.3%** | 101 MB+ |

Two facts follow, and together they are the whole argument:

1. **Quality is a step function of window size, not of model size.** The same weights are
   near-perfect on polarity at 4–12k (1.8% inversions) and unreliable above 12k (15.3%).
   Scaling the model did not move this cliff: Qwen3-0.6B, Qwen3.5-0.8B and (pending)
   Qwen3.5-2B all degrade at the same place.
2. **Cost is superlinear in window size.** KV grows linearly (13 → 403 MB from 4k to 128k)
   and prefill degrades superlinearly (−31% tok/s from 4k to 16k, measured).

So a system that never asks the model to exceed ~12k gets the good numbers *and* the cheap
numbers. The open question is whether the orchestration around it can preserve global
coherence — which is what this research is about.

## 3. Constraint: the harness must be mobile-weight too

Non-negotiable, because it runs inside VoxSumDroid on ARM:

- **No Python, no server, no vector DB, no embedding model.** The harness must be
  implementable in plain Kotlin against the existing `TextGen` interface.
- **Bounded state.** External memory is a small structured text object (target ≤2k tokens),
  not a growing store. Memory + chunk must always fit the short window.
- **Deterministic control flow.** A 0.8B model is not reliable at free-form tool selection.
  The harness decides *when* to call; the model decides *what the content is*. Any
  model-driven branching must be a constrained choice (a bounded enum), never open-ended.
- **Bounded worst case.** Total LLM calls must be predictable — a user cannot wait on an
  agent that decides to re-read 40 times.

## 4. Approach — three rungs, cheapest first

**Rung 0 (baseline, exists today):** hierarchical map-reduce. Map each ~10k chunk to notes,
fold pairwise. Stateless between chunks — cannot carry a decision made in chunk 1 into
chunk 7, and the reduce step sees only summaries, never evidence.

**Rung 1 — running structured state (prompt-only).** Memory *is* the NOTES object.
For each chunk: `(current NOTES, chunk_i) → updated NOTES`. Sections accumulate; ACTIONS and
DECISIONS get appended or amended. Known risk: **error accumulation** — an early wrong claim
is copied forward and never revisited. Mitigation to test: emit each memory item with a
provenance tag (`[c3]`) so later steps can contradict it.

**Rung 2 — gist + selective re-read (prompt-only, ReadAgent-like).** Pass 1 builds a
per-chunk *gist* (2–3 lines + a chunk id). Pass 2 assembles the answer from gists, and where
a section is under-specified, re-reads **only** the 1–2 chunks whose gists claim relevance.
Re-reads are capped (e.g. ≤3) to bound latency. This is where "agentic" earns its keep: the
model chooses what to look at again, but from a closed list.

**Rung 3 — fine-tune for the memory operations (only if 1–2 plateau).** Distil trajectories
from a large teacher: (state, chunk) → state′ updates, gist writing, and re-read selection.
Train the sub-1B on those operations rather than on one-shot summarization. This is where the
"train a sub-1B for agentic memory" idea belongs — *after* we know how much the harness alone
buys.

## 5. Evaluation

Same instrument as the rest of the project, so numbers are comparable:

- **Primary:** faith≤2 rate and **inversion rate** on >12k documents (current: 46% / 15.3%).
- **Secondary:** coverage, NOTES format compliance, cross-lingual behaviour.
- **Cost:** total tokens prefilled, peak KV, wall-clock, number of LLM calls.
- **Baselines:** (a) single-pass 32k, (b) hierarchical map-reduce, (c) rungs 1–3.

A rung only wins if it improves faithfulness **without** inflating total prefill — on ARM,
prefill is the binding cost, so an approach that re-reads everything twice must justify it.

## 6. Failure modes to measure, not assume

- **Error accumulation / drift** in running state (rung 1's central risk).
- **Lost-in-the-middle** inside each chunk — position bias persists even in short windows.
- **Re-read thrash**: the model asking for chunks that don't help, burning prefill.
- **Global structure loss**: decisions that only make sense across the whole meeting
  (e.g. a budget agreed in chunk 2 and reversed in chunk 9) — the hardest case, and the
  one where inversion errors will show up.

## 7. Immediate next step

Implement rungs 1 and 2 as a prototype harness against the *already-shipped*
`voxsum-qwen35-0.8b` at 8k and 12k windows, and measure on the same 48 held-out meetings.
No training. If either rung takes >12k-token inversions materially below 15.3% at equal or
lower total prefill, the harness is the answer and the fine-tune is optional.
