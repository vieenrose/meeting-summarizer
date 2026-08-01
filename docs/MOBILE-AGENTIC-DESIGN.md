# Mobile agentic summarizer — delivering the Hindsight/Hermes capability set on a Boox

**Goal:** what `Qwen3.5-35B-A3B + Hindsight + Hermes-agent` does for agentic summarization,
running on an e-ink ARM device with ~1.5 GB and no server.

## 1. What the reference stack actually provides

| | Hindsight | Hermes-agent |
|---|---|---|
| memory tiers | World / Experiences / Mental Models | persistent memory, user profiles, skills |
| write path | LLM extracts facts, entities, relations → normalize → index | agent-curated memory |
| retrieval | 4 parallel strategies (dense vector, BM25, entity graph, temporal) → reciprocal-rank fusion → **cross-encoder rerank** | FTS5 session search + LLM summarization |
| analysis | `reflect` — LLM re-reads memories to form new connections | closed learning loop |
| infra | PostgreSQL, vector index, API server, Docker/K8s | CLI + gateways + 40+ tools + MCP |

Neither can run on a Boox: they assume a database, an embedding model, a reranker, a
server, and a model that is reliable at free-form tool calls. But **most of what they buy
is achievable without any of that**, because our task is narrower — one document, one
session, structured output.

## 2. Capability-by-capability mobile mapping

| capability | reference implementation | mobile equivalent | why it works here |
|---|---|---|---|
| **fact extraction** (retain) | LLM extracts entities/relations into a graph | **typed NOTES per chunk with `[timestamp]` + `[cN]` anchors** — measured 4.75 faith | our schema *is* the fact type system; the model only writes content it can see |
| **canonical entities** | entity normalization + graph | **speaker tags are already canonical** (`S1…Sn` by first appearance, per transcript-format v1) | ASR diarization gives us for free what Hindsight must infer |
| **temporal index** | time-series store | **timestamps are in every line**; anchors sort by `anchor_sec()` in code | ordering is exact, not inferred |
| **semantic retrieval** | dense vectors + FAISS/pgvector | **not needed for one document** — anchors point at exact chunk ids | retrieval over 5 chunks is an array index, not a search problem |
| **keyword retrieval (BM25)** | Lucene/pg full-text | **substring + speaker filter over the transcript**, ~50 lines of Kotlin | a transcript is small; scanning is microseconds |
| **cross-encoder rerank** | separate reranker model | **deterministic rank by anchor recency + section** | "later supersedes earlier" is a rule, not a judgement |
| **reciprocal-rank fusion** | merge 4 retrievers | **single ordered list** | one retriever, nothing to fuse |
| **reflect** (form new connections) | LLM re-reads memory | **bounded compress-per-section pass** with anchors in view | scoped to one section; this is where our residual 6.2% inversions live |
| **agent loop / tool calls** | Hermes ReAct-style, 40+ tools | **deterministic control flow; zero tool calls** | sub-1B scores 3.6% on BFCL multi-turn — free-form tool use is not available to us |
| **cross-session memory** | Postgres + user profile | **per-meeting JSON on disk**; optional append-only "series" file for recurring meetings | matches the product: meetings, not a lifelong assistant |
| **skills / self-improvement** | skill persistence | **out of scope** | no evidence a sub-1B model can curate skills |

**The core substitution:** Hindsight spends LLM calls and a vector database to *recover*
structure from unstructured memories. A meeting transcript **already has** that structure —
speaker tags, timestamps, chronological order. We index what is already there instead of
re-deriving it, which is what lets a 0.8B model plus ~200 lines of Kotlin stand in for a
35B model plus Postgres.

## 3. Measured evidence this works

16 held-out long meetings (median 16.2k tokens), `voxsum-qwen35-0.8b`, teacher-judged:

| mode | completed | faith | faith≤2 | invert | cover | calls | prompt tok |
|---|---|---|---|---|---|---|---|
| single-pass 32k | **8/16** | 4.00 | 25.0% | 0.0% | 4.25 | 1.0 | 14.5k |
| **map + typed merge** | **16/16** | **4.75** | **6.2%** | 6.2% | **4.62** | 5.2 | 19.3k |
| + agentic re-read | 16/16 | 4.75 | 6.2% | 6.2% | 4.62 | 5.4 | 19.6k |

Single-pass **failed outright on half the meetings** (context overflow, one at 30k tokens).
The harness completed all of them and cut the bad tail 4×.

**The agentic re-read rung did not pay.** It fired on 2/16 and changed nothing — matching
the published finding that a second routing level *"never helps and sometimes breaks
accuracy outright"*. Keep the harness deterministic.

## 4. Mobile budget

| component | cost |
|---|---|
| model weights (q4, int8 head) | ~600 MB |
| KV @ 8k window | **25 MB** |
| typed memory state | < 100 KB |
| harness code | ~200 lines Kotlin, no deps |
| **peak RSS** | **~1.0–1.1 GB** |
| calls for a 3-h meeting | ~8–10 (bounded, predictable) |

Compare the reference stack: a 35B-A3B model alone needs ~18 GB at q4, plus Postgres, plus
an embedding model, plus a cross-encoder.

## 5. What we give up honestly

- **No cross-document knowledge.** Hindsight's World/Mental-Model tiers accumulate across
  sessions; we keep per-meeting state. Recurring-meeting memory is a later increment.
- **No semantic search.** Fine for one transcript; would matter for "what did we decide
  about X three months ago", which is a different product.
- **No self-improvement.** Hermes' skill learning has no sub-1B evidence behind it.
- **Residual 6.2% inversions**, concentrated in the compress step — the one place the model
  still merges claims it cannot see the evidence for. Next target.

## 6. Next increments, in value order

1. **Fix compress, not routing.** Give the compress step the anchor *lines* (not just the
   note text) for the bullets it is merging — turning a judgement into a lookup. This is
   where the remaining inversions are.
2. **KV reuse.** Put memory *before* the chunk so the prefix is cacheable across chunks
   (published at 69% cache reuse). On ARM, prefill is the dominant cost.
3. **Recurring-meeting series memory** — an append-only per-series file, still no database.
4. Only then consider training the compress operation (rung 3).
