# VoxSum meeting agent — a mobile-fit agentic summarizer

Four Kotlin files, no dependencies, no server, no database, no embedding model, no
tool-calling. Drops into VoxSumDroid against the existing `TextGen` interface.

| file | role |
|---|---|
| `MeetingNotes.kt` | typed external memory (`Section`, `NoteItem`, `NotesMemory`) |
| `MeetingAgent.kt` | the orchestrator + line-boundary chunker |
| `NotesParser.kt` | tolerant output parser + prompts |

## Why not an existing framework

Every agent framework surveyed assumes infrastructure a Boox does not have:

| framework | blocker |
|---|---|
| Hindsight | PostgreSQL, dense vector index, BM25 index, cross-encoder reranker, API server |
| Hermes-agent | CLI + gateways + 40+ tools + MCP; free-form tool calling |
| LangChain / LlamaIndex / CrewAI | Python runtime, server-side, heavyweight |
| MediaPipe / LiteRT-LM / ExecuTorch | inference engines — no orchestration layer |

They also assume a model that reliably emits well-formed tool/memory calls. Measured, that
is not available below ~4B: sub-1B models score **3.6% on BFCL multi-turn** and produce
**~30% malformed memory writes at 3B**.

## The one design rule

**The model never emits a memory operation.** It is only ever asked to write notes about a
chunk it can currently see. Merging, de-duplication, ordering and contradiction resolution
happen in Kotlin.

That inverts the usual agent design, and it is why this works at 0.8B.

## Measured

16 held-out long meetings, median 16.2k tokens, `voxsum-qwen35-0.8b`, teacher-judged:

| | completed | faith | faith≤2 | coverage |
|---|---|---|---|---|
| single pass @32k | **8/16** (rest overflowed) | 4.00 | 25.0% | 4.25 |
| **this agent** | **16/16** | **4.75** | **6.2%** | **4.62** |

Cost: 5.2 calls and +33% prompt tokens versus one single-pass call — bounded and
predictable, which matters because ARM prefill dominates wall-clock (~63 s per 8k tokens
on a Galaxy S25 CPU).

A more agentic variant (model picks chunks to re-read) was built and measured: it fired on
2/16 meetings and changed no scores. Deliberately omitted.

## Memory budget

| | |
|---|---|
| weights (q4, int8 head) | ~600 MB |
| KV cache @8k window | 25 MB |
| typed memory state | <100 KB |
| **peak RSS** | **~1.0–1.1 GB** |

## Usage

```kotlin
val agent = MeetingAgent(llm = engine, lang = MeetingAgent.Lang.ZH_TW)
val notes = agent.run(TranscriptFormat.format(utterances)) { p ->
    setProgress(p.step.toFloat() / p.total, p.phase)   // real progress, not a spinner
}
```

Input is transcript-format v1; output is transcript-format v2 NOTES. Both unchanged, so
the existing post-processing (`stripThink`, `cleanSummary`) still applies.

## Interface required

```kotlin
interface TextGen {
    fun generateBlocking(prompt: String, maxTokens: Int): String
    fun countTokens(text: String): Int
}
```

`LiteLlmEngine` already provides generation; `countTokens` can be the engine's tokenizer or
a ~0.6 chars/token estimate for CJK — chunk sizing is not sensitive to a few percent.

## Known gaps

- **Residual 6.2% inversions**, concentrated in the compress step — the only place the model
  merges claims without the source lines in view. Next fix: pass the anchored transcript
  lines alongside the bullets, turning a judgement into a lookup.
- **No cross-meeting memory.** Per-meeting state only; a recurring-series file is a later
  increment.
- **No semantic retrieval** — unnecessary for one transcript, would be needed for
  "what did we decide about X three months ago".
