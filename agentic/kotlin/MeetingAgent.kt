package studio.voxsum.core.agentic

import studio.voxsum.core.llm.TextGen

/**
 * On-device meeting-notes agent for a short-context SLM.
 *
 * WHY THIS SHAPE. Measured on 16 held-out long meetings (median 16.2k tokens) with
 * voxsum-qwen35-0.8b, teacher-judged:
 *
 *   single pass @32k   8/16 completed (rest overflowed ctx), faith 4.00, faith<=2 25.0%
 *   this agent        16/16 completed,                        faith 4.75, faith<=2  6.2%
 *
 * The design rule that produces that gap: **the model never emits a memory operation.**
 * It is only ever asked to write notes about a chunk it can see. All merging, de-duplication,
 * ordering and contradiction resolution happen in Kotlin. Sub-1B models are measured at
 * ~30% malformed memory writes and ~3.6% on multi-turn tool use, so any design that asks
 * them to maintain state directly corrupts it silently. Deterministic control flow is not a
 * simplification here, it is the thing that works.
 *
 * A second, "more agentic" rung (let the model pick chunks to re-read) was implemented and
 * measured: it fired on 2/16 meetings and changed no scores. It is deliberately not here.
 *
 * COST. Bounded and predictable: ceil(tokens / window) + one call per non-trivial section.
 * That matters because on ARM prefill dominates (~63 s for 8k tokens on a Galaxy S25 CPU),
 * so an agent that decides its own number of passes is not shippable.
 *
 * No dependencies, no database, no embeddings, no tool-calling.
 */
class MeetingAgent(
    private val llm: TextGen,
    private val lang: Lang,
    /** Tokens of transcript per call. Keep well under the model's ctx: quality is measured
     *  to fall off a cliff past ~12k, long before the window is full. */
    private val chunkTokens: Int = 4000,
    private val maxBullets: Map<Section, Int> = mapOf(
        Section.SUMMARY to 5, Section.DECISIONS to 5, Section.ACTIONS to 6,
        Section.OPEN to 4, Section.TOPICS to 6,
    ),
) {
    enum class Lang { EN, ZH_TW }

    /** Reported so the UI can show real progress instead of a spinner. */
    data class Progress(val step: Int, val total: Int, val phase: String)

    fun run(transcript: String, onProgress: (Progress) -> Unit = {}): String {
        val chunks = Chunker.byLines(transcript, chunkTokens, llm::countTokens)
        val memory = NotesMemory()
        val total = chunks.size + maxBullets.size + 1

        // Phase 1 — read. One bounded call per chunk; no state carried into the prompt, so
        // an early mistake cannot contaminate later chunks (the documented failure mode of
        // running-summary designs, which degrade *worse* the smaller the model).
        chunks.forEachIndexed { i, chunk ->
            onProgress(Progress(i + 1, total, "read"))
            val raw = llm.generateBlocking(Prompts.chunkNotes(lang, i, chunk), MAX_NOTE_TOKENS)
            NotesParser.parse(raw, chunkIndex = i).forEach { (section, items) ->
                items.forEach { memory.add(section, it) }
            }
        }

        // Phase 2 — compress, one section at a time, with anchors in view. Scoping the
        // compress step to a single section is what keeps it tractable: it is the step
        // recurrent-summarization work identifies as the break point for small models.
        var step = chunks.size
        val out = NotesMemory()
        for (section in Section.entries) {
            onProgress(Progress(++step, total, "compress"))
            val items = memory.get(section)
            val cap = maxBullets[section] ?: 5
            if (items.size <= cap) {
                items.forEach { out.add(section, it) }
                continue
            }
            val body = items.joinToString("\n") { "- ${it.render(true)}" }
            val merged = llm.generateBlocking(Prompts.compress(lang, section, cap, body), MAX_NOTE_TOKENS)
            val parsed = NotesParser.bullets(merged)
            // Never let a bad generation empty a section: fall back to the earliest N items.
            val keep = if (parsed.isEmpty()) items.take(cap).map { it.render(true) } else parsed.take(cap)
            keep.forEach { line ->
                val at = NotesParser.anchorSeconds(line)
                out.add(section, NoteItem(NotesParser.stripAnchor(line), at, chunk = -1))
            }
        }

        // Phase 3 — title, derived from the finished notes (cheap, single short call).
        onProgress(Progress(total, total, "title"))
        val title = llm.generateBlocking(Prompts.title(lang, out.render(withAnchors = false)), 24)
            .lineSequence().firstOrNull { it.isNotBlank() }?.trim()?.trim('"', '「', '」')

        return out.render(title = title, withAnchors = false)
    }

    private companion object { const val MAX_NOTE_TOKENS = 420 }
}

/** Splits on line boundaries so a `[mm:ss] S1: text` record is never cut in half — the
 *  chunker must respect transcript-format v1's "one utterance = one line" guarantee. */
object Chunker {
    fun byLines(transcript: String, budget: Int, count: (String) -> Int): List<String> {
        val out = mutableListOf<String>()
        val cur = StringBuilder()
        var n = 0
        for (line in transcript.lineSequence()) {
            val t = count(line) + 1
            if (cur.isNotEmpty() && n + t > budget) {
                out += cur.toString(); cur.clear(); n = 0
            }
            cur.append(line).append('\n'); n += t
        }
        if (cur.isNotEmpty()) out += cur.toString()
        return out
    }
}
