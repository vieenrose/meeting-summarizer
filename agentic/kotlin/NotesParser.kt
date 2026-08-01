package studio.voxsum.core.agentic

/**
 * Tolerant parser for model output.
 *
 * Everything here degrades to "fewer items" rather than throwing. A 0.8B model will
 * occasionally emit a stray heading, drop a section, or forget an anchor; on-device there is
 * no operator to retry, so the parser must always return *something* usable and let the
 * agent's fallbacks handle emptiness.
 */
object NotesParser {

    private val SECTION_LINE = Regex("^([A-Z]+)\\s*[:：]\\s*$")
    private val BULLET = Regex("^\\s*[-*•·]\\s+")
    private val ANCHOR = Regex("\\[(\\d+):(\\d{2})(?::(\\d{2}))?]\\s*$")

    /** Parse a chunk-notes generation into typed items tagged with their source chunk. */
    fun parse(raw: String, chunkIndex: Int): Map<Section, List<NoteItem>> {
        val out = Section.entries.associateWith { mutableListOf<NoteItem>() }
        var current: Section? = null
        for (line in raw.lineSequence()) {
            val trimmed = line.trim()
            if (trimmed.isEmpty()) continue
            SECTION_LINE.find(trimmed)?.let { m ->
                current = Section.entries.firstOrNull { it.name == m.groupValues[1] }
                return@let
            }
            val sec = current ?: continue
            if (!BULLET.containsMatchIn(trimmed)) continue
            val body = BULLET.replace(trimmed, "").trim()
            // "-" alone is the canonical empty marker, not an item.
            if (body.isEmpty() || body == "-") continue
            out.getValue(sec) += NoteItem(stripAnchor(body), anchorSeconds(body), chunkIndex)
        }
        return out
    }

    /** Bullet lines from a compress generation, anchors preserved. */
    fun bullets(raw: String): List<String> = raw.lineSequence()
        .map { it.trim() }
        .filter { BULLET.containsMatchIn(it) }
        .map { BULLET.replace(it, "").trim() }
        .filter { it.isNotEmpty() && it != "-" }
        .toList()

    fun anchorSeconds(text: String): Int {
        val m = ANCHOR.find(text) ?: return -1
        val a = m.groupValues[1].toInt()
        val b = m.groupValues[2].toInt()
        val c = m.groupValues[3].takeIf { it.isNotEmpty() }?.toInt()
        return if (c != null) a * 3600 + b * 60 + c else a * 60 + b
    }

    fun stripAnchor(text: String): String = ANCHOR.replace(text, "").trim()
}

/** Prompt text. Kept in one object so the wording stays in lockstep with the training data —
 *  the model was fine-tuned on these exact shapes, and drift here silently costs quality. */
object Prompts {

    fun chunkNotes(lang: MeetingAgent.Lang, index: Int, chunk: String): String =
        if (lang == MeetingAgent.Lang.ZH_TW) """
            請閱讀以下會議逐字稿的其中一段，並針對這一段做筆記。
            請「完全」使用這個格式，每一點結尾用方括號標註支持該點的時間戳記：
            SUMMARY:
            - 重點 [0:12]
            DECISIONS:
            - 這一段做成的決策 [1:03]
            ACTIONS:
            - 負責人: 要做的事 [2:20]
            OPEN:
            - 未解決的問題 [3:04]
            TOPICS:
            - 討論的議題

            只寫這一段真的有講到的內容。若某區段沒有內容，該行只寫「-」。不要前言。

            逐字稿片段 c$index:
            $chunk
        """.trimIndent() else """
            Read this section of a meeting transcript and write notes about it.
            Use EXACTLY this format, ending every bullet with the timestamp that supports it:
            SUMMARY:
            - point [0:12]
            DECISIONS:
            - decision made in this section [1:03]
            ACTIONS:
            - owner: what they will do [2:20]
            OPEN:
            - unresolved question [3:04]
            TOPICS:
            - topic discussed

            Only write what this section actually says. If a section has nothing, put exactly "-".
            No preamble.

            Transcript section c$index:
            $chunk
        """.trimIndent()

    /**
     * Compression is scoped to ONE section with anchors visible, and the supersede rule is
     * stated explicitly: this is the step where a later reversal ("finance rejected it")
     * must beat an earlier approval, and where small models otherwise invert polarity.
     */
    fun compress(lang: MeetingAgent.Lang, section: Section, cap: Int, items: String): String =
        if (lang == MeetingAgent.Lang.ZH_TW) """
            以下是同一場會議「$section」區段的筆記，來自逐字稿的不同部分。請合併成最多 $cap 點。
            規則：保留 [時間戳記]。若兩點互相矛盾，保留時間較晚的那一點並刪去較早的——後面的說法會取代前面的。不要杜撰。只輸出條列。

            $items
        """.trimIndent() else """
            These are notes for the "$section" section of one meeting, collected from different
            parts of the transcript. Merge them into at most $cap bullets.
            Rules: keep the [timestamp] anchors. If two bullets disagree, keep the LATER
            timestamp and drop the earlier one — a later statement supersedes an earlier one.
            Do not invent anything. Output only the bullets.

            $items
        """.trimIndent()

    fun title(lang: MeetingAgent.Lang, notes: String): String =
        if (lang == MeetingAgent.Lang.ZH_TW)
            "請為以下會議記錄取一個簡短標題（8 個字以內）。只輸出標題本身。\n\n$notes"
        else
            "Write ONE short title (at most 8 words) for the meeting notes below. " +
                "Output only the title.\n\n$notes"
}
