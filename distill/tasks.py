"""Prompt suite the student is trained on.

Family A — VoxSumDroid-compat: templates copied VERBATIM from Summarizer.kt /
ActionItemExtractor.kt (+ validate_llm.py's speaker prompt). If the app changes,
re-sync by hand and retrain.

Family B — insight tasks: LFM2-Transcript-style fixed prompts, en + zh-TW, that the app
can adopt for a richer "insights" panel.
"""

# ---------------------------------------------------------------- family A: verbatim
MAP_TEMPLATE = (
    "%s\nWrite the summary of the transcript section below %s. "
    "Output only the summary itself — no headings, no multiple versions, no preamble.\n\n"
    "Transcript:\n%s"
)
REDUCE_TEMPLATE = (
    "%s\nCombine the partial summaries below %s. "
    "Output only the summary itself — no headings, no multiple versions, no preamble.\n\n"
    "Partial summaries:\n%s"
)
TITLE_TEMPLATE = (
    "Write ONE short title (at most 8 words) for the summary below.%s "
    "Output only the title text — no quotes, no list, no preamble.\n\nSummary:\n%s"
)
SHRINK_TEMPLATE = (
    "%s\nThe summary below is too long. Rewrite it %s. Keep ONLY the most important points"
    " and drop minor detail. Output only the summary itself — no headings, no multiple"
    " versions, no preamble.\n\nSummary:\n%s"
)
MAP_TEMPLATE_ZH = (
    "請將以下逐字稿整理成一份簡潔的摘要，條列重點（每點 20 字以內）。"
    "只輸出摘要本身——不要標題、不要多個版本、不要前言。\n\n逐字稿:\n%s"
)
REDUCE_TEMPLATE_ZH = (
    "請將以下多段部分摘要合併成一份簡潔的摘要，條列最重要的重點（最多 7 點，每點 20 字以內），"
    "合併重複內容、刪去次要細節。只輸出摘要本身——不要標題、不要多個版本、不要前言。\n\n部分摘要:\n%s"
)
TITLE_TEMPLATE_ZH = (
    "請為以下摘要取一個簡短標題（8 個字以內）。只輸出標題本身——不要引號、不要條列、不要前言。\n\n摘要:\n%s"
)
SHRINK_TEMPLATE_ZH = (
    "以下摘要太長了。請改寫成最多 7 點的條列摘要（每點 20 字以內），只保留最重要的重點、刪去次要細節。"
    "只輸出摘要本身——不要標題、不要多個版本、不要前言。\n\n摘要:\n%s"
)
ACTION_MAP_TEMPLATE = (
    "From the transcript section below, list the concrete ACTION ITEMS (who needs to do what, "
    "with any deadline) and any key DECISIONS made, as short bullet points.%s Output only the "
    "bullets — no headings, no preamble. If there are none, output exactly \"-\".\n\n"
    "Transcript:\n%s\n\nItems:"
)
ACTION_REDUCE_TEMPLATE = (
    "Combine and de-duplicate the action items and decisions below into one short bullet list, "
    "keeping who-does-what.%s Output only the bullets — no headings, no preamble.\n\n"
    "Items:\n%s\n\nItems:"
)
SPK_SYS = (
    "You are an expert at analyzing speech patterns and identifying speaker identities from"
    " transcripts. Be precise and only suggest names when you have clear evidence. IMPORTANT: You"
    " MUST respond in the EXACT SAME LANGUAGE as the input text. Do not translate to English."
)
SPK_USER = (
    "Analyze the following utterances from a single speaker and suggest a name for this speaker."
    " Provide your answer in this exact format:\nNAME: [suggested name]\nCONFIDENCE: [high/medium/low]"
    "\nREASON: [brief explanation]\n\nUtterances from this speaker:\n%s"
)

# SummaryStyle enum: (map instruction, reduce instruction, map tokens, reduce tokens)
STYLES = {
    "bullet": (
        "as 3-5 short bullet points (each under 20 words)",
        "into ONE concise summary of AT MOST 7 short bullet points — keep only the most "
        "important points, merge overlapping ones, drop minor detail (each bullet under 20 words)",
        224, 288,
    ),
    "executive": (
        "as a 2-3 sentence executive summary",
        "into ONE tight executive summary of 2-3 sentences (at most 60 words) — keep only what matters most",
        200, 224,
    ),
    "narrative": (
        "as a short, flowing paragraph",
        "into ONE cohesive paragraph of at most 6 sentences — keep only the most important "
        "points, do not try to include everything",
        288, 384,
    ),
}

TARGET_LANG = {"en": "English", "zh-TW": "Traditional Chinese (繁體中文)"}


def lang_clause(target: str | None) -> str:
    """Summarizer's strengthened output-language clause. target = human-readable name or None."""
    if target is None:
        return " Write it in the same language as the transcript."
    return (
        f" Write the ENTIRE output in {target}. The transcript may be in another language —"
        f" translate as you summarize. Do not use any language other than {target}."
    )


def action_lang_clause(target: str | None) -> str:
    """ActionItemExtractor's strengthened clause."""
    if target is None:
        return " Write them in the same language as the transcript."
    return (
        f" Write the ENTIRE output in {target}. The transcript may be in another language —"
        f" translate as you extract. Do not use any language other than {target}."
    )


# Default + varied user instructions (cfg.summaryPrompt slot in the map/reduce templates)
USER_INSTRUCTIONS_EN = [
    "Summarize the key points of this transcript.",
    "Summarize this meeting.",
    "Summarize the discussion, focusing on outcomes.",
    "Give me the key takeaways from this meeting.",
]
USER_INSTRUCTIONS_ZH = [
    "請摘要這份逐字稿的重點。",
    "請總結這場會議。",
    "請摘要討論內容，聚焦在結論與成果。",
]

# ---------------------------------------------------------------- family B: insights
# Fixed task prompts (en / zh-TW). {t} = transcript. Output plain bullets/text — the app
# renders raw text. Each entry: (prompt_en, prompt_zh, max_tokens)
INSIGHT_TASKS = {
    "exec_summary": (
        "Provide a brief executive summary (2-3 sentences) of the key outcomes and decisions "
        "of the meeting transcript below.%s Output only the summary.\n\nTranscript:\n%s",
        "請為以下會議逐字稿提供簡短的執行摘要（2-3 句），聚焦關鍵成果與決策。只輸出摘要。\n\n逐字稿:\n%s",
        224,
    ),
    "detailed_summary": (
        "Provide a detailed summary of the meeting transcript below, covering all major topics, "
        "discussions, and outcomes in paragraph form.%s Output only the summary.\n\nTranscript:\n%s",
        "請為以下會議逐字稿提供詳細摘要，以段落形式涵蓋所有主要議題、討論與結果。只輸出摘要。\n\n逐字稿:\n%s",
        512,
    ),
    "action_items": (
        "List the specific action items that were assigned during this meeting — who needs to do "
        "what, with any deadline.%s Output only the bullet list. If there are none, output exactly \"-\".\n\nTranscript:\n%s",
        "請列出這場會議指派的具體行動項目——誰要做什麼、期限為何。只輸出條列清單。若沒有任何項目，請輸出「-」。\n\n逐字稿:\n%s",
        384,
    ),
    "decisions": (
        "List the key decisions that were made during this meeting.%s Output only the bullet list. "
        "If there are none, output exactly \"-\".\n\nTranscript:\n%s",
        "請列出這場會議做成的關鍵決策。只輸出條列清單。若沒有任何決策，請輸出「-」。\n\n逐字稿:\n%s",
        320,
    ),
    "topics": (
        "List the main topics and subjects that were discussed in this meeting.%s Output only the "
        "bullet list.\n\nTranscript:\n%s",
        "請列出這場會議討論的主要議題。只輸出條列清單。\n\n逐字稿:\n%s",
        256,
    ),
    "open_questions": (
        "List the open questions, unresolved issues, and follow-ups left at the end of this "
        "meeting.%s Output only the bullet list. If there are none, output exactly \"-\".\n\nTranscript:\n%s",
        "請列出這場會議結束時仍未解決的問題、待釐清事項與後續追蹤事項。只輸出條列清單。若沒有，請輸出「-」。\n\n逐字稿:\n%s",
        320,
    ),
    "risks_disagreements": (
        "List any notable disagreements, concerns, or risks raised in this meeting, and who raised "
        "them.%s Output only the bullet list. If there are none, output exactly \"-\".\n\nTranscript:\n%s",
        "請列出這場會議中值得注意的分歧、疑慮或風險，以及提出者。只輸出條列清單。若沒有，請輸出「-」。\n\n逐字稿:\n%s",
        320,
    ),
}
