# Summarizer output format

## v2: structured meeting notes (the NOTES task)

One call → one complete note (LFM2-2.6B-Transcript-inspired, but single-pass instead of
six calls). Fixed **ASCII section keys** so parsing is locale-independent; the *content*
is in the transcript's (or requested target) language.

```
TITLE: <one line, ≤ 8 words>
SUMMARY:
- 3-5 short bullets
DECISIONS:
- key decisions made
ACTIONS:
- <owner>: <task> (期限/due: <deadline>)
OPEN:
- open questions / follow-ups
TOPICS:
- main topics discussed
```

Rules:

- Section keys `TITLE: SUMMARY: DECISIONS: ACTIONS: OPEN: TOPICS:` appear at
  start-of-line, uppercase ASCII, **always in this order**, always all present.
- `TITLE:` content is inline on the same line; every other section's content is `- `
  bullets on the following lines.
- An empty section contains exactly one line `-` (same canonical marker as v1).
- `ACTIONS` bullet grammar: `- <owner>: <task>` with optional ` (due: …)` en /
  ` (期限: …)` zh. Owner is the speaker label/name as it appears in the transcript.
- No markdown beyond the `- ` bullets; blank lines between sections are permitted and
  ignored by parsers.
- Parse rule: a new section starts at a line matching `^[A-Z]+:`; everything until the
  next key line belongs to the current section. Unknown keys: preserve as opaque text.
- The app renders localized headers itself (摘要/決策/行動項目/…) — keys are wire
  format, not UI text.

## v1: per-task outputs

What `voxsum-qwen3-0.6b` is trained to emit, per task. Counterpart to VoxSumDroid's
`TRANSCRIPT-FORMAT.md` (input side). "Trained" means high-probability behavior of a
0.6B model, not a grammar guarantee — the app's post-processing backstops
(`stripThink`, `cleanSummary`, `tooLong`→shrink, sentence dedup) stay in place.

## Global rules (all tasks)

- **Plain text only.** No markdown headings, no code fences, no bold/italic.
- **No preamble, no multiple versions, no trailing commentary.** The answer starts at
  the first content character.
- A leading **empty `<think>\n\n</think>` block may appear** (Qwen3 runtime habit) —
  consumers must strip it (`SummaryText.stripThink` already does).
- **Language** = the transcript's language, unless the prompt carries the
  target-language clause — then the entire output is in that language
  (measured compliance: 0.98-1.00).
- **Bullets use `- ` (dash + space), one item per line.** The app renders them as `• `.
- **No timestamps or speaker tags echoed** into outputs; speakers are referred to by
  their label/name in prose ("小陳 will…", "S1 asked…") only when attribution matters.

## Per task

| task (prompt family) | shape |
|---|---|
| summarize / bullet style | 3-5 bullets (single chunk) or ≤7 bullets (combined), each under ~20 words |
| summarize / executive | 2-3 sentences of prose, ≤ ~60 words, no bullets |
| summarize / narrative | one flowing paragraph, ≤ 6 sentences |
| title | ONE line, ≤ 8 words (en) / short phrase (zh-TW), no quotes, no numbering, no "Title:" |
| action items + decisions (`ActionItemExtractor` + insight `action_items`, `decisions`) | bullet list of *who does what, with deadline when stated*; **exactly `-` when none exist** |
| topics | bullet list of discussed subjects |
| open questions / follow-ups | bullet list; exactly `-` when none |
| risks / disagreements | bullet list incl. who raised them; exactly `-` when none |
| detailed summary | paragraph-form prose covering major topics and outcomes |

## The `-` empty marker

For extraction tasks (actions, decisions, open questions, risks), an output of exactly
`-` is the canonical "nothing found" marker — the app treats it as empty, never as an
item. Trained explicitly; consumers must check for it before rendering.

## Length backstop

The model targets the counts above, but on very dense hour-long input it can overrun.
`SummaryText.tooLong` (>12 non-blank lines or >1200 chars) triggering one shrink pass
remains the guaranteed bound — keep it.
