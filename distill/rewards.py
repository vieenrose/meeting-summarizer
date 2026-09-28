"""Programmatic reward for the summary (synthesis) step.

GRPO needs a score for every sampled summary. Everything this project already checks by rule is a
usable signal, so the reward is assembled from those checks rather than from a judge model:

  structure    numbered points, 1-8 of them, each <= 130 characters, total inside the band.
  citations    share of points carrying at least one timestamp that exists in the transcript.
  coverage     share of the meeting's thirds (those with real speech) that the citations reach.
  recall       share of the gold summary's figures and names the sample reproduces.
  grounding    share of the sample's own figures and names that occur in the transcript --
               the derivability rule the gold corpus was repaired against, applied to the student.

Two failures seen in the first SFT run are hard zeros rather than small penalties, because a
gradient that merely discourages them was not enough to stop them: copying the prompt's own
formatting rules back as the summary, and a single run-on point holding the whole meeting.

Recall carries the most weight. Without it the easiest route to a high score is a short, safe,
well-cited summary that says nothing -- structure and citations are cheap, content is not.
"""
import re

from summarizer.ingest import resolve_citation
from summarizer.pipeline import uncoverable_thirds

CITE = re.compile(r"[\[［](\d{1,2}(?::\d{2}){1,2})[\]］]")
POINT = re.compile(r"^\s*\d+[.、]")
FIGURE = re.compile(r"\d[\d.]*(?:億|萬|千|%|人|案|條)")
NAME = re.compile(r"[一-鿿]{2,3}(?=委員|部長|次長|署長|主席|召委|院長)")
# Fragments of the synthesis prompt's instructions. A summary containing them has copied the rules.
PROMPT_ECHO = ("每點 60 到 100 字", "不含時間標記", "每點一行", "句尾附上出處時間", "時間必須來自筆記",
               "會議前、中、後三段都要有內容", "只輸出編號清單", "優先寫決議")

WEIGHTS = {"structure": 0.15, "citations": 0.2, "coverage": 0.15, "recall": 0.35, "grounding": 0.15}


def _norm(text):
    return re.sub(r"[\s,，]", "", text)


def _facts(text):
    t = _norm(text)
    return set(FIGURE.findall(t)) | set(NAME.findall(t))


def body_len(point):
    return len(re.sub(r"[\[［][^\]］]*[\]］]|\s", "", point))


def score_summary(text, lines, gold_summary, detail=False):
    points = [p.strip() for p in text.splitlines() if POINT.match(p)]
    parts = dict.fromkeys(WEIGHTS, 0.0)
    reason = None
    if not points:
        reason = "no numbered points"
    elif any(marker in text for marker in PROMPT_ECHO):
        reason = "copied the prompt's rules"
    elif max(body_len(p) for p in points) > 260:
        reason = "run-on point"
    if reason:
        return (0.0, {"reason": reason, **parts}) if detail else 0.0

    total = sum(body_len(p) for p in points)
    ok_len = sum(1 for p in points if body_len(p) <= 130) / len(points)
    ok_count = 1.0 if 3 <= len(points) <= 8 else 0.5
    ok_total = 1.0 if 150 <= total <= 620 else max(0.0, 1 - abs(total - 385) / 700)
    parts["structure"] = (ok_len + ok_count + ok_total) / 3

    n = max(1, len(lines))
    covered, cited = set(), 0
    for p in points:
        idx = [i for i in (resolve_citation(c, lines) for c in CITE.findall(p)) if i is not None]
        if idx:
            cited += 1
            covered.update(min(2, 3 * i // n) for i in idx)
    parts["citations"] = cited / len(points)
    needed = {0, 1, 2} - uncoverable_thirds(lines)
    parts["coverage"] = len(covered & needed) / len(needed) if needed else 1.0

    gold = _facts("\n".join(gold_summary))
    body = _norm(text)
    parts["recall"] = sum(1 for f in gold if f in body) / len(gold) if gold else 1.0

    hay = _norm(" ".join(l.text for l in lines))
    own = _facts(text)
    parts["grounding"] = sum(1 for f in own if f in hay) / len(own) if own else 1.0

    score = sum(WEIGHTS[k] * parts[k] for k in WEIGHTS)
    # Length as a gate, not a weighted part. The first overnight run showed why: with length only
    # a slice of `structure`, GRPO learned that longer points recall more of the gold's facts, and
    # by step 460 held-out summaries broke the validator's per-point and total limits 3-4x as
    # often while mean reward barely moved. The validator treats either overrun as a failure, so
    # the reward now halves for each.
    if any(body_len(p) > 130 for p in points):
        score *= 0.5
        parts["over_long_point"] = True
    if total > 620:
        score *= 0.5
        parts["over_total"] = True
    return (score, parts) if detail else score


# ---------------------------------------------------------------------------------------------
# Notes step
# ---------------------------------------------------------------------------------------------

NOTE_WEIGHTS = {"structure": 0.15, "citations": 0.2, "recall": 0.45, "grounding": 0.2}


def window_lines(prompt_text):
    """The window's transcript lines, taken from the notes prompt itself."""
    from summarizer.ingest import parse_line
    body = prompt_text.split("TRANSCRIPT", 1)[-1]
    return [parse_line(l) for l in body.splitlines() if re.match(r"^\s*\[\d", l)]


def score_notes(text, prompt_text, gold_notes, detail=False):
    """Reward for one window's notes.

    Same shape as the summary reward, scoped to the window: citations must fall inside the window
    the student was shown, and grounding is checked against that window's text. Recall is against
    the teacher's notes for the same window, so a student that notes the procedural chatter and
    misses the vote scores low even when every note it writes is true.
    """
    from summarizer.pipeline import _FORMAT_ECHOES, parse_turn
    turn = parse_turn(text)
    parts = dict.fromkeys(NOTE_WEIGHTS, 0.0)
    reason = None
    if not turn.notes:
        reason = "no notes"
    elif any(m in text for m in _FORMAT_ECHOES) or any(m in text for m in PROMPT_ECHO):
        reason = "copied the prompt"
    if reason:
        return (0.0, {"reason": reason, **parts}) if detail else 0.0

    lines = window_lines(prompt_text)
    notes = turn.notes
    parts["structure"] = (1.0 if 1 <= len(notes) <= 6 else 0.5) * (1.0 if not turn.rejected else 0.7)
    parts["citations"] = sum(1 for _, ts, _ in notes if resolve_citation(ts, lines) is not None) / len(notes)
    gold = _facts(gold_notes)
    body = _norm(text)
    parts["recall"] = sum(1 for f in gold if f in body) / len(gold) if gold else 1.0
    hay = _norm(" ".join(l.text for l in lines))
    own = _facts(text)
    parts["grounding"] = sum(1 for f in own if f in hay) / len(own) if own else 1.0
    score = sum(NOTE_WEIGHTS[k] * parts[k] for k in NOTE_WEIGHTS)
    return (score, parts) if detail else score
