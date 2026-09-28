"""Local grounding: are a note's figures and names said NEAR its own anchor?

Global grounding (distill/rewards.py) accepts a figure found anywhere in the window, so a note that
pins the right amount on the wrong article or the wrong speaker scores full marks. The judge's
errors are mostly of that kind. Here each fact must appear in the transcript from the anchor line
to SPAN seconds after it: the note's anchor is where its claim starts, and a statement rarely runs
longer than that.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill.numerals import values  # noqa: E402
from distill.rewards import NAME, _norm  # noqa: E402
from summarizer.ingest import resolve_citation  # noqa: E402

SPAN = 90


def note_local(ts, text, lines):
    """(facts found near the anchor, facts in the note); None if the note has no facts."""
    own = values(text) | {("name", n) for n in NAME.findall(_norm(text))}
    if not own:
        return None
    i = resolve_citation(ts, lines)
    if i is None:
        return 0, len(own)
    t0 = lines[i].start_s
    near = " ".join(l.text for l in lines[i:] if l.start_s - t0 <= SPAN)
    have = values(near)
    hay = _norm(near)
    return sum(1 for f in own if (f[1] in hay if f[0] == "name" else f in have)), len(own)


def session_local(notes, lines):
    hit = tot = 0
    for n in notes:
        r = note_local(n["ts"], re.sub(r"^\S+?[:：]\s*", "", n["text"]), lines)
        if r:
            hit, tot = hit + r[0], tot + r[1]
    return hit / tot if tot else 1.0
