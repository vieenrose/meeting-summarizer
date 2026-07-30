"""Transcript rendering — VoxSumDroid unified summarizer transcript format **v1**.

Spec: VoxSumDroid/docs/TRANSCRIPT-FORMAT.md (2026-07-29). Shape, one utterance per line:

    [<start>] <speaker>: <text>      diarized, name unknown  → S1, S2, …
    [<start>] <name>: <text>         diarized, name known    → real name
    [<start>] <text>                 no diarization          → no speaker field

Timestamp = start only: `M:SS` under 1 h, `H:MM:SS` from 1 h — seconds and
minutes-in-hour zero-padded, leading unit unpadded. S-tags numbered by order of first
appearance. No header/footer/markdown. Text emitted as-is, no escaping.

Corpora carry no timings, so start times are synthesized from utterance length at
realistic speaking rates (deterministic per meeting id).
"""
import random
import re

# chars/sec of speech (en ~150 wpm; zh ~280 chars/min) + inter-utterance pause
RATE = {"en": 14.0, "zh-TW": 4.7}
PAUSE_S = (0.3, 1.8)

# name pools for the "diarized, name known" variant
NAMES = {
    "en": ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Chris", "Dana", "Riley",
           "Jamie", "Casey", "Robin", "Quinn"],
    "zh-TW": ["王小明", "陳美玲", "林志豪", "張雅婷", "李建宏", "黃淑芬", "吳俊傑",
              "劉思穎", "蔡孟軒", "鄭家豪", "許婉婷", "楊宗翰"],
}


def clock(sec: float) -> str:
    """v1 timestamp: M:SS under one hour, H:MM:SS from 1 h; leading unit unpadded."""
    t = max(int(sec), 0)
    h, m, s = t // 3600, (t % 3600) // 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"


class SpeakerStyle:
    """'stags' (S1…Sn), 'names' (name pool / corpus-native), 'none' (no speaker field)."""

    def __init__(self, kind: str = "stags"):
        assert kind in ("stags", "names", "none")
        self.kind = kind


def _speaker_map(meeting: dict, style: SpeakerStyle) -> dict:
    # first-appearance order, independent of corpus ids (mirrors the spec rule)
    order = []
    for u in meeting["utterances"]:
        if u["speaker"] and u["speaker"] not in order:
            order.append(u["speaker"])
    if style.kind == "none":
        return {s: None for s in order}
    if style.kind == "stags":
        return {s: f"S{i + 1}" for i, s in enumerate(order)}
    # names: corpus-native labels that look like real names/roles pass through verbatim;
    # synthetic ids (Person 1, spk_3, …) get a pooled name, deterministically shuffled.
    rng = random.Random(meeting["id"])
    pool = list(NAMES[meeting["lang"]])
    rng.shuffle(pool)
    synthetic = re.compile(r"^(person|speaker|spk|s)[ _]?\d+$", re.I)
    out = {}
    for i, s in enumerate(order):
        if synthetic.match(s):
            out[s] = pool[i] if i < len(pool) else f"S{i + 1}"
        else:
            out[s] = s  # e.g. QMSum "Project Manager", real names
    return out


def render_transcript(meeting: dict, style: SpeakerStyle = SpeakerStyle()) -> str:
    """Normalized meeting -> v1 transcript string. Multi-line utterance text is joined
    to one line (one utterance = one line is a hard spec rule)."""
    rng = random.Random(meeting["id"] + "/ts")
    rate = RATE[meeting["lang"]]
    smap = _speaker_map(meeting, style)
    t = 0.0
    lines = []
    for u in meeting["utterances"]:
        text = " ".join(u["text"].split())
        if not text:
            continue
        label = smap.get(u["speaker"]) if u["speaker"] else None
        line = f"[{clock(t)}] "
        if label:
            line += f"{label}: "
        lines.append(line + text)
        t += len(text) / rate + rng.uniform(*PAUSE_S)
    return "\n".join(lines)


_TS = re.compile(r"^\[\d+:\d{2}(?::\d{2})?\] ")


def parse_line(line: str):
    """Reference parser per spec: split on FIRST '] ' then first ': ' after it.
    Returns (timestamp, speaker|None, text)."""
    m = _TS.match(line)
    if not m:
        return None, None, line
    ts = m.group(0)[1:-2]
    rest = line[m.end():]
    if ": " in rest:
        spk, text = rest.split(": ", 1)
        # a speaker field never contains '] ' or ': '; oddly long "speakers" mean the
        # colon belonged to the text (no diarization)
        if 0 < len(spk) <= 40:
            return ts, spk, text
    return ts, None, rest


def strip_tags(transcript: str) -> str:
    """Bare-text degradation (robustness training): drop timestamp + speaker fields."""
    return "\n".join(parse_line(l)[2] for l in transcript.split("\n"))
