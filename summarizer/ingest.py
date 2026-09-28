"""VibeVoice-ASR-Streaming chunk texts -> summarizer transcript lines.

Segmentation is delegated to upstream ``chunk_segments`` (vllm_plugin/asr_streaming.py)
instead of re-implemented, so this side cannot drift from what VibeVoice ships.
The Kotlin port on the phone must match this module line for line.

Output line format (app transcript contract v1):
    [M:SS] S1: text          under one hour
    [H:MM:SS] S1: text       from one hour
    [M:SS] text              speaker unknown ("?")
"""
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

VIBEVOICE_PLUGIN = os.environ.get("VIBEVOICE_PLUGIN", "/home/luigi/vibevoice/vllm_plugin")
if VIBEVOICE_PLUGIN not in sys.path:
    sys.path.insert(0, VIBEVOICE_PLUGIN)

from asr_streaming import ChunkGeometry, chunk_segments  # noqa: E402

# Released streaming checkpoint: 22 frames x 3200 samples / 24 kHz = 2.9333 s per chunk.
STREAMING_1P5B = ChunkGeometry(sample_rate=24000, frame_samples=3200,
                               chunk_frames=22, lookahead_frames=4)

# Non-speech tags such as [Noise] or [Laughter]. Only ASCII-word tags are removed, so a
# bracket that is part of the speech (e.g. a quoted title) survives.
_NON_SPEECH_TAG_RE = re.compile(r"\[[A-Za-z][A-Za-z _-]*\]")
# Standalone hesitation fillers only. 那個 / 就是 often carry meaning and are kept.
_FILLER_RE = re.compile(r"(?:(?<=^)|(?<=[，。！？、\s]))[嗯啊呃]+[，。、]?")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Line:
    start_s: int
    speaker: Optional[str]  # "S1", "S2", ... or None when unknown
    text: str

    def render(self) -> str:
        return f"[{format_ts(self.start_s)}] " + (f"{self.speaker}: " if self.speaker else "") + self.text


def format_ts(seconds: int) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def clean_text(text: str) -> str:
    text = _NON_SPEECH_TAG_RE.sub("", text)
    text = _FILLER_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def segments_to_lines(segments: List[Dict],
                      convert: Callable[[str], str] = lambda s: s) -> List[Line]:
    """Renumber speakers by first appearance, clean text, drop segments left empty."""
    speaker_map: Dict[str, str] = {}
    lines: List[Line] = []
    for seg in segments:
        text = clean_text(convert(seg["Content"]))
        if not text:
            continue
        raw = str(seg.get("Speaker", "?"))
        if raw == "?":
            speaker = None
        else:
            speaker = speaker_map.setdefault(raw, f"S{len(speaker_map) + 1}")
        lines.append(Line(start_s=int(seg["Start"]), speaker=speaker, text=text))
    return lines


def opencc_s2twp() -> Callable[[str], str]:
    import opencc
    conv = opencc.OpenCC("s2twp")
    return conv.convert


def chunks_to_transcript(chunk_texts: List[str],
                         geometry: ChunkGeometry = STREAMING_1P5B,
                         duration: Optional[float] = None,
                         convert: Optional[Callable[[str], str]] = None) -> str:
    segments = chunk_segments(chunk_texts, geometry, duration)
    lines = segments_to_lines(segments, convert or opencc_s2twp())
    return "\n".join(line.render() for line in lines)


def parse_line(raw: str) -> Line:
    """Inverse of Line.render: split on the first '] ' then the first ': '."""
    if not raw.startswith("["):
        raise ValueError(f"not a transcript line: {raw!r}")
    ts, _, rest = raw[1:].partition("] ")
    parts = [int(p) for p in ts.split(":")]
    start = parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]
    speaker, sep, text = rest.partition(": ")
    if sep and re.fullmatch(r"S\d+|[^\s:]{1,20}", speaker):
        return Line(start, speaker, text)
    return Line(start, None, rest)


def resolve_citation(ts: str, lines: List[Line]) -> Optional[int]:
    """Index of the earliest line starting at [ts], or None if the citation is invented."""
    try:
        target = parse_line(f"[{ts}] x").start_s
    except (ValueError, IndexError):
        return None
    for i, line in enumerate(lines):
        if line.start_s == target:
            return i
    return None


_TS_ITEM = r"\d{1,2}(?::\d{2}){1,2}"
_BRACKET = re.compile(r"[\[［]([^\]］]*)[\]］]")


def extract_citations(point: str) -> List[str]:
    """Every timestamp cited in a point, whether written as adjacent brackets `[a][b]` (the
    corpus's own style) or comma-joined inside one bracket `[a, b]` (a style the teacher also
    produces, which a bracket-must-be-exactly-one-timestamp regex used to miss entirely -- a
    point citing only that style would fail 'no valid citation' though every timestamp in it was
    real)."""
    out = []
    for group in _BRACKET.findall(point):
        parts = [p.strip() for p in group.split(",")]
        if parts and all(re.fullmatch(_TS_ITEM, p) for p in parts):
            out.extend(parts)
    return out
