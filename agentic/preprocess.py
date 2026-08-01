"""Transcript compaction — the cheapest way to cut prefill time, which dominates on ARM.

ASR meeting transcripts are highly redundant: backchannels ("嗯", "對對對", "uh-huh"),
fillers, and one-utterance-per-VAD-segment fragmentation that splits a single sentence
across several timestamped lines. None of it carries information a summary needs, and all
of it is prefilled at ~140 tok/s on a Boox.

Everything here is deterministic string work — no model, no cost — and it is directly
portable to Kotlin.

Rules, in order:
1. drop pure backchannel/filler utterances
2. strip inline fillers
3. merge consecutive utterances from the same speaker (keeping the FIRST timestamp, so
   anchors still point at where the thought started)
4. collapse ASR stutter repeats

Timestamps and speaker tags are preserved, so transcript-format v1 still holds and every
anchor the model emits remains resolvable.
"""
import re

# Pure-backchannel lines: the whole utterance is acknowledgement, no content.
BACKCHANNEL_ZH = re.compile(r"^[嗯欸誒喔噢哦啊呃唉哎呀吧嘛耶阿はい,，。、!！?？~～\s]*$|"
                            r"^(對|對對|對對對|好|好的|好好好|是|是的|沒錯|了解|OK|ok)[。，,\s!！]*$")
BACKCHANNEL_EN = re.compile(r"^(uh|um|erm|mm|mhm|hmm|uh-huh|yeah|yep|yes|ok|okay|right|sure|"
                            r"got it|i see|exactly|true)[\s,.!?]*$", re.I)

# Inline fillers: removed mid-sentence without changing meaning.
FILLER_EN = re.compile(r"\b(you know|i mean|sort of|kind of|like,|um|uh|erm)\b[,]?\s*", re.I)
FILLER_ZH = re.compile(r"(那個|這個|就是說|然後呢|對不對|你知道嗎)(?=[，,、\s])")

# ASR stutter: "the the", "我我我", "we we"
STUTTER_EN = re.compile(r"\b(\w+)( \1\b)+", re.I)
STUTTER_ZH = re.compile(r"([一-鿿])\1{2,}")

LINE = re.compile(r"^\[(\d+:\d{2}(?::\d{2})?)\]\s*(?:([^:\n]{1,40}):\s*)?(.*)$")


def _clean(text: str, zh: bool) -> str:
    t = STUTTER_ZH.sub(r"\1", text) if zh else STUTTER_EN.sub(r"\1", text)
    t = FILLER_ZH.sub("", t) if zh else FILLER_EN.sub("", t)
    return re.sub(r"\s{2,}", " ", t).strip(" ,")


def compact(transcript: str, zh: bool = False, merge_speakers: bool = True) -> str:
    """Return a shorter transcript in the same format. Idempotent."""
    out = []
    for line in transcript.split("\n"):
        m = LINE.match(line.strip())
        if not m:
            if line.strip():
                out.append((None, None, line.strip()))
            continue
        ts, spk, text = m.group(1), m.group(2), m.group(3)
        bc = BACKCHANNEL_ZH if zh else BACKCHANNEL_EN
        if not text.strip() or bc.match(text.strip()):
            continue                      # rule 1
        text = _clean(text, zh)           # rules 2 + 4
        if not text:
            continue
        if (merge_speakers and out and out[-1][1] is not None and out[-1][1] == spk):
            ts0, spk0, prev = out[-1]     # rule 3: keep the first timestamp
            joiner = "" if zh else " "
            out[-1] = (ts0, spk0, f"{prev}{joiner}{text}")
        else:
            out.append((ts, spk, text))
    lines = []
    for ts, spk, text in out:
        if ts is None:
            lines.append(text)
        elif spk:
            lines.append(f"[{ts}] {spk}: {text}")
        else:
            lines.append(f"[{ts}] {text}")
    return "\n".join(lines)
