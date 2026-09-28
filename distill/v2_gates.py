"""Automatic gates for corpus v2, built from what corpus v1 needed repaired by hand.

v1 took 12 review rounds and 132 repairs for 39 sessions. Most repairs fell in a few classes that
are cheap to detect mechanically; this module detects them, repairs the one class that is safe to
repair deterministically, and reports the rest so the generator can regenerate.

  asr_gate            before any teacher call: drop sessions whose transcript is not usable.
  auto_repair_names   name substitutions (馬文軍 -> 馬文君): the teacher writes the real person where the
                      transcript is garbled one character away. Repaired to the transcript's spelling.
  check_session       structural validation, organisations absent from the transcript, one case given
                      contradictory adopted outcomes, and a hedge (疑似/可能/推測) lost between a note and
                      the summary point citing it.

Thresholds are calibrated on the 34 v1 sessions that have a reference transcript: every usable
session had a Chinese-character share of at least 0.41 and a bigram recall against the reference of
at least 0.41; the one session with an unusable stretch (ivod_17258) had a share of 0.19.
"""
import collections
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.validate_teacher import check  # noqa: E402
from summarizer.ingest import extract_citations, parse_line  # noqa: E402

HAN = re.compile(r"[一-鿿]")
MIN_HAN_SHARE = 0.35
MIN_BIGRAM_RECALL = 0.35
MAX_REPEAT_SHARE = 0.25
HEDGES = ("疑似", "可能", "推測", "似為")
# 「楊委員（推測）」, 「發言者（疑似健保署官員）」, 「疑似楊曜」: the item a guess is about.
HEDGED_ITEM = re.compile(r"([\u4e00-\u9fff]{2,6})[（(](?:推測|疑似)[）)]|[（(]疑似([\u4e00-\u9fff]{2,8})[）)]|疑似([\u4e00-\u9fff]{2,4})")
SURNAMES = set("陳林黃張李王吳劉蔡楊許鄭謝郭洪曾邱廖賴徐周葉蘇莊呂江何蕭羅高潘簡朱鍾游彭詹胡施沈余"
               "盧梁趙顏柯翁魏孫戴范方宋鄧杜傅侯曹薛丁卓馬阮董唐溫藍蔣石古紀姚連馮歐程湯黎田康白"
               "涂尤巫韓龔嚴袁鐘")
NAMED = re.compile(r"([一-鿿]{3})(?=委員|部長|次長|署長|處長|召委|院長|主委|等\d|等[一二三四五六七八九十]+人)")


def _han_only(text):
    text = re.sub(r"\[[\d:]+\]\s*(S\d+:)?", "", text)
    return "".join(HAN.findall(text))


def asr_gate(transcript_text, reference_text=None):
    """(ok, metrics, reasons). Drops a transcript that is mostly non-speech, looping, or unrelated."""
    lines = [l for l in transcript_text.splitlines() if l.strip()]
    bodies = [re.sub(r"^\[[\d:]+\]\s*(S\d+:)?\s*", "", l) for l in lines]
    han = sum(len(HAN.findall(b)) for b in bodies)
    total = sum(len(b) for b in bodies) or 1
    metrics = {"lines": len(lines), "han_share": han / total}
    counts = collections.Counter(b for b in bodies if len(b) > 4)
    metrics["repeat_share"] = sum(n for n in counts.values() if n > 2) / max(1, len(bodies))
    thirds = [bodies[i * len(bodies) // 3:(i + 1) * len(bodies) // 3] for i in range(3)]
    metrics["worst_third_han"] = min(sum(len(HAN.findall(b)) for b in t) for t in thirds) if bodies else 0
    reasons = []
    if metrics["han_share"] < MIN_HAN_SHARE:
        reasons.append(f"Chinese-character share {metrics['han_share']:.2f} < {MIN_HAN_SHARE}")
    if metrics["repeat_share"] > MAX_REPEAT_SHARE:
        reasons.append(f"repeated lines {metrics['repeat_share']:.2f} > {MAX_REPEAT_SHARE} (ASR loop)")
    if reference_text:
        a, b = _han_only(transcript_text), _han_only(reference_text)
        A = collections.Counter(a[i:i + 2] for i in range(len(a) - 1))
        B = collections.Counter(b[i:i + 2] for i in range(len(b) - 1))
        metrics["bigram_recall"] = sum((A & B).values()) / max(1, sum(B.values()))
        if metrics["bigram_recall"] < MIN_BIGRAM_RECALL:
            reasons.append(f"agreement with reference {metrics['bigram_recall']:.2f} < {MIN_BIGRAM_RECALL}")
    return (not reasons), metrics, reasons


def _surfaces(run):
    for n in run.get("notes") or []:
        yield n["text"]
    yield from run.get("summary") or []
    if run.get("prose"):
        yield run["prose"]


def _near_miss_map(run, transcript_text):
    tr = re.sub(r"\s", "", transcript_text)
    names = collections.Counter(m for m in NAMED.findall(tr) if m[0] in SURNAMES)
    gold = {m for text in _surfaces(run) for m in NAMED.findall(re.sub(r"\s", "", text)) if m[0] in SURNAMES}
    mapping, ambiguous = {}, []
    for g in gold:
        if g in tr:
            continue
        near = [n for n in names if sum(x != y for x, y in zip(g, n)) == 1]
        if len(near) == 1:
            mapping[g] = near[0]
        elif len(near) > 1:
            ambiguous.append((g, near))
    return mapping, ambiguous


def auto_repair_names(run, transcript_text):
    """Replace names one character away from a single transcript name with the transcript's spelling."""
    mapping, ambiguous = _near_miss_map(run, transcript_text)
    if mapping:
        pattern = re.compile("|".join(map(re.escape, sorted(mapping, key=len, reverse=True))))
        sub = lambda t: pattern.sub(lambda m: mapping[m.group(0)], t)   # noqa: E731
        for n in run.get("notes") or []:
            n["text"] = sub(n["text"])
        run["summary"] = [sub(p) for p in run.get("summary") or []]
        if run.get("prose"):
            run["prose"] = sub(run["prose"])
        run.setdefault("repairs", []).extend(
            {"stage": "v2-auto", "kind": "REFERENCE LEAK", "before": g, "after": t,
             "why": f"transcript spells it {t}; {g} occurs nowhere in it"} for g, t in sorted(mapping.items()))
    return mapping, ambiguous


CASE = re.compile(r"第\s*(\d+)\s*案")
AMOUNT = re.compile(r"\d[\d.]*\s*(?:億|萬)")
ADOPTED = re.compile(r"決議|裁示|通過|改凍結|改列|照案|保留")


def check_session(run, transcript_text):
    lines = [parse_line(l) for l in transcript_text.splitlines() if l.strip()]
    problems = [("structure", p) for p in check(run, lines, 0, 650, 0)]

    # One case, two different adopted amounts across notes: the v1 wrong-pairing class.
    adopted = collections.defaultdict(set)
    for n in run.get("notes") or []:
        text = n["text"]
        if not ADOPTED.search(text):
            continue
        for clause in re.split(r"[；;。]", text):
            cases = CASE.findall(clause)
            amounts = {re.sub(r"\s", "", a) for a in AMOUNT.findall(clause)}
            if len(cases) == 1 and amounts and ADOPTED.search(clause):
                adopted[cases[0]].add(frozenset(amounts))
    for case, sets in adopted.items():
        if len(sets) > 1 and not any(a <= b or b <= a for a in sets for b in sets if a is not b):
            problems.append(("contradiction", f"第{case}案 has conflicting adopted amounts: "
                             + " / ".join("、".join(sorted(s)) for s in sets)))

    # A guess in a note stated as fact in the summary point citing it. Checked on the hedged item
    # itself, not the note as a whole: 「發言者（疑似健保署官員）」 hedges who spoke, and a summary
    # point about what was said has nothing to hedge. Only 「X（推測）」, 「（疑似X）」 and 「疑似X」 are
    # guesses about X; a bare 可能 is usually a speaker's own modal verb, and 「應為」 notes an ASR
    # correction. On the reviewed v1 gold the note-level version flagged 11 points, nearly all false.
    hedged_items = collections.defaultdict(set)
    for n in run.get("notes") or []:
        for m in HEDGED_ITEM.finditer(n["text"]):
            item = next(g for g in m.groups() if g)
            # 「疑似語音辨識錯誤」 comments on the transcript, not on a fact the summary could overstate.
            if len(item) >= 2 and not item.startswith(("語音", "辨識", "錯字", "誤植")):
                hedged_items[n["ts"]].add(item)
    for i, point in enumerate(run.get("summary") or [], 1):
        cited = set(extract_citations(point))
        for ts in cited:
            for item in hedged_items.get(ts, ()):
                if item in point and not any(h in point for h in HEDGES):
                    problems.append(("hedge", f"point {i} states 「{item}」 as fact; the note at {ts} marked it a guess"))

    mapping, ambiguous = _near_miss_map(run, transcript_text)
    for g, near in ambiguous:
        problems.append(("name", f"{g} is not in the transcript; near {'/'.join(near)}"))
    return problems
