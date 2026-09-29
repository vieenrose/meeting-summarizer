"""Mechanical number check for agent notes: every number a note states must be said in the
transcript within WINDOW seconds of the note's own timestamp, otherwise the note is dropped.

Wrong amounts and article numbers are the student's worst error class (the key-figures section is
contradicted at ~25 %), and a 2B model cannot be trusted to verify them itself. Here the harness
does it: numbers are normalised on both sides -- Arabic (1,432 / 1.5 / 45%) and Chinese numerals
with 萬/億 units (一千四百三十二萬, 1.5億, 兩千) -- and compared as values.

  python3 eval/number_check.py --run rt-gemma4-e2b-ft-ep0-v3x --out rt-gemma4-e2b-ft-ep0-v3x-num
rebuilds the minutes of an existing run from the notes that pass, without regenerating anything.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.ingest import parse_line, resolve_citation  # noqa: E402

WINDOW = 90
DIG = {"零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
SMALL = {"十": 10, "百": 100, "千": 1000}
BIG = {"萬": 10 ** 4, "万": 10 ** 4, "億": 10 ** 8, "亿": 10 ** 8}
CN = re.compile(r"[零〇一二兩两三四五六七八九十百千萬万億亿]+")
AR = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*([萬万億亿])?")


def cn_value(s):
    """Chinese numeral -> int (None when not a number, e.g. a lone 萬)."""
    total, section, num = 0, 0, None
    for ch in s:
        if ch in DIG:
            num = DIG[ch]
        elif ch in SMALL:
            section += (num if num is not None else 1) * SMALL[ch]
            num = None
        elif ch in BIG:
            section += num or 0
            if section == 0:
                return None
            total += section * BIG[ch]
            section, num = 0, None
    section += num or 0
    v = total + section
    return v if v > 0 or s in ("零", "〇") else None


def digits_seq(s):
    """一一二 (read digit by digit, as years and article numbers often are) -> 112."""
    if all(c in DIG for c in s) and len(s) > 1:
        return int("".join(str(DIG[c]) for c in s))
    return None


def values(text):
    """Every number stated in text, as floats, with 萬/億 applied."""
    out = set()
    for m in AR.finditer(text):
        v = float(m.group(1).replace(",", ""))
        out.add(v * BIG[m.group(2)] if m.group(2) else v)
    for m in CN.finditer(text):
        s = m.group(0)
        for v in (cn_value(s), digits_seq(s)):
            if v is not None:
                out.add(float(v))
        # a numeral followed by a unit and more numerals: also keep the leading part (三千萬 -> 3e7)
    return out


def meaningful(text):
    """Numbers worth checking: skip 1-digit counts (一位, 三案) that ASR renders every which way."""
    return {v for v in values(text) if v >= 10}


def note_ok(note, lines):
    need = meaningful(re.sub(r"[\[［]\d+:\d{2}(?::\d{2})?[\]］]", "", note["text"]))
    if not need:
        return True
    i = resolve_citation(note["ts"], lines)
    if i is None:
        return False
    t0 = lines[i].start_s
    near = "".join(l.text for l in lines if -30 <= l.start_s - t0 <= WINDOW)
    have = values(near)
    return all(any(abs(v - h) <= 1e-6 * max(1.0, v) for h in have) for v in need)


SECTIONS = {"決議事項": ["DECISION"], "待辦與負責人": ["ACTION"], "保留與未決": ["OPEN-ISSUE"], "重要數字": ["NUMBER"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir under runs/student")
    ap.add_argument("--out", required=True)
    ap.add_argument("--transcripts", default="data/v2/transcripts")
    a = ap.parse_args()
    os.makedirs(f"runs/student/{a.out}", exist_ok=True)
    kept = dropped = 0
    for f in sorted(glob.glob(f"runs/student/{a.run}/ivod_*.json")):
        sid = os.path.basename(f)[:-5]
        r = json.load(open(f, encoding="utf-8"))
        lines = [parse_line(l) for l in open(os.path.join(a.transcripts, sid + ".txt"), encoding="utf-8")
                 .read().splitlines() if l.strip()]
        ok = [n for n in r["notes"] if note_ok(n, lines)]
        kept, dropped = kept + len(ok), dropped + len(r["notes"]) - len(ok)
        out = []
        for title, tags in SECTIONS.items():
            items = [n for n in ok if (n["tag"] or "").upper() in tags]
            out += [f"【{title}】"] + ([f"- {n['text'].rstrip('。')} [{n['ts']}]" for n in items] or ["- 無"])
        r["notes"], r["minutes"] = ok, "\n".join(out)
        items = [l[2:].strip() for l in out if l.startswith("- ") and l[2:].strip() != "無"]
        r["prose"] = "".join(x if x.endswith("。") else x + "。" for x in items)
        r["number_check"] = {"dropped": len(r["notes"]) - len(ok)}
        json.dump(r, open(f"runs/student/{a.out}/{sid}.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{a.run} -> {a.out}: kept {kept} notes, dropped {dropped} ({dropped / max(1, kept + dropped):.0%})")


if __name__ == "__main__":
    main()
