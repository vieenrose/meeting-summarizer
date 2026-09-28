"""Compare figures by value, whatever the script they are written in.

The ASR writes numbers in Chinese (一億四千六百一十八萬二千元, 百分之二十), and the notes
normally write them in Arabic digits (1億4618萬2000元, 20%). A lexical match between the two
fails exactly when a note normalises correctly, and succeeds when it copies the transcript
verbatim. So every figure is reduced to (value, kind) before comparing: kind is "%" for
percentages and "" otherwise.
"""
import re

DIG = {"零": 0, "〇": 0, "一": 1, "壹": 1, "二": 2, "兩": 2, "两": 2, "三": 3, "四": 4, "五": 5,
       "六": 6, "七": 7, "八": 8, "九": 9}
SMALL = {"十": 10, "百": 100, "千": 1000}
BIG = {"萬": 10 ** 4, "万": 10 ** 4, "億": 10 ** 8, "亿": 10 ** 8}
CN = "零〇一壹二兩两三四五六七八九十百千萬万億亿"
# A digit run may carry Chinese big units (1億4618萬2000); a Chinese run is any numeral string.
NUM = re.compile(rf"(?:\d[\d,]*(?:\.\d+)?[萬万億亿]?)+(?:\d[\d,]*(?:\.\d+)?)?|[{CN}]+")


def _cn(s):
    total = section = num = 0
    for ch in s:
        if ch in DIG:
            num = DIG[ch]
        elif ch in SMALL:
            section += (num or 1) * SMALL[ch]
            num = 0
        elif ch in BIG:
            total += (section + num) * BIG[ch]
            section = num = 0
    return total + section + num


def _mixed(s):
    """'1億4618萬2000' or '3.5萬' -> float value."""
    total, cur = 0.0, ""
    for ch in s.replace(",", ""):
        if ch in BIG:
            total += float(cur or 1) * BIG[ch]
            cur = ""
        else:
            cur += ch
    return total + (float(cur) if cur else 0.0)


def values(text):
    """Set of (value, kind) for every figure worth checking in text (values below 10 are too
    common to carry information: 一 and 兩 are everyday words)."""
    out = set()
    for m in NUM.finditer(text):
        s = m.group(0)
        if s == "百" and text[m.end():m.end() + 2] == "分之":
            continue
        if s[0].isdigit():
            v = _mixed(s)
        else:
            if len(s) == 1 and s not in "十百千萬万億亿":
                continue
            v = _cn(s)
        before, after = text[max(0, m.start() - 3):m.start()], text[m.end():m.end() + 1]
        kind = "%" if after in ("%", "％") or before.endswith("百分之") else ""
        if v >= 10 or kind:
            out.add((round(v, 4), kind))
    return out
