"""Find names and figures in the gold data that do not occur in the transcript it came from.

The summariser reads only our noisy ASR transcript. So a proper noun or a figure in a gold
summary has to be present in that transcript, garbled or not. When the gold says 監察院 and the
transcript only ever says 行業監督院, the gold is right about the world and wrong as a training
target: no student reading the transcript could produce it, so the row teaches the model to
guess a real-world name it cannot see.

Eight rounds of semantic review missed this whole class, because the corrected value looks
right to a reader -- the error is invisible unless you check the source. It is, however, easy
to check mechanically, which is what this does.

Reported, per session: every CJK proper-noun-like token and every numeric figure that appears
in the notes, summary or prose and nowhere in the transcript. Hits are candidates, not verdicts
-- a summary legitimately writes 四百八十九億 for a transcript's 489億, and compounds names
differently -- so each needs a look. The point is that the list is short enough to look at.
"""
import argparse
import json
import os
import re

# Institutions and titles that a summary may legitimately spell out; checked as substrings too.
# A person named right before a title is the highest-value check: these are the tokens a
# summariser is most tempted to "correct" toward the real world, and a wrong one misattributes
# a statement. Organisations are a closed list of the bodies these meetings involve, because an
# open pattern for 院/部/會 swallows ordinary prose.
NUM = re.compile(r"\d[\d,.]*(?:\u5104|\u842c|\u5343)")
NAME = re.compile(r"(?<![\u4e00-\u9fff])[\u4e00-\u9fff]{2,3}(?=\u7b49\d|\u7b49\u4eba|\u59d4\u54e1|\u90e8\u9577|\u7f72\u9577|\u6b21\u9577|\u8655\u9577|\u53f8\u9577|\u5c40\u9577|\u4e3b\u59d4|\u53ec\u59d4|\u53c3\u4e8b|\u7e3d\u9577)")
ORGS = ("\u76e3\u5bdf\u9662", "\u8003\u8a66\u9662", "\u53f8\u6cd5\u9662", "\u4e2d\u9078\u6703", "\u91d1\u7ba1\u6703",
        "\u50d1\u59d4\u6703", "\u4e3b\u8a08\u7e3d\u8655", "\u5be9\u8a08\u90e8", "\u8ca1\u653f\u90e8", "\u5167\u653f\u90e8",
        "\u6559\u80b2\u90e8", "\u6cd5\u52d9\u90e8", "\u7d93\u6fdf\u90e8", "\u4ea4\u901a\u90e8", "\u8fb2\u696d\u90e8",
        "\u904b\u52d5\u90e8", "\u570b\u9632\u90e8", "\u885b\u798f\u90e8", "\u6587\u5316\u90e8", "\u5916\u4ea4\u90e8",
        "\u570b\u79d1\u6703", "\u571f\u5730\u9280\u884c", "\u9435\u9053\u5c40", "\u904b\u7814\u6240", "\u5de5\u7a0b\u6703",
        "\u570b\u5b89\u5c40")


def normalise(s):
    """The transcript spaces digits from their units (15 萬), the gold does not."""
    return re.sub(r"[\s,\uff0c]+", "", s)


# A summary may abbreviate a body the transcript names in full; that stays derivable.
ABBREV = {"\u885b\u798f\u90e8": "\u885b\u751f\u798f\u5229\u90e8",
          "\u570b\u79d1\u6703": "\u570b\u5bb6\u79d1\u5b78\u53ca\u6280\u8853\u59d4\u54e1",
          "\u91d1\u7ba1\u6703": "\u91d1\u878d\u76e3\u7763\u7ba1\u7406\u59d4\u54e1\u6703",
          "\u50d1\u59d4\u6703": "\u50d1\u52d9\u59d4\u54e1\u6703",
          "\u4e2d\u9078\u6703": "\u4e2d\u592e\u9078\u8209\u59d4\u54e1\u6703"}


def transcript_text(path):
    return normalise(open(path, encoding="utf-8").read())


def gold_texts(run):
    out = [(f"note {n['ts']}", n["text"]) for n in run.get("notes", [])]
    out += [(f"point {i}", p) for i, p in enumerate(run.get("summary", []), 1)]
    if run.get("prose"):
        out.append(("prose", run["prose"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/gold/w4000")
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    sessions = args.only or sorted(f[:-5] for f in os.listdir(args.run_dir) if f.endswith(".json"))
    total = 0
    for sid in sessions:
        run = json.load(open(os.path.join(args.run_dir, sid + ".json"), encoding="utf-8"))
        hay = transcript_text(os.path.join(args.transcripts, sid + ".txt"))
        seen, hits = set(), []
        for where, text in gold_texts(run):
            body = normalise(text)
            for tok in NAME.findall(body) + NUM.findall(body):
                if tok in seen or tok in hay:
                    continue
                seen.add(tok)
                hits.append((tok, where, "name" if NAME.fullmatch(tok) or not any(c.isdigit() for c in tok) else "figure"))
            for org in ORGS:
                if org in body and org not in hay and ABBREV.get(org, org) not in hay \
                        and org not in seen:
                    seen.add(org)
                    hits.append((org, where, "org"))
        if hits:
            total += len(hits)
            print(f"\n{sid}: {len(hits)} token(s) absent from the transcript")
            for tok, where, kind in hits[:40]:
                print(f"   {kind:7} {tok!r:14} first seen in {where}")
    print(f"\n{total} candidate(s) across {len(sessions)} session(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
