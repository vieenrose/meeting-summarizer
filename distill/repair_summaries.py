"""Targeted repair of gold summaries that fail validation.

Each failure class is repaired by its actual cause, not by re-rolling the whole session:

  over-long point   the teacher's own sentence is split at its semicolon boundaries and the
                    citations distributed in order. Pure reformatting -- no new wording.
  missing third,    a point is built from a note the teacher already wrote inside the uncovered
  notes available   range, so the text stays the teacher's.
  missing third,    nothing to build from; reported for a notes-level re-run instead of invented.
  no notes

Nothing here writes new summary prose. A gap that cannot be filled from teacher output is left
failing and named, because a dataset that hides its holes is worse than one that lists them.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.ingest import parse_line, resolve_citation  # noqa: E402

CITE = re.compile(r"[\[［](\d{1,2}(?::\d{2}){1,2})[\]］]")
NUM = re.compile(r"^\s*\d+[.、]\s*")


def body_len(point):
    return len(re.sub(r"[\[［][^\]］]*[\]］]|\s", "", point))


def split_long(point, limit):
    """Split one over-long point at semicolons, dealing citations out in order."""
    cites = CITE.findall(point)
    text = NUM.sub("", CITE.sub("", point)).strip().rstrip("。")
    clauses = [c.strip() for c in re.split(r"[；;]", text) if c.strip()]
    if len(clauses) < 2:
        return None
    # Greedily pack clauses so every produced point is under the limit.
    groups, cur = [], []
    for c in clauses:
        trial = cur + [c]
        if cur and len("；".join(trial)) > limit:
            groups.append(cur)
            cur = [c]
        else:
            cur = trial
    if cur:
        groups.append(cur)
    if len(groups) < 2:
        # Greedy packing measures the joined clauses, which runs a few characters short of
        # body_len (it counts the numbering and trailing stop too). A point just over the limit
        # therefore packs into one group and would go unrepaired, so split at the clause
        # boundary nearest the middle instead.
        if len(clauses) < 2:
            return None
        half = len(text) // 2
        cut, best = 1, None
        run = 0
        for k, c in enumerate(clauses[:-1]):
            run += len(c) + 1
            if best is None or abs(run - half) < best:
                best, cut = abs(run - half), k + 1
        groups = [clauses[:cut], clauses[cut:]]
    per = max(1, len(cites) // len(groups))
    out = []
    for i, g in enumerate(groups):
        take = cites[i * per:(i + 1) * per] if i < len(groups) - 1 else cites[i * per:]
        marks = "".join(f"[{c}]" for c in take) or (f"[{cites[0]}]" if cites else "")
        out.append("；".join(g).rstrip("。") + "。" + marks)
    return out


def thirds_covered(points, lines):
    n = max(1, len(lines))
    got = set()
    for p in points:
        for c in CITE.findall(p):
            idx = resolve_citation(c, lines)
            if idx is not None:
                got.add(min(2, 3 * idx // n))
    return got


def note_for_third(notes, lines, third):
    """Best teacher note inside the uncovered third: a tagged one if any, else the longest."""
    n = max(1, len(lines))
    lo, hi = third * n // 3, (third + 1) * n // 3
    cands = []
    for nt in notes:
        idx = resolve_citation(nt["ts"], lines)
        if idx is not None and lo <= idx < hi:
            cands.append(nt)
    if not cands:
        return None
    tagged = [c for c in cands if c.get("tag") in ("DECISION", "ACTION")]
    pool = tagged or cands
    return max(pool, key=lambda c: len(c["text"]))


def repair(path, transcripts, max_point, max_total):
    run = json.load(open(path, encoding="utf-8"))
    stem = os.path.splitext(os.path.basename(path))[0]
    lines = [parse_line(l) for l in
             open(os.path.join(transcripts, f"{stem}.txt"), encoding="utf-8").read().splitlines()
             if l.strip()]
    points = list(run["summary"])
    actions = []

    # 1. over-long points
    changed = True
    while changed:
        changed = False
        for i, p in enumerate(points):
            if body_len(p) > max_point:
                parts = split_long(p, max_point)
                if parts:
                    points[i:i + 1] = parts
                    actions.append(f"split point {i+1} ({body_len(p)} chars) into {len(parts)}")
                    changed = True
                    break

    # 2. uncovered thirds, filled only from notes the teacher already wrote
    for third in sorted({0, 1, 2} - thirds_covered(points, lines)):
        nt = note_for_third(run["notes"], lines, third)
        name = {0: "前段", 1: "中段", 2: "後段"}[third]
        if not nt:
            actions.append(f"UNFIXABLE: no teacher note exists in the {name} third")
            continue
        text = nt["text"].strip().rstrip("。")
        if len(text) > max_point:
            text = text[:max_point - 1].rstrip("，、；") + "…"
        point = f"{text}。[{nt['ts']}]"
        at = 0
        for j, p in enumerate(points):
            idxs = [resolve_citation(c, lines) for c in CITE.findall(p)]
            idxs = [i for i in idxs if i is not None]
            if idxs and min(idxs) < third * len(lines) // 3:
                at = j + 1
        points.insert(at, point)
        actions.append(f"inserted a {name} point from the teacher's note at [{nt['ts']}]")

    # 3. total overrun: trim the longest point at a clause boundary rather than drop a point.
    #    Dropping loses a distinct claim; trimming loses only the tail of an enumeration.
    guard = 0
    while sum(body_len(p) for p in points) > max_total and guard < 20:
        guard += 1
        i = max(range(len(points)), key=lambda k: body_len(points[k]))
        p = points[i]
        cites = "".join(f"[{c}]" for c in CITE.findall(p))
        text = NUM.sub("", CITE.sub("", p)).strip().rstrip("。")
        cut = max(text.rfind("；"), text.rfind("，"))
        if cut < len(text) // 2:
            break                      # no sensible boundary left; stop rather than mangle
        before = body_len(p)
        points[i] = text[:cut].rstrip("，、；") + "。" + cites
        actions.append(f"trimmed point {i+1} from {before} to {body_len(points[i])} chars")

    points = [NUM.sub("", p).strip() for p in points]
    points = [f"{i}. {p}" for i, p in enumerate(points, 1)]
    total = sum(body_len(p) for p in points)
    if total > max_total:
        actions.append(f"WARNING: total {total} chars still over {max_total}")
    run["summary"] = points
    run.setdefault("repairs", []).extend(actions)
    return run, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/gold/w4000")
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--max-point", type=int, default=130)
    ap.add_argument("--max-total", type=int, default=620)
    ap.add_argument("--apply", action="store_true", help="write changes back")
    args = ap.parse_args()

    for s in args.sessions:
        path = os.path.join(args.run_dir, f"{s}.json")
        run, actions = repair(path, args.transcripts, args.max_point, args.max_total)
        print(f"=== {s}")
        for a in actions:
            print(f"   {a}")
        for p in run["summary"]:
            print(f"   {body_len(p):3d} | {p[:96]}")
        if args.apply:
            json.dump(run, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print("   written")


if __name__ == "__main__":
    main()
