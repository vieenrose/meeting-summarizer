"""Training rows for a map-step VERIFY pass: the model rereads a window and checks each note.

The student's remaining errors are systematic misreadings (a real figure pinned to the wrong
article, a clause "passed" that was held, a proposal written as a decision); resampling the same
model does not remove them (eval/notes_filter.py consistency: no effect). A second pass that
looks at one note and the transcript at a time is a different, narrower task.

Rows are synthetic, from the QA'd gold: a window's gold notes with some notes corrupted in exactly
those ways, and the target verdict per note -- OK, or FIX with the gold note restored. Corruptions:

  swap     two notes of the window exchange a figure (the wrong-pairing class)
  status   a status word flips (通過 <-> 保留, 照案通過 <-> 不予處理, 決議 -> 建議, 同意 <-> 反對)
  figure   one figure changes value (a digit, or a factor of 10)

A note is never corrupted in a way that leaves its text unchanged, and a window keeps at least half
its notes intact so that OK stays the common answer.
"""
import argparse
import json
import random
import re

from summarizer.pipeline import parse_turn

FIG = re.compile(r"\d[\d,]*(?:\.\d+)?")
STATUS = [("照案通過", "不予處理"), ("通過", "保留"), ("決議", "建議"), ("同意", "反對"), ("刪減", "凍結")]

PROMPT = """以下是一段會議逐字稿（語音辨識結果，可能有錯字），以及根據它寫的筆記。請逐則查核筆記是否與逐字稿相符，特別注意：數字是否對應到正確的案號或對象、誰主張什麼、結果是通過還是保留、是建議還是決議。

TRANSCRIPT
{window}

NOTES
{notes}

逐則輸出一行，格式：
N. OK
或
N. FIX: 修正後的完整筆記
只輸出這些行。"""


def corrupt(note, others, rng):
    kinds = ["swap", "status", "figure"]
    rng.shuffle(kinds)
    for k in kinds:
        if k == "swap":
            mine = FIG.findall(note)
            pool = [f for o in others for f in FIG.findall(o) if f not in mine]
            if mine and pool:
                a, b = rng.choice(mine), rng.choice(pool)
                return note.replace(a, b, 1), k
        elif k == "status":
            pairs = [(x, y) for x, y in STATUS if x in note] + [(y, x) for x, y in STATUS if y in note]
            if pairs:
                x, y = rng.choice(pairs)
                return note.replace(x, y, 1), k
        elif k == "figure":
            mine = [f for f in FIG.findall(note) if len(f.replace(",", "")) >= 2]
            if mine:
                a = rng.choice(mine)
                d = a.replace(",", "")
                b = d + "0" if rng.random() < 0.5 else d[:-1] + str((int(d[-1]) + rng.randint(1, 8)) % 10)
                if b != d:
                    return note.replace(a, b, 1), k
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="data/train/train_ivod_ali.jsonl")
    ap.add_argument("--out", default="data/train/verify_rows.jsonl")
    ap.add_argument("--rate", type=float, default=0.3, help="share of notes corrupted")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    n_rows = n_fix = 0
    kinds = {}
    with open(args.out, "w", encoding="utf-8") as f:
        for line in open(args.rows, encoding="utf-8"):
            r = json.loads(line)
            if r["kind"] != "notes":
                continue
            window = r["messages"][1]["content"].split("TRANSCRIPT\n", 1)[-1]
            gold = [(tag, ts, t) for tag, ts, t in parse_turn(r["messages"][-1]["content"]).notes]
            if len(gold) < 2:
                continue
            shown, target = [], []
            budget = max(1, int(len(gold) * args.rate)) if rng.random() < 0.85 else 0
            order = list(range(len(gold)))
            rng.shuffle(order)
            bad = set(order[:budget])
            for i, (tag, ts, t) in enumerate(gold):
                line_ = (f"({tag}) " if tag else "") + f"[{ts}] {t}"
                c, k = (corrupt(t, [g[2] for j, g in enumerate(gold) if j != i], rng) if i in bad else (None, None))
                if c and c != t:
                    shown.append((f"({tag}) " if tag else "") + f"[{ts}] {c}")
                    target.append(f"{i + 1}. FIX: {line_}")
                    n_fix += 1
                    kinds[k] = kinds.get(k, 0) + 1
                else:
                    shown.append(line_)
                    target.append(f"{i + 1}. OK")
            prompt = PROMPT.format(window=window, notes="\n".join(f"{i + 1}. {s}" for i, s in enumerate(shown)))
            f.write(json.dumps({"kind": "verify", "session": r["session"], "window": r["window"], "teacher": "synthetic",
                                "messages": [{"role": "user", "content": prompt},
                                             {"role": "assistant", "content": "\n".join(target)}]},
                               ensure_ascii=False) + "\n")
            n_rows += 1
    print(f"{n_rows} verify rows, {n_fix} corrupted notes {kinds} -> {args.out}")


if __name__ == "__main__":
    main()
