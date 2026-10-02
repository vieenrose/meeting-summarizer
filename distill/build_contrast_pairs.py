"""Contrastive pairs aimed at the students' three dominant errors (eval/contradiction_types.py):
binding (the right fact on the wrong object), attribution (the wrong actor) and outcome (an
inverted result). In the style of CLIFF / ConFiT: the chosen reply is the teacher's reply for a
reading window; the rejected reply is the same with ONE note perturbed, so the pair differs in a
single span and DPO's signal falls on that span.

  binding      swap a figure, year, article or case number in the note for another of the same
               kind that the window also contains
  attribution  swap a body or role in the note for another one the window also names
  outcome      invert the note's result with a lexicon (通過 <-> 保留, 增加 <-> 減少, ...), or write a
               proposal as a decision

Prompts are the GRPO reading pool (distill/rl_prompts.py: training sessions, v5 prompt, compacted
journal before the window); the teacher's notes for the window come with each prompt ("ref").
"""
import argparse
import collections
import json
import random
import re

# Numbers are matched whole (no digit on either side), and a swap keeps the kind, the unit and
# the digit count: 2004 年 -> 2015 年, 1543 萬元 -> 2860 萬元, never 2004 年 -> 15 年 or 萬元 -> %.
NUM = [re.compile(p) for p in (r"第\s*\d+\s*條(?:之\s*\d+)?", r"第\s*\d+\s*案", r"(?<!\d)\d{3,4}\s*年度?",
                                r"(?<![\d.])\d+(?:\.\d+)?\s*(?:億|萬|千|百)?\s*(?:元|%|％|人|件|億|萬)")]
# A closed list of bodies and roles (an open "...會" pattern matched 院會, 散會, 晚會).
BODIES = ("行政院 立法院 司法院 監察院 考試院 內政部 外交部 國防部 財政部 教育部 法務部 經濟部 交通部 勞動部 "
          "衛福部 衛生福利部 農業部 環境部 文化部 數位發展部 數位部 國發會 金管會 原民會 客委會 海委會 僑委會 "
          "退輔會 國科會 公平會 通傳會 中選會 陸委會 主計總處 人事總處 審計部 海巡署 健保署 疾管署 食藥署 "
          "移民署 警政署 消防署 關務署 國稅局 公路局 鐵道局 民航局 觀光署 林業署 水利署 氣象署 能源署 "
          "國家安全局 國安局 調查局 故宮 中央銀行 央行 台電 中油 臺鐵").split()
ROLES = "主席 召委 部長 次長 署長 局長 院長 主委 處長 司長 秘書長 總經理 董事長 校長".split()
ACTOR = re.compile("|".join(sorted(map(re.escape, BODIES + ROLES), key=len, reverse=True)))


def shape(x):
    """What a swap must keep: the non-digit part (unit, 第…條) and the digit count."""
    return re.sub(r"\d", "0", re.sub(r"\s", "", x))
FLIP = [("通過", "保留"), ("同意", "反對"), ("支持", "反對"), ("增加", "減少"), ("提高", "降低"),
        ("上升", "下降"), ("擴大", "縮減"), ("已", "未"), ("可以", "不得"), ("應", "不應"), ("照案", "暫緩")]


def render(n):
    return f"NOTE [{n['ts']}] ({n['tag'] or '-'}) {n['text']}" if n.get("tag") else f"NOTE [{n['ts']}] {n['text']}"


def reply(notes):
    return "\n".join(map(render, notes)) + "\nNEXT"


def perturb(kind, note, window, rng):
    text = note["text"]
    if kind == "binding":
        for pat in rng.sample(NUM, len(NUM)):
            here = pat.findall(text)
            if not here:
                continue
            pick = rng.choice(here)
            norm = lambda x: re.sub(r"\s", "", x)  # noqa: E731
            pool = sorted({x for x in pat.findall(window) if shape(x) == shape(pick) and norm(x) not in map(norm, here)})
            if pool:
                new = rng.choice(pool)   # keep the note's own spacing, or the space itself gives the swap away
                return {**note, "text": text.replace(pick, new if re.search(r"\s", pick) else re.sub(r"\s", "", new), 1)}
    elif kind == "attribution":
        here = ACTOR.findall(text)
        if here:
            pick = rng.choice(here)
            same = BODIES if pick in BODIES else ROLES   # a body for a body, a role for a role
            pool = sorted({x for x in ACTOR.findall(window) if x in same} - set(here))
            if pool:
                return {**note, "text": text.replace(pick, rng.choice(pool), 1)}
    elif kind == "outcome":
        if (note.get("tag") or "").upper() == "PROPOSAL" and "建議" in text and rng.random() < 0.5:
            return {**note, "tag": "DECISION", "text": text.replace("建議", "決定", 1)}
        opts = [(a, b) for a, b in FLIP if a in text] + [(b, a) for a, b in FLIP if b in text]
        if opts:
            a, b = rng.choice(opts)
            return {**note, "text": text.replace(a, b, 1)}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/train/rl_prompts.jsonl")
    ap.add_argument("--out", default="data/train/contrast_pairs.jsonl")
    ap.add_argument("--per-window", type=int, default=2)
    ap.add_argument("--max-per-kind", type=int, default=900)
    a = ap.parse_args()
    rng = random.Random(0)
    stats = collections.Counter()
    with open(a.out, "w", encoding="utf-8") as fo:
        for line in open(a.prompts, encoding="utf-8"):
            r = json.loads(line)
            if r["task"] != "read" or not r["ref"]:
                continue
            notes = [{"ts": n["ts"], "tag": n.get("tag"), "text": n["text"]} for n in r["ref"]]
            chosen = reply(notes)
            made = 0
            for kind in rng.sample(["binding", "attribution", "outcome"], 3):
                if made >= a.per_window:
                    break
                for i in rng.sample(range(len(notes)), len(notes)):
                    if stats[kind] >= a.max_per_kind:
                        break
                    bad = perturb(kind, notes[i], r["window_text"], rng)
                    if bad and bad != notes[i]:
                        rejected = reply(notes[:i] + [bad] + notes[i + 1:])
                        fo.write(json.dumps({"session": r["session"], "window": r["window"], "kind": kind,
                                             "messages": r["messages"], "chosen": chosen, "rejected": rejected,
                                             "before": notes[i]["text"], "after": bad["text"]}, ensure_ascii=False) + "\n")
                        stats[kind] += 1
                        made += 1
                        break
    print(dict(stats))


if __name__ == "__main__":
    main()
