"""Replace real-world names and figures that the transcript never says.

Eight rounds of semantic review looked for claims contradicted by the transcript. This is the
opposite defect: claims the transcript never makes at all, which are true of the world. The
teacher recognised 馬文軍 as 馬文君, 行業監督院 as 監察院, 李陽 as 李洋, and wrote the real
name. A reader checking the summary against reality would approve every one.

They are still wrong as training targets. The student reads only our noisy transcript, so a row
that maps 馬文軍 to 馬文君 teaches it to supply a name it cannot see -- to guess a plausible
public figure whenever the audio is unclear. That is the hallucination this project exists to
avoid, taught deliberately.

So each is replaced by what our transcript actually says, everywhere it occurs, across notes,
summary and prose. eval/check_derivable.py finds them mechanically; this fixes them in bulk,
because they repeat within a session.
"""
import argparse
import json
import os

# session -> (gold spelling, our transcript's spelling, why)
LEAKS = {
    "ivod_14064": [("費鴻泰", "費永泰",
                    "逐字稿作「費永泰委員」，全篇無「費鴻泰」。")],
    "ivod_16786": [("黃捷委員（黃傑委員）", "黃傑委員",
                    "逐字稿只作「黃傑」6 次，摘要卻把真名擺在前、辨識結果放進括號。"),
                   ("黃捷委員", "黃傑委員", "同上。")],
    "ivod_17211": [("\u99ac\u6587\u541b\u59d4\u54e1", "\u99ac\u662d\u57fa\u59d4\u54e1",
                    "\u9010\u5b57\u7a3f\u53ea\u4f5c\u300c\u4e3b\u5e2d\u99ac\u662d\u57fa\u59d4\u54e1\u6587\u541b\u300d\uff08\u540d\u5b57\u8fa8\u8b58\u4e0d\u6e05\uff09\uff0c\u5168\u7bc7\u7121\u300c\u99ac\u6587\u541b\u300d\u3002")],
    "ivod_17393": [("主席（莊瑞雄委員）", "主席",
                    "逐字稿全篇未出現「莊瑞雄」，主席姓名無從得知。")],
    "ivod_17421": [("監察院", "行業監督院",
                    "逐字稿作「行業監督院」，全篇無「監察院」。")],
    "ivod_17507": [("48.9億", "489億",
                    "逐字稿作「489 億」，小數點是摘要自行加上的，差了一個數量級。")],
    "ivod_17529": [("馬文君", "馬文軍",
                    "逐字稿作「馬文軍」22 次，全篇無「馬文君」。")],
    "ivod_17627": [("\u9283\u6558\u90e8", "\u5168\u7dd2\u90e8",
                    "\u9010\u5b57\u7a3f\u4f5c\u300c\u5168\u7dd2\u90e8\u300d\uff0c\u5168\u7bc7\u7121\u300c\u9283\u6558\u90e8\u300d\u3002"),
                   ("\u6aa2\u5bdf\u53f8\u53f8\u9577", "\u6aa2\u5bdf\u5e2b\u5e2b\u9577",
                    "\u9010\u5b57\u7a3f\u4f5c\u300c\u6aa2\u5bdf\u5e2b\u5e2b\u9577\u300d\u3002")],
    "ivod_17698": [("李洋", "李陽",
                    "逐字稿作「李陽」3 次、「李安」2 次，全篇無「李洋」。")],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/gold/w4000")
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for sid, pairs in sorted(LEAKS.items()):
        path = os.path.join(args.run_dir, sid + ".json")
        doc = json.load(open(path, encoding="utf-8"))
        hay = open(os.path.join(args.transcripts, sid + ".txt"), encoding="utf-8").read()
        for gold, ours, why in pairs:
            bare = ours.replace("委員", "")
            if bare and bare not in hay.replace(" ", ""):
                print(f"SKIP {sid}: {ours!r} is not in the transcript either")
                continue
            n = 0
            for note in doc["notes"]:
                if gold in note["text"]:
                    note["text"] = note["text"].replace(gold, ours)
                    n += 1
            for i, point in enumerate(doc["summary"]):
                if gold in point:
                    doc["summary"][i] = point.replace(gold, ours)
                    n += 1
            if doc.get("prose") and gold in doc["prose"]:
                doc["prose"] = doc["prose"].replace(gold, ours)
                n += 1
            print(f"{sid}: {gold!r} -> {ours!r} in {n} place(s)")
            if not args.dry_run and n:
                doc.setdefault("repairs", []).append({
                    "stage": "derivability", "point": None, "kind": "REFERENCE LEAK",
                    "before": gold, "after": ours, "why": why,
                })
        if not args.dry_run:
            json.dump(doc, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
