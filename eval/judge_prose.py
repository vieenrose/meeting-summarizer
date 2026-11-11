"""Judge a student's prose abstract against the QA'd gold for the same held-out session.

The gold notes are the evidence and the gold numbered points are the key facts. Each is checked
once per session, and two numbers come out of it:

  coverage   how many gold points the prose conveys (0..1)
  errors     claims in the prose that contradict the gold notes (an inversion: wrong outcome, wrong
             figure, wrong person) or that the notes nowhere support (a fabrication)

The judge sees the notes rather than the transcript. A 2-3 hour transcript does not fit the judge's
context, and the notes are the QA'd record of what the transcript says. A student may state a real
fact the gold notes left out; the judge is told that "not in the notes" alone is not a
contradiction, and only counts a claim the notes make implausible.
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.pipeline import ChatClient  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from haiku_chat import client  # noqa: E402

PROMPT = """你是嚴格的會議摘要審查員。以下是一場會議的完整筆記（已人工校對，視為事實依據）、該會議的關鍵要點，以及一份待審的會議紀要。

## 筆記
{notes}

## 關鍵要點
{points}

## 待審紀要
{prose}

請逐項判斷：
1. 每一個關鍵要點，待審紀要是否有傳達其核心內容（決議、數字、負責人或爭點）。
2. 待審紀要中的每一個事實陳述，是否與筆記相矛盾（結果相反、數字錯誤、張冠李戴），或筆記中完全找不到依據且看起來是捏造。筆記未提及但合理的細節不算錯誤；只列出明確有問題的陳述。

只輸出 JSON：
{{"covered": [已傳達的要點編號], "errors": [{{"claim": "紀要原文片段", "type": "contradiction 或 fabrication", "why": "簡短理由"}}]}}"""


UNITS_PROMPT = """你是嚴格的會議摘要審查員。以下是一場會議的完整筆記（已人工校對，視為事實依據）、該會議的關鍵要點，以及一份待審的會議紀要（已逐句編號）。

## 筆記
{notes}

## 關鍵要點
{points}

## 待審紀要
{prose}

請逐項判斷：
1. 每一個關鍵要點，待審紀要是否有傳達其核心內容；有的話，列出傳達它的句子編號，並判斷細節是否完整正確（"exact": 要點中的具體數字、名稱、對象、負責人都正確寫出為 true；只提到主題或細節缺漏、錯誤為 false）。
2. 待審紀要中的每一個事實陳述，是否與筆記相矛盾（結果相反、數字錯誤、張冠李戴），或筆記中完全找不到依據且看起來是捏造。筆記未提及但合理的細節不算錯誤；只列出明確有問題的陳述。
3. 哪些句子只是重複前面某一句已經說過的內容（沒有新資訊），列出 [後出現的句子編號, 被重複的句子編號]。沒有傳達的要點不要列在 covered 裡。

只輸出 JSON：
{{"covered": {{"要點編號": {{"units": [句子編號], "exact": true 或 false}}}}, "errors": [{{"claim": "紀要原文片段", "type": "contradiction 或 fabrication", "why": "簡短理由"}}], "repeats": [[後句編號, 前句編號]]}}"""


LIST_HEAD = "重點清單"      # distill/convert_targets.py --dense-list: the overview, then this header and the list


def units(prose: str) -> list:
    """The candidate cut into judged units: sentences, and list lines (which carry no 。)."""
    return [u.strip() for u in re.split(r"(?<=[。！？])|\n", prose)
            if len(re.sub(r"\[[\d:]+\]", "", u).strip()) >= 4 and u.strip() != LIST_HEAD]


def units_blocks(prose: str) -> list:
    """For each unit, 0 if it belongs to the overview and 1 if to the dense list (after LIST_HEAD): an overview
    restating its own list is by design, so eval/summary_metrics.py counts repeats within a block only."""
    head, _, rest = prose.partition(LIST_HEAD)
    return [0] * len(units(head)) + [1] * len(units(rest))


def salvage(reply: str) -> dict:
    """A long verdict can be malformed or cut after "covered", and temperature 0 repeats it: keep the parts that still decode."""
    m = re.search(r'"covered"\s*:\s*', reply)
    if not m:
        return {}
    try:
        covered = json.JSONDecoder().raw_decode(reply, m.end())[0]
    except json.JSONDecodeError:
        return {}
    errors = [{"claim": c, "type": t} for c, t in re.findall(r'"claim"\s*:\s*"([^"]*)"\s*,\s*"type"\s*:\s*"([^"]*)"', reply)]
    return {"covered": covered, "errors": errors, "repeats": [], "salvaged": True}


def parse(reply: str) -> dict:
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return salvage(reply)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="student run dir (one JSON per session)")
    ap.add_argument("--gold", default="runs/v2/w4000")
    ap.add_argument("--split", default="data/split_v2.json")
    ap.add_argument("--judge-url", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--units", action="store_true", help="number the candidate's sentences/lines: per point, the "
                    "units that convey it and whether its details are exact, plus repeated units (eval/summary_metrics.py)")
    args = ap.parse_args()
    chat = client(args.judge_url, args.judge_model, 6000)
    sessions = json.load(open(args.split))["heldout"]

    def one(sid):
        path = os.path.join(args.candidate, sid + ".json")
        if not os.path.exists(path):
            return None
        prose = json.load(open(path, encoding="utf-8")).get("prose", "")
        gold = json.load(open(os.path.join(args.gold, sid + ".json"), encoding="utf-8"))
        points = [re.sub(r"^\d+\.\s*", "", p) for p in gold["summary"]]
        if not prose.strip():
            return {"id": sid, "coverage": 0.0, "errors": [], "n_points": len(points), "empty": True}
        notes = "\n".join(f"[{n['ts']}] {n['text']}" for n in gold["notes"])
        if args.units:
            us = units(prose)
            user = UNITS_PROMPT.format(notes=notes, prose="\n".join(f"{k}. {u}" for k, u in enumerate(us, 1)),
                                       points="\n".join(f"{i}. {p}" for i, p in enumerate(points, 1)))
            for _ in range(3):
                v = parse(chat([{"role": "user", "content": user}]))
                if isinstance(v.get("covered"), dict):
                    break
            cov = v.get("covered") if isinstance(v.get("covered"), dict) else {}
            where, exact = {}, set()
            for i, c in cov.items():
                ws = sorted({int(k) for k in (c.get("units") or []) if str(k).isdigit() and 1 <= int(k) <= len(us)}) \
                    if isinstance(c, dict) else []
                if str(i).isdigit() and 1 <= int(i) <= len(points) and ws:      # a point listed with no unit is not conveyed
                    where[int(i)] = ws
                    if c.get("exact") is True:
                        exact.add(int(i))
            ok = lambda k: str(k).isdigit() and 1 <= int(k) <= len(us)
            reps = sorted({(int(r[0]), int(r[1])) for r in v.get("repeats", []) if isinstance(r, list) and len(r) == 2
                           and ok(r[0]) and ok(r[1])})
            return {"id": sid, "coverage": len(where) / max(1, len(points)), "covered": sorted(where),
                    "exact": sorted(exact), "where": {str(k): w for k, w in where.items()}, "units": us, "blocks": units_blocks(prose),
                    "repeats": reps, "errors": v.get("errors", []), "n_points": len(points),
                    "parsed": isinstance(v.get("covered"), dict), "salvaged": bool(v.get("salvaged"))}
        user = PROMPT.format(notes=notes, prose=prose,
                             points="\n".join(f"{i}. {p}" for i, p in enumerate(points, 1)))
        for _ in range(3):
            v = parse(chat([{"role": "user", "content": user}]))
            if "covered" in v:
                break
        covered = {int(i) for i in v.get("covered", []) if str(i).isdigit() and 1 <= int(i) <= len(points)}
        return {"id": sid, "coverage": len(covered) / max(1, len(points)), "covered": sorted(covered),
                "errors": v.get("errors", []), "n_points": len(points), "parsed": "covered" in v}

    rows = [r for r in ThreadPoolExecutor(args.parallel).map(one, sessions) if r]
    n = len(rows)
    cov = sum(r["coverage"] for r in rows) / n
    err = sum(len(r["errors"]) for r in rows) / n
    contra = sum(any(e.get("type") == "contradiction" for e in r["errors"]) for r in rows) / n
    print(f"{args.candidate}: n={n} coverage {cov:.2f}  errors/session {err:.2f}  "
          f"sessions with a contradiction {contra:.0%}  unparsed {sum(not r.get('parsed', True) for r in rows)}")
    json.dump({"candidate": args.candidate, "n": n, "coverage": cov, "errors_per_session": err,
               "contradiction_rate": contra, "rows": rows},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
