"""2026-11-11 questionnaire score of a summary (see qa_build.py): an answerer (q38) sees ONLY the summary and answers every question or says it
cannot; a second call grades each answer against the reference. Reports correct / wrong / no-answer shares, by third of the meeting, and correct per
1000 chars. Usage: qa_score.py QA_DIR CANDIDATE_DIR OUT_JSON [--cut N]   (--cut N: truncate the summary to N chars, for equal-length comparisons)"""
import argparse, json, os, re, requests
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser(); ap.add_argument("qa"); ap.add_argument("cand"); ap.add_argument("out"); ap.add_argument("--cut", type=int, default=0)
ap.add_argument("--url", default="http://127.0.0.1:8791/v1"); ap.add_argument("--model", default="q38"); ap.add_argument("--parallel", type=int, default=3)
a = ap.parse_args()
PA = """下面是一份會議摘要與一組問題。請只根據摘要回答每一題，答案要簡短。摘要沒有提供足以回答的資訊時，必須寫「無法回答」，不可用常識或猜測補上。
只輸出 JSON 陣列，每題一個字串，順序相同。

摘要：
{s}

問題：
{q}"""
PJ = """比對每一題的標準答案與作答，輸出評分：
- 1：作答與標準答案意思相同（數字、對象、結論一致，用字不同或有語音辨識錯字沒關係）。
- 0：作答是「無法回答」或沒有作答。
- -1：作答給了與標準答案不同或矛盾的內容。
只輸出 JSON 陣列（1、0 或 -1），順序相同。

{q}"""


def ask(p, mt):
    r = requests.post(a.url + "/chat/completions", json={"model": a.model, "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": mt, "chat_template_kwargs": {"enable_thinking": False}}, timeout=3600).json()
    t = r["choices"][0]["message"].get("content") or "" if "choices" in r else ""
    m = re.search(r"\[.*\]", t, re.S)
    try: return json.loads(m.group(0)) if m else []
    except Exception: return []


def one(f):
    d = json.load(open(f"{a.qa}/{f}", encoding="utf-8")); qs = d["questions"]; c = f"{a.cand}/{f}"
    if not qs: return None
    if not os.path.exists(c): return {"id": d["id"], "missing": True, "n": len(qs), "v": [0] * len(qs), "slice": [x["slice"] for x in qs], "slices": d["slices"], "chars": 0}
    s = json.load(open(c, encoding="utf-8")); s = s.get("prose") or s.get("display") or s.get("raw") or ""
    if a.cut: s = s[:a.cut]
    n = len(re.sub(r"\[\d+(?::\d+)+\]|\s|#|-", "", s)); v = None
    for _ in range(2):
        ans = ask(PA.format(s=s, q="\n".join(f"{i + 1}. {x['q']}" for i, x in enumerate(qs))), 5000)
        if len(ans) != len(qs): continue
        j = ask(PJ.format(q="\n".join(f"{i + 1}. 標準答案：{x['a']}｜作答：{y}" for i, (x, y) in enumerate(zip(qs, ans)))), 2000)
        if len(j) == len(qs): v = [1 if x == 1 else -1 if x == -1 else 0 for x in j]; v = [0 if re.search("無法回答", str(y)) else x for x, y in zip(v, ans)]; break
    if v is None: return {"id": d["id"], "unparsed": True}
    return {"id": d["id"], "n": len(qs), "v": v, "answers": ans, "slice": [x["slice"] for x in qs], "slices": d["slices"], "chars": n}


with ThreadPoolExecutor(a.parallel) as ex:
    rows = [r for r in ex.map(one, sorted(os.listdir(a.qa))) if r]
ok = [r for r in rows if "v" in r]; N = sum(r["n"] for r in ok) or 1; sh = lambda k: sum(r["v"].count(k) for r in ok) / N
th = [[], [], []]
for r in ok:
    for v, s in zip(r["v"], r["slice"]): th[min(2, s * 3 // r["slices"])].append(v == 1)
ch = sum(r["chars"] for r in ok) or 1
res = {"candidate": a.cand, "n": len(ok), "unparsed": len(rows) - len(ok), "missing": sum(1 for r in ok if r.get("missing")), "questions": N, "correct": sh(1), "wrong": sh(-1), "none": sh(0),
       "by_third": [sum(t) / max(1, len(t)) for t in th], "macro_correct": sum(r["v"].count(1) / r["n"] for r in ok) / max(1, len(ok)), "correct_per_1000": 1000 * sum(r["v"].count(1) for r in ok) / ch, "median_chars": sorted(r["chars"] for r in ok)[len(ok) // 2] if ok else 0, "rows": rows}
json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
print(f"qa {a.cand}: n={res['n']} q={N} correct {res['correct']:.2f} wrong {res['wrong']:.2f} none {res['none']:.2f} | thirds {'/'.join(f'{x:.2f}' for x in res['by_third'])} | per1000 {res['correct_per_1000']:.1f} chars {res['median_chars']} | unparsed {res['unparsed']} missing {res['missing']}")
