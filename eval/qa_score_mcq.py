"""2026-11-11 multiple-choice questionnaire score (qa_mcq.py): the answerer sees only the summary and picks A-D or E (not in the summary); grading is by
code. Reports correct / wrong / abstain, net = correct - wrong/3 (chance-corrected), by third, per 1000 chars.
Usage: qa_score_mcq.py QA_MCQ_DIR CANDIDATE_DIR OUT_JSON [--cut N]"""
import argparse, json, os, re, requests
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser(); ap.add_argument("qa"); ap.add_argument("cand"); ap.add_argument("out"); ap.add_argument("--cut", type=int, default=0)
ap.add_argument("--url", default="http://127.0.0.1:8791/v1"); ap.add_argument("--model", default="q38"); ap.add_argument("--parallel", type=int, default=3)
a = ap.parse_args()
PA = """下面是一份會議摘要與一組選擇題。請只根據摘要作答。摘要沒有明確提供答案時必須選 E，不可用常識、猜測或刪去法。
只輸出 JSON 陣列，每題一個字母（A 至 E），順序相同。

摘要：
{s}

題目：
{q}"""


def one(f):
    d = json.load(open(f"{a.qa}/{f}", encoding="utf-8")); qs = d["questions"]; c = f"{a.cand}/{f}"
    if not qs: return None
    base = {"id": d["id"], "n": len(qs), "slice": [x["slice"] for x in qs], "slices": d["slices"]}
    if not os.path.exists(c): return {**base, "missing": True, "v": [0] * len(qs), "chars": 0}
    s = json.load(open(c, encoding="utf-8")); s = s.get("prose") or s.get("display") or s.get("raw") or ""
    if a.cut: s = s[:a.cut]
    q = "\n".join(f"{i + 1}. {x['q']}\n" + "　".join(f"{l}. {o}" for l, o in zip("ABCDE", x["options"])) for i, x in enumerate(qs))
    for t in (0, 0.3):
        r = requests.post(a.url + "/chat/completions", json={"model": a.model, "messages": [{"role": "user", "content": PA.format(s=s, q=q)}], "temperature": t, "max_tokens": 1500, "chat_template_kwargs": {"enable_thinking": False}}, timeout=3600).json()
        m = re.search(r"\[.*\]", r["choices"][0]["message"].get("content") or "" if "choices" in r else "", re.S)
        try: ans = [str(y).strip()[:1].upper() for y in json.loads(m.group(0))]
        except Exception: ans = []
        if len(ans) == len(qs):
            return {**base, "v": [1 if y == x["key"] else 0 if y not in "ABCD" or not y else -1 for x, y in zip(qs, ans)], "answers": ans, "chars": len(re.sub(r"\[\d+(?::\d+)+\]|\s|#|-", "", s))}
    return {"id": d["id"], "unparsed": True}


with ThreadPoolExecutor(a.parallel) as ex: rows = [r for r in ex.map(one, sorted(os.listdir(a.qa))) if r]
ok = [r for r in rows if "v" in r]; N = sum(r["n"] for r in ok) or 1; sh = lambda k: sum(r["v"].count(k) for r in ok) / N; th = [[], [], []]
for r in ok:
    for v, s in zip(r["v"], r["slice"]): th[min(2, s * 3 // r["slices"])].append(v == 1)
res = {"candidate": a.cand, "n": len(ok), "unparsed": len(rows) - len(ok), "questions": N, "correct": sh(1), "wrong": sh(-1), "abstain": sh(0), "net": sh(1) - sh(-1) / 3, "by_third": [sum(t) / max(1, len(t)) for t in th],
       "correct_per_1000": 1000 * sum(r["v"].count(1) for r in ok) / (sum(r["chars"] for r in ok) or 1), "median_chars": sorted(r["chars"] for r in ok)[len(ok) // 2] if ok else 0, "rows": rows}
json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
print(f"mcq {a.cand}: n={res['n']} q={N} correct {res['correct']:.2f} wrong {res['wrong']:.2f} abstain {res['abstain']:.2f} net {res['net']:.2f} | thirds {'/'.join(f'{x:.2f}' for x in res['by_third'])} | per1000 {res['correct_per_1000']:.1f} chars {res['median_chars']} | unparsed {res['unparsed']}")
