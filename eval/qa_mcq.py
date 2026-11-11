"""2026-11-11 user: make the questionnaire more operational (multiple choice). Converts data/qa/SET (open questions, qa_build.py) to multiple choice:
q38 writes 3 wrong options per question, preferably other values/people/outcomes said in the SAME meeting (our dominant error is wrong binding).
Code shuffles A-D (seeded) and adds E = the summary does not say. A blind pass (no text) drops questions answered right from the options alone.
Usage: qa_mcq.py QA_DIR OUT_DIR"""
import json, os, random, re, requests, sys, zlib
from concurrent.futures import ThreadPoolExecutor
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); URL = "http://127.0.0.1:8791/v1"; E = "摘要未提及"
PD = """下面是關於同一場會議的問答題，附標準答案與逐字稿原文。請為每一題寫 3 個錯誤選項：
- 與標準答案同一類型、長度相近、看起來同樣合理；優先使用同一場會議其他題目或原文中出現的其他數字、人物、對象或相反的結論。
- 不可與標準答案同義，也不可明顯荒謬。
只輸出 JSON 陣列，每題一個含 3 個字串的陣列，順序相同。

{q}"""
PB = """請不查任何資料回答下列選擇題。真的無法判斷就選 E。只輸出 JSON 陣列，每題一個字母，順序相同。

{q}"""


def ask(p, mt=4000):
    r = requests.post(URL + "/chat/completions", json={"model": "q38", "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": mt, "chat_template_kwargs": {"enable_thinking": False}}, timeout=3600).json()
    t = r["choices"][0]["message"].get("content") or "" if "choices" in r else ""; m = re.search(r"\[.*\]", t, re.S)
    try: return json.loads(m.group(0)) if m else []
    except Exception: return []


def fmt(qs): return "\n".join(f"{i + 1}. {x['q']}\n" + "　".join(f"{l}. {o}" for l, o in zip("ABCDE", x["options"])) for i, x in enumerate(qs))


def one(f):
    if os.path.exists(f"{out}/{f}"): return -1
    d = json.load(open(f"{src}/{f}", encoding="utf-8")); qs = d["questions"]; keep = []
    ds = ask(PD.format(q="\n".join(f"{i + 1}. {x['q']}｜標準答案：{x['a']}｜原文：{x['quote']}" for i, x in enumerate(qs)))) if qs else []
    if len(ds) == len(qs):
        for x, w in zip(qs, ds):
            w = [str(y).strip() for y in w if str(y).strip() and str(y).strip() != x["a"]] if isinstance(w, list) else []
            if len(set(w)) < 3: continue
            o = [x["a"]] + list(dict.fromkeys(w))[:3]; random.Random(zlib.crc32((d["id"] + x["q"]).encode())).shuffle(o)
            keep.append({**x, "options": o + [E], "key": "ABCD"[o.index(x["a"])]})
    n0 = len(keep)
    if keep:
        b = ask(PB.format(q=fmt(keep)), 1500)
        if len(b) == len(keep): keep = [x for x, y in zip(keep, b) if str(y).strip()[:1].upper() != x["key"]]
    json.dump({"id": d["id"], "slices": d["slices"], "open": len(qs), "mcq": n0, "questions": keep}, open(f"{out}/{f}", "w", encoding="utf-8"), ensure_ascii=False, indent=1); return len(keep)


with ThreadPoolExecutor(3) as ex: print(out, "questions/session", list(ex.map(one, sorted(os.listdir(src)))))
