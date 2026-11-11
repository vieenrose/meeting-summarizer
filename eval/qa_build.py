"""2026-11-11 user idea: content quality = a reader holding only the summary can answer a well-covering questionnaire about the transcript.
Builds the questionnaire: the transcript is cut into N equal-time slices, q38 writes K questions per slice, each with a short answer, the
timestamp and a verbatim quote. Code gates: timestamp inside the slice, quote found verbatim in the transcript. A closed-book pass then drops
questions the model answers correctly without any text (guessable). Usage: qa_build.py SPLIT TRANSCRIPTS OUT_DIR [--slices 8 --per 4]"""
import argparse, json, os, re, requests
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser(); ap.add_argument("split"); ap.add_argument("tr"); ap.add_argument("out"); ap.add_argument("--slices", type=int, default=8); ap.add_argument("--per", type=int, default=4)
ap.add_argument("--url", default="http://127.0.0.1:8791/v1"); ap.add_argument("--model", default="q38"); ap.add_argument("--parallel", type=int, default=3)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
sec = lambda t: sum(int(x) * 60 ** i for i, x in enumerate(reversed(t.split(":")))); TS = re.compile(r"^\[(\d+(?::\d+)+)\]")
nz = lambda s: re.sub(r"\s", "", s)
PQ = """下面是一場會議逐字稿的一個片段（語音辨識結果，可能有錯字）。請出 {k} 個問題，用來檢驗一份會議摘要是否涵蓋這個片段的重要內容。
- 只問這場會議中重要的具體事實：誰提出或決定了什麼、數字、理由、結論、待辦事項。不要問瑣事、寒暄或程序性發言。
- 問題要能獨立讀懂（寫明主題，不用「他」「這個」），不提時間，也不能從常識猜到答案。
- 答案要簡短（最多 25 字）且必須在片段中明確說出。
- quote 必須是片段中一段逐字照抄的原文（8 至 40 字），足以證明答案；ts 是該行開頭的時間。
- 片段若沒有重要內容，可以少出，甚至輸出 []。
只輸出 JSON 陣列：[{{"q": "...", "a": "...", "ts": "分:秒", "quote": "..."}}]

片段：
{t}"""
PB = """請不查任何資料，直接回答下列關於某場會議的問題。不知道就寫「不知道」，不要猜。只輸出 JSON 陣列，每題一個字串，順序相同。

{q}"""
PJ = """比對每一題的標準答案與作答。作答與標準答案意思相同（數字、對象一致）才算 1，否則 0。只輸出 JSON 陣列（0 或 1），順序相同。

{q}"""


def ask(p, mt=3000):
    r = requests.post(a.url + "/chat/completions", json={"model": a.model, "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": mt, "chat_template_kwargs": {"enable_thinking": False}}, timeout=3600).json()
    t = r["choices"][0]["message"].get("content") or "" if "choices" in r else ""
    m = re.search(r"\[.*\]", t, re.S)
    try: return json.loads(m.group(0)) if m else []
    except Exception: return []


def one(sid):
    o = f"{a.out}/{sid}.json"
    if os.path.exists(o): return sid, -1
    lines = [l for l in open(f"{a.tr}/{sid}.txt", encoding="utf-8").read().split("\n") if TS.match(l)]; ts = [sec(TS.match(l).group(1)) for l in lines]
    t0, sp = min(ts), max(1, max(ts) - min(ts) + 1); full = nz("".join(lines)); qs = []
    for i in range(a.slices):
        sl = [l for l, t in zip(lines, ts) if (t - t0) * a.slices // sp == i]
        if not sl: continue
        have = {TS.match(l).group(1) for l in sl}
        for x in ask(PQ.format(k=a.per, t="\n".join(sl))):
            if isinstance(x, dict) and all(isinstance(x.get(k), str) and x[k] for k in ("q", "a", "ts", "quote")) and x["ts"].strip("[]") in have and len(nz(x["quote"])) >= 6 and nz(x["quote"]) in full:
                qs.append({"q": x["q"], "a": x["a"], "ts": x["ts"].strip("[]"), "quote": x["quote"], "slice": i})
    n0 = len(qs)
    if qs:
        b = ask(PB.format(q="\n".join(f"{i + 1}. {x['q']}" for i, x in enumerate(qs))), 4000)
        if len(b) == len(qs):
            j = ask(PJ.format(q="\n".join(f"{i + 1}. 標準答案：{x['a']}｜作答：{y}" for i, (x, y) in enumerate(zip(qs, b)))), 1500)
            if len(j) == len(qs): qs = [x for x, v in zip(qs, j) if v != 1]
    json.dump({"id": sid, "slices": a.slices, "written": n0, "questions": qs}, open(o, "w", encoding="utf-8"), ensure_ascii=False, indent=1); return sid, len(qs)


with ThreadPoolExecutor(a.parallel) as ex:
    r = [x for x in ex.map(one, json.load(open(a.split))["heldout"])]
print(a.out, "sessions", len(r), "questions/session", [n for _, n in r])
