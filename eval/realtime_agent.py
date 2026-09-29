"""Reading agent that keeps up with a live meeting on a phone CPU (Qwen3.5-2B Q4_0 on llama.cpp).

eval/journal_agent.py rebuilds every turn from scratch (journal view + window), so each turn pays
a full prefill. On a hybrid model (Qwen3.5: linear-attention layers carry a recurrent state)
llama-server can only reuse the KV cache when the new prompt *extends* the previous one -- a
prompt that diverges after a shared prefix is recomputed. So here the session is one growing
conversation:

  [system] [user: journal so far] ([user: window k] [assistant: actions] [user: check] [assistant: fixes])*

Each window costs only its own tokens plus the short check turn. When the conversation would pass
the context budget it restarts from [system] [user: whole journal]. The check turn verifies the
notes just written against the window, which is still in context (no re-prefill), replacing the
per-item verification that journal_agent runs after the meeting.

Arrival is simulated: window k is available when its last line has been spoken. From the server's
own counts (prompt tokens actually computed, tokens generated) and the phone's measured speeds,
the report gives, per session, the worst lag behind the live meeting and the wait after its end.
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.journal_agent import ACT, CITE, as_prose  # noqa: E402
from summarizer.ingest import parse_line, resolve_citation  # noqa: E402

WINDOW_TOKENS, CTX, OUTPUT_TOKENS, MAX_LOOKBACKS, LOOKBACK_LINES, MAX_ACTIONS = 2000, 32768, 1000, 2, 60, 8
# Reno7 CPU, 8 threads, Qwen3.5-2B Q4_0 (llama-bench pp512 / tg32); --phone-pp/--phone-tg per model.
# Each trace entry keeps its arrival time and token counts, so the timing can be recomputed.
PHONE_PP, PHONE_TG = 40.4, 8.8

SYSTEM = """你是會議閱讀助理，會議正在進行，你依序收到逐字稿片段（語音辨識結果，可能有錯字；講者標籤 S1、S2 不可靠，請從內容判斷身分，無法確定寫「發言者」）。
你有一本筆記本，記錄會議至今的重點。每收到一段，輸出動作，每行一個：
NOTE [時間] (類型) 內容 —— 新增筆記。類型為 DECISION、ACTION、NUMBER、OPEN-ISSUE 或 -。時間照抄本片段中的一行。內容寫誰主張什麼、數字、結果；建議與決議要分清，保留與通過要分清。
REVISE #編號 [時間] 內容 —— 本片段改變了先前某則筆記（例如先前保留、現在通過），改寫該則。
LOOKBACK 起始時間-結束時間 —— 需要重讀先前的逐字稿才能確認時使用。
NEXT —— 本片段處理完畢。
筆記是摘要，不是抄寫：每段最多 5 則，每則不超過 40 字，只記決議、待辦、數字、爭議與各方主張；程序性發言、寒暄、逐句內容都不要記。
只寫逐字稿中有的內容，不要推測。沒有重點就只輸出 NEXT。最後一行必須是 NEXT 或 LOOKBACK。

範例輸出：
NOTE [1:02:15] (NUMBER) 某部資訊系統預算 1200 萬元，較去年增 300 萬元，主要用於系統汰換
NOTE [1:05:40] (OPEN-ISSUE) 一位委員質疑補助遭刪減，部長稱將另案說明
NEXT"""

CHECK = """逐條核對你剛寫的筆記與本片段原文（數字、對象、誰主張、建議或決議、通過或保留）：
{notes}
有錯就輸出 FIX #編號 [時間] 改正後內容；原文找不到依據就輸出 DROP #編號；全部正確就只輸出 OK。"""

# Small models often drop the brackets around the time: accept "NOTE 24:35 (TYPE) text" too.
NOTE = re.compile(r"^\s*\[?(\d+:\d{2}(?::\d{2})?)\]?\s*(?:\((\w[\w-]*)\)\s*)?(.+)$")
OVERVIEW = """以下是一場會議的筆記：

{journal}

用三到五句話說明會議目的、主要爭點與結果。每句句尾附上它所依據的筆記時間 [時間]，時間必須照抄筆記中的時間。只輸出這幾句。"""


def similar(a, b):
    bg = lambda t: {t[i:i + 2] for i in range(len(t) - 1)}  # noqa: E731
    x, y = bg(a), bg(b)
    return len(x & y) / max(1, len(x | y))


NOTHINK = "<think>\n</think>\n"
FIX = re.compile(r"^\s*FIX\s*#(\d+)\s*(.*)$")
DROP = re.compile(r"^\s*DROP\s*#(\d+)")


def render(e):
    return f"#{e['id']} [{e['ts']}] " + (f"({e['tag']}) " if e["tag"] else "") + e["text"]


class Session:
    """One growing conversation on one server slot; restarts from the journal when too long."""

    def __init__(self, url, model, slot, count):
        self.url, self.model, self.slot, self.count = url.rstrip("/"), model, slot, count
        self.msgs, self.ctx_used = [], 0
        self.calls, self.restarts, self.prefill, self.decode, self.max_ctx = 0, 0, 0, 0, 0
        self.phone_pp, self.phone_tg = PHONE_PP, PHONE_TG
        self.nothink = False

    def chat(self, content, max_tokens=OUTPUT_TOKENS, keep=True, stop_next=False):
        msgs = self.msgs + [{"role": "user", "content": content}]
        # A model that always thinks (LFM2.5-2.6B) gets its answer started after an empty think
        # block; the block stays in the history (preserve_thinking) so the next prompt still
        # extends the cached one.
        prefix = [{"role": "assistant", "content": NOTHINK}] if self.nothink else []
        kwargs = {"enable_thinking": False, **({"preserve_thinking": True} if self.nothink else {})}
        for attempt in range(3):
            r = requests.post(self.url + "/chat/completions", timeout=1800, json={
                "model": self.model, "messages": msgs + prefix, "temperature": 0.2 + 0.3 * attempt,
                "max_tokens": max_tokens, "id_slot": self.slot, "cache_prompt": True,
                # Stop a reading turn at NEXT: small models ramble on to max_tokens otherwise.
                **({"stop": ["\nNEXT"]} if stop_next else {}),
                "chat_template_kwargs": kwargs}).json()
            if "choices" in r:
                break
        if "choices" not in r:
            raise RuntimeError(f"server error: {str(r)[:300]}")
        m = r["choices"][0]["message"]
        if not self.nothink and not (m.get("content") or "").strip() and m.get("reasoning_content"):
            self.nothink = True                     # the model thinks regardless: prefill from now on
            return self.chat(content, max_tokens, keep, stop_next)
        # llama-server reports what it actually computed (timings); vLLM only reports usage.
        u = r.get("usage", {})
        t = r.get("timings") or {"prompt_n": u.get("prompt_tokens", 0), "predicted_n": u.get("completion_tokens", 0)}
        self.calls += 1
        self.prefill += t["prompt_n"]
        self.decode += t["predicted_n"]
        self.ctx_used = t["prompt_n"] + t.get("cache_n", 0) + t["predicted_n"]
        self.max_ctx = max(self.max_ctx, self.ctx_used)
        raw = r["choices"][0]["message"]["content"] or ""
        if stop_next and r["choices"][0].get("finish_reason") == "stop":
            raw += "\nNEXT"                        # keep the history equal to what was generated
        out = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
        if keep:
            said = (NOTHINK + raw[len(NOTHINK):] if raw.startswith(NOTHINK) else NOTHINK + raw) if self.nothink else out
            self.msgs = msgs + [{"role": "assistant", "content": said}]
        return out, t["prompt_n"], t["predicted_n"]

    def restart(self, journal):
        self.restarts += bool(self.msgs)
        self.msgs = [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": "## 筆記本（至今）\n" + ("\n".join(map(render, journal)) or "（尚無筆記）")},
                     {"role": "assistant", "content": "NEXT"}]
        self.ctx_used = self.count(json.dumps(self.msgs, ensure_ascii=False))


def windows_of(lines, count):
    out, cur, tok = [], [], 0
    for l in lines:
        t = count(l.render())
        if cur and tok + t > WINDOW_TOKENS:
            out.append(cur)
            cur, tok = [], 0
        cur.append(l)
        tok += t
    return out + ([cur] if cur else [])


def run_session(s, text, check=True):
    lines = [parse_line(l) for l in text.splitlines() if l.strip()]
    journal, trace, clock, worst_lag = [], [], 0.0, 0.0
    proto = {"lines": 0, "actions": 0, "capped": 0, "duplicates": 0}
    s.restart(journal)

    def cost(pp, tg):
        return pp / s.phone_pp + tg / s.phone_tg

    wins = windows_of(lines, s.count)
    for k, win in enumerate(wins, 1):
        block = "\n".join(l.render() for l in win)
        arrive = win[-1].start_s + 5                # the window's last line has been spoken
        clock = max(clock, arrive)
        need = s.count(block) + OUTPUT_TOKENS * 2 + 600
        if s.ctx_used + need > CTX:
            s.restart(journal)
        user, n_lb, before = f"## 逐字稿片段 {k}\n{block}", 0, len(journal)
        while True:
            reply, pp, tg = s.chat(user, stop_next=True)
            clock += cost(pp, tg)
            trace.append({"window": k, "arrive": arrive, "pp": pp, "tg": tg, "reply": reply})
            proto["lines"] += sum(1 for x in reply.splitlines() if x.strip())
            proto["actions"] += sum(1 for x in reply.splitlines() if ACT.match(x))
            want_lb, n_act = None, 0
            for line in reply.splitlines():
                m = ACT.match(line)
                if m and m.group(1) in ("NOTE", "REVISE"):
                    n_act += 1
                    if n_act > MAX_ACTIONS:         # a small model can loop on one action
                        proto["capped"] += 1
                        continue
                if not m:
                    continue
                act, rest = m.group(1), m.group(2).strip()
                if act == "NOTE" and (n := NOTE.match(rest)) and resolve_citation(n.group(1), lines) is not None:
                    if any(similar(n.group(3), e["text"]) > 0.6 for e in journal[-30:]):
                        proto["duplicates"] += 1
                        continue                    # the same point restated at a later line
                    journal.append({"id": len(journal) + 1, "window": k, "ts": n.group(1), "tag": n.group(2), "text": n.group(3)})
                elif act == "REVISE" and (r := re.match(r"#(\d+)\s*(.*)$", rest)):
                    i = int(r.group(1)) - 1
                    if 0 <= i < len(journal) and (n := NOTE.match(r.group(2))) and not any(
                            similar(n.group(3), e["text"]) > 0.6 for j, e in enumerate(journal) if j != i):
                        journal[i].update(ts=n.group(1), tag=n.group(2) or journal[i]["tag"], text=n.group(3), revised=k)
                elif act == "LOOKBACK" and n_lb < MAX_LOOKBACKS:
                    want_lb = rest
            if not want_lb:
                break
            ts = re.findall(r"\d+:\d{2}(?::\d{2})?", want_lb)
            i0 = resolve_citation(ts[0], lines) if ts else None
            i1 = resolve_citation(ts[-1], lines) if ts else None
            if i0 is None:
                break
            user = "## 重讀的先前逐字稿\n" + "\n".join(l.render() for l in lines[i0:max(i0 + 1, (i1 or i0) + 1)][:LOOKBACK_LINES])
            n_lb += 1
        new = [e for e in journal[before:] if not e.get("dropped")]
        if check and new:
            reply, pp, tg = s.chat(CHECK.format(notes="\n".join(map(render, new))), max_tokens=600)
            clock += cost(pp, tg)
            trace.append({"window": k, "arrive": arrive, "pp": pp, "tg": tg, "check": reply})
            for line in reply.splitlines():
                if (m := DROP.match(line)) and before < int(m.group(1)) <= len(journal):
                    journal[int(m.group(1)) - 1]["dropped"] = True
                elif (m := FIX.match(line)) and before < int(m.group(1)) <= len(journal) and (n := NOTE.match(m.group(2))):
                    i = int(m.group(1)) - 1
                    if resolve_citation(n.group(1), lines) is not None and not any(
                            similar(n.group(3), e["text"]) > 0.6 for j, e in enumerate(journal) if j != i):
                        journal[i].update(ts=n.group(1), text=n.group(3), fixed=k)
        worst_lag = max(worst_lag, clock - arrive)
    kept = [e for e in journal if not e.get("dropped")]
    # A small model merges and invents at the reduce step, so the minutes are assembled from the
    # checked notes by type; the model only writes the overview.
    sections = {"決議事項": ["DECISION"], "待辦與負責人": ["ACTION"], "保留與未決": ["OPEN-ISSUE"]}
    out = []
    for title, tags in sections.items():
        items = [e for e in kept if (e["tag"] or "").upper() in tags]
        out += [f"【{title}】"] + ([f"- {e['text'].rstrip('。')} [{e['ts']}]" for e in items] or ["- 無"])
    s.msgs = []                                     # a fresh, short context
    digest = "\n".join(f"[{e['ts']}] " + (f"({e['tag']}) " if e["tag"] else "") + e["text"] for e in kept)
    overview, pp, tg = s.chat(OVERVIEW.format(journal=digest), max_tokens=400, keep=False)
    clock += cost(pp, tg)
    trace.append({"window": "overview", "arrive": lines[-1].start_s + 5, "pp": pp, "tg": tg, "overview": overview})
    known = {e["ts"] for e in kept}
    lines_ov = [l.strip() for l in re.split(r"(?<=。)", overview.replace("\n", "")) if CITE.search(l)
                and set(CITE.findall(l)) <= known]
    out += ["【會議概要】"] + ([f"- {l.lstrip('- ')}" for l in lines_ov] or ["- 無"])
    minutes = "\n".join(out)
    end = lines[-1].start_s + 5
    timing = {"meeting_s": end, "worst_lag_s": round(worst_lag), "after_end_s": round(clock - end),
              "windows": len(wins), "protocol": proto}
    return journal, minutes, trace, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8095/v1")
    ap.add_argument("--model", default="q2b")
    ap.add_argument("--split", default="data/split_v2.json")
    ap.add_argument("--transcripts", default="data/v2/transcripts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=4, help="must not exceed the server's -np")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--nothink-prefill", action="store_true", help="for models that always think")
    ap.add_argument("--phone-pp", type=float, default=PHONE_PP, help="phone prefill tok/s for the timing model")
    ap.add_argument("--phone-tg", type=float, default=PHONE_TG, help="phone decode tok/s for the timing model")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B-FP8")
    count = lambda t: len(tok.encode(t, add_special_tokens=False))  # noqa: E731
    os.makedirs(a.out, exist_ok=True)

    def one(i_sid):
        i, sid = i_sid
        if os.path.exists(os.path.join(a.out, sid + ".json")):
            return f"skip {sid}"
        s = Session(a.url, a.model, i % a.parallel, count)
        s.phone_pp, s.phone_tg = a.phone_pp, a.phone_tg
        s.nothink = a.nothink_prefill
        text = open(os.path.join(a.transcripts, sid + ".txt"), encoding="utf-8").read()
        journal, minutes, trace, timing = run_session(s, text, check=not a.no_check)
        notes = [{k: e[k] for k in ("id", "window", "ts", "text", "tag")} for e in journal if not e.get("dropped")]
        rec = {"notes": notes, "minutes": minutes, "prose": as_prose(minutes), "trace": trace, "timing": timing,
               "calls": s.calls, "restarts": s.restarts, "prefill_tokens": s.prefill, "decode_tokens": s.decode,
               "max_ctx": s.max_ctx, "dropped": sum(1 for e in journal if e.get("dropped")),
               "fixed": sum(1 for e in journal if e.get("fixed")), "revisions": sum(1 for e in journal if e.get("revised"))}
        json.dump(rec, open(os.path.join(a.out, sid + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
        return (f"{sid}: {len(notes)} notes ({rec['dropped']} dropped, {rec['fixed']} fixed), {s.calls} calls, "
                f"{s.restarts} restarts, prefill {s.prefill} decode {s.decode} max ctx {s.max_ctx} | "
                f"meeting {timing['meeting_s'] / 60:.0f} min, worst lag {timing['worst_lag_s'] / 60:.1f} min, "
                f"minutes {timing['after_end_s'] / 60:.1f} min after end")

    sids = a.only or json.load(open(a.split))["heldout"]
    with ThreadPoolExecutor(a.parallel) as ex:
        for line in ex.map(one, enumerate(sids)):
            print(line, flush=True)


if __name__ == "__main__":
    main()
