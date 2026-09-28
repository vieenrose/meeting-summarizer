"""Reading agent with an external journal, every turn within a short context (<= 32k tokens).

Built for a model that fits a phone but not a whole meeting in its context budget (Bonsai 2 27B
ternary on the PrismML llama-server). The transcript is read once, in order, window by window.
The journal -- the agent's notes -- lives outside the context; each turn sees a bounded view of
it: the most recent entries, the entries that share keywords with the current window, and a
count of the rest. Per turn the agent emits actions, one per line:

  NOTE [ts] (TYPE) text        add an entry (TYPE: DECISION / ACTION / NUMBER / OPEN-ISSUE / -)
  REVISE #id [ts] text         rewrite entry id when this window changes it (held -> passed ...)
  LOOKBACK t1-t2               read earlier transcript lines between two timestamps, then decide
  NEXT                         go on

After the last window the journal becomes a structured minutes (decisions, actions, open items,
overview), each item citing a transcript line, and every item is checked against the transcript
lines around its citation (Chain-of-Verification) -- again in a short context.
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.ingest import format_ts, parse_line, resolve_citation  # noqa: E402

WINDOW_TOKENS, JOURNAL_TOKENS, LOOKBACK_LINES, MAX_LOOKBACKS = 6000, 12000, 80, 2
# Every turn must fit the phone's context: prompt + thinking budget + action output <= CTX.
CTX, OUTPUT_TOKENS = 32768, 1500

SYSTEM = """你是會議閱讀助理，依序閱讀一場會議的逐字稿片段（語音辨識結果，可能有錯字；講者標籤 S1、S2 不可靠，請從內容判斷身分，無法確定寫「發言者」）。
你有一本筆記本（日誌），記錄會議至今的重點。每讀一段，輸出動作，每行一個：
NOTE [時間] (類型) 內容 —— 新增筆記。類型為 DECISION、ACTION、NUMBER、OPEN-ISSUE 或 -。時間照抄本片段中的一行。內容寫誰主張什麼、數字、結果；建議與決議要分清，保留與通過要分清。
REVISE #編號 [時間] 內容 —— 本片段改變了先前某則筆記（例如先前保留、現在通過），改寫該則。
LOOKBACK 起始時間-結束時間 —— 需要重讀先前的逐字稿才能確認時使用（最多兩次）。
NEXT —— 本片段處理完畢。
只寫逐字稿中有的內容，不要推測。最後一行必須是 NEXT 或 LOOKBACK。"""

FINAL = """以下是你閱讀整場會議時寫下的筆記本：

{journal}

請據此寫出會議紀錄，分四節，每節標題獨立一行：
【決議事項】已作成的決議（不要列建議或未決事項）
【待辦與負責人】待辦事項、負責機關或人員、期限
【保留與未決】被保留、另行協商或仍有爭議的事項
【會議概要】三到五句說明會議目的、主要爭點與結果
每項一行，以「- 」開頭，句尾附上出處時間 [時間]，時間必須來自筆記；沒有就寫「- 無」。"""

VERIFY = """以下是會議紀錄中的一項，以及逐字稿中該項引用時間附近的原文。

## 紀錄項目
{item}

## 逐字稿
{excerpt}

這一項是否與逐字稿相符？相符就原樣輸出該項；有錯（數字、對象、誰主張、建議或決議、通過或保留）就輸出改正後的一行；找不到依據就只輸出「刪除」。"""

ACT = re.compile(r"^\s*(NOTE|REVISE|LOOKBACK|NEXT)\b(.*)$")
NOTE = re.compile(r"^\s*\[(\d+:\d{2}(?::\d{2})?)\]\s*(?:\((\w[\w-]*)\)\s*)?(.+)$")
CITE = re.compile(r"[\[［](\d+:\d{2}(?::\d{2})?)")


class Agent:
    def __init__(self, url, model, count, think_budget=0):
        self.url, self.model, self.count = url.rstrip("/"), model, count
        self.think = think_budget           # tokens of thinking per turn; 0 = thinking off
        self.calls = self.think_tokens = 0
        self.max_prompt = 0

    def prompt_limit(self):
        return CTX - OUTPUT_TOKENS - self.think - 512

    def chat(self, messages, max_tokens=OUTPUT_TOKENS):
        """One stateless turn. Thinking, when on, is bounded by the server's --reasoning-budget and
        returned apart (reasoning_content); it is counted, never fed back into any later turn."""
        self.calls += 1
        self.max_prompt = max(self.max_prompt, sum(self.count(m["content"]) for m in messages))
        r = requests.post(self.url + "/chat/completions", timeout=1800, json={
            "model": self.model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens + self.think,
            "chat_template_kwargs": {"enable_thinking": self.think > 0}}).json()
        msg = r["choices"][0]["message"]
        if msg.get("reasoning_content"):
            self.think_tokens += self.count(msg["reasoning_content"])
        return re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.S).strip()

    def journal_view(self, journal, window_text, budget=JOURNAL_TOKENS):
        """Recent entries plus entries sharing words with the window, within budget tokens."""
        render = lambda e: f"#{e['id']} [{e['ts']}] " + (f"({e['tag']}) " if e["tag"] else "") + e["text"]  # noqa: E731
        words = set(re.findall(r"[一-鿿]{2,4}|\d+", window_text))
        scored = sorted(journal, key=lambda e: -len(words & set(re.findall(r"[一-鿿]{2,4}|\d+", e["text"]))))
        chosen, used = {}, 0
        for e in list(reversed(journal[-25:])) + scored[:40]:
            if e["id"] in chosen:
                continue
            t = self.count(render(e))
            if used + t > budget:
                break
            chosen[e["id"]], used = e, used + t
        shown = [render(e) for e in sorted(chosen.values(), key=lambda e: e["id"])]
        rest = len(journal) - len(shown)
        return "\n".join(shown) + (f"\n（另有 {rest} 則較早的筆記未顯示）" if rest > 0 else "") or "（尚無筆記）"


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


def run_session(agent, text):
    lines = [parse_line(l) for l in text.splitlines() if l.strip()]
    journal, trace = [], []
    for k, win in enumerate(windows_of(lines, agent.count), 1):
        block = "\n".join(l.render() for l in win)
        lookback, n_lb = "", 0
        while True:
            budget = JOURNAL_TOKENS
            while True:                     # shrink the journal view until the turn fits CTX
                user = (f"## 筆記本\n{agent.journal_view(journal, block, budget)}\n\n"
                        + (f"## 重讀的先前逐字稿\n{lookback}\n\n" if lookback else "")
                        + f"## 逐字稿片段 {k}\n{block}")
                if agent.count(SYSTEM + user) <= agent.prompt_limit() or budget < 1000:
                    break
                budget //= 2
            reply = agent.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
            trace.append({"window": k, "reply": reply})
            want_lb = None
            for line in reply.splitlines():
                m = ACT.match(line)
                if not m:
                    continue
                act, rest = m.group(1), m.group(2).strip()
                if act == "NOTE" and (n := NOTE.match(rest)) and resolve_citation(n.group(1), lines) is not None:
                    journal.append({"id": len(journal) + 1, "window": k, "ts": n.group(1), "tag": n.group(2), "text": n.group(3)})
                elif act == "REVISE" and (r := re.match(r"#(\d+)\s*(.*)$", rest)):
                    i = int(r.group(1)) - 1
                    if 0 <= i < len(journal) and (n := NOTE.match(r.group(2))):
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
            lookback = "\n".join(l.render() for l in lines[i0:max(i0 + 1, (i1 or i0) + 1)][:LOOKBACK_LINES])
            n_lb += 1
            # Entries this turn already added stay; the model sees them in the journal next turn.
    journal_text = "\n".join(f"[{e['ts']}] " + (f"({e['tag']}) " if e["tag"] else "") + e["text"] for e in journal)
    cut = 400
    while agent.count(FINAL.format(journal=journal_text)) > agent.prompt_limit() - 1000 and cut > 40:
        cut //= 2                           # long meeting: shorten entries, keep them all
        journal_text = "\n".join(f"[{e['ts']}] " + (f"({e['tag']}) " if e["tag"] else "") + e["text"][:cut] for e in journal)
    minutes = agent.chat([{"role": "user", "content": FINAL.format(journal=journal_text)}], max_tokens=2500)
    verified = []
    for line in minutes.splitlines():
        if not line.startswith("- ") or line[2:].strip() == "無":
            verified.append(line)
            continue
        ts = CITE.findall(line)
        idx = [i for i in (resolve_citation(t, lines) for t in ts) if i is not None]
        if not idx:
            verified.append(line)
            continue
        t0 = lines[min(idx)].start_s
        ex = "\n".join(l.render() for l in lines if -30 <= l.start_s - t0 <= 150)[:6000]
        v = agent.chat([{"role": "user", "content": VERIFY.format(item=line, excerpt=ex)}], max_tokens=400)
        if v.strip().startswith("刪除"):
            continue
        verified.append(v if v.startswith("- ") else "- " + v)
    return journal, minutes, "\n".join(verified), trace


def as_prose(minutes):
    items = [l[2:].strip() for l in minutes.splitlines() if l.startswith("- ") and l[2:].strip() != "無"]
    return "".join(x if x.endswith("。") else x + "。" for x in items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default="http://127.0.0.1:8090/v1,http://127.0.0.1:8091/v1")
    ap.add_argument("--model", default="bonsai")
    ap.add_argument("--split", default="data/split_v2.json")
    ap.add_argument("--transcripts", default="data/v2/transcripts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--think", type=int, default=0, help="thinking budget per turn (must match the server's)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B-FP8")
    count = lambda t: len(tok.encode(t, add_special_tokens=False))  # noqa: E731
    urls = a.urls.split(",")
    for d in (a.out, a.out + "-draft"):
        os.makedirs(d, exist_ok=True)

    def one(i_sid):
        i, sid = i_sid
        if os.path.exists(os.path.join(a.out, sid + ".json")):
            return f"skip {sid}"
        agent = Agent(urls[i % len(urls)], a.model, count, a.think)
        text = open(os.path.join(a.transcripts, sid + ".txt"), encoding="utf-8").read()
        journal, minutes, verified, trace = run_session(agent, text)
        notes = [{"id": e["id"], "window": e["window"], "ts": e["ts"], "text": e["text"], "tag": e["tag"]} for e in journal]
        base = {"notes": notes, "calls": agent.calls, "max_prompt_tokens": agent.max_prompt, "think_tokens": agent.think_tokens,
                "revisions": sum(1 for e in journal if e.get("revised")), "trace": trace}
        json.dump({**base, "minutes": verified, "prose": as_prose(verified)},
                  open(os.path.join(a.out, sid + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
        json.dump({**base, "minutes": minutes, "prose": as_prose(minutes)},
                  open(os.path.join(a.out + "-draft", sid + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
        return (f"{sid}: {len(journal)} notes, {base['revisions']} revised, {agent.calls} calls, "
                f"max prompt {agent.max_prompt} tok, thinking {agent.think_tokens} tok")

    sids = a.only or json.load(open(a.split))["heldout"]
    with ThreadPoolExecutor(a.parallel) as ex:
        for line in ex.map(one, enumerate(sids)):
            print(line, flush=True)


if __name__ == "__main__":
    main()
