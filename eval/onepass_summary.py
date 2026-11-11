"""One-pass whole-transcript summary (no reading agent): the full transcript in one prompt,
output in the S5 shape (重點清單 + [m:ss] lines) so judge_prose scores it like the agent arms."""
import argparse, json, os, re, sys, time
import requests

PROMPT = """以下是一場會議或節目的完整逐字稿（含時間戳與講者）。請用繁體中文寫出重點清單：
- 第一行寫「重點清單」，之後每行一個重點，格式為「[分:秒] 重點內容」，例如「[12:40] 某某主張……」，時間戳取自逐字稿中該重點出現處。
- 依時間順序，涵蓋全場所有重要主題、數字、決定、待辦事項與講者立場，約 30 至 60 條。
- 只寫逐字稿中明確出現的內容，數字與人名必須與原文一致，不得推測。

逐字稿：
{tx}"""

# 2026-11-02 user target: final summary about 500 tokens
PROMPT_500 = """以下是一場會議或節目的完整逐字稿（含時間戳與講者）。請用繁體中文寫出精簡的重點清單：
- 第一行寫「重點清單」，之後每行一個重點，格式為「[分:秒] 重點內容」，時間戳取自逐字稿中該重點出現處。
- 依時間順序，只保留全場最重要的主題、關鍵數字、決定、待辦事項與講者立場，共 10 至 15 條，每條一句、不超過 40 字，全文不超過 600 字。
- 只寫逐字稿中明確出現的內容，數字與人名必須與原文一致，不得推測。

逐字稿：
{tx}"""

PROMPT_1K = """以下是一場會議或節目的完整逐字稿（含時間戳與講者）。請用繁體中文寫出精簡的重點清單：
- 第一行寫「重點清單」，之後每行一個重點，格式為「[分:秒] 重點內容」，時間戳取自逐字稿中該重點出現處。
- 依時間順序，只保留全場最重要的主題、關鍵數字、決定、待辦事項與講者立場，共 20 至 30 條，每條一句、不超過 40 字，全文不超過 1200 字（不論逐字稿長短，篇幅都維持在這個範圍）。
- 只寫逐字稿中明確出現的內容，數字與人名必須與原文一致，不得推測。

逐字稿：
{tx}"""

# 2026-11-03 user: the final summary is synthesized PROSE with [m:ss] anchors, not a list of key points
PROMPT_PROSE_1K = """以下是一場會議或節目的完整逐字稿（含時間戳與講者）。請用繁體中文寫出一篇綜合性的會議摘要：
- 用連貫的段落行文，不要條列。每段圍繞一個主題，說明來龍去脈、主要論點、關鍵數字、決定與待辦事項，並整合跨時段的相關內容，不要依時間逐條流水帳。
- 在對應的句子後面附上時間錨點，格式為「[分:秒]」，時間取自逐字稿中該內容出現處，例如「外資調升目標價 [0:20]」。
- 全文約 600 至 900 字，3 至 6 段（不論逐字稿長短，篇幅都維持在這個範圍）。
- 只寫逐字稿中明確出現的內容，數字與人名必須與原文一致，不得推測。

逐字稿：
{tx}"""

PROMPT_PROSE_500 = """以下是一場會議或節目的完整逐字稿（含時間戳與講者）。請用繁體中文寫出一篇綜合性的會議摘要：
- 用連貫的段落行文，不要條列。每段圍繞一個主題，說明來龍去脈、主要論點、關鍵數字、決定與待辦事項，並整合跨時段的相關內容，不要依時間逐條流水帳。
- 在對應的句子後面附上時間錨點，格式為「[分:秒]」，時間取自逐字稿中該內容出現處，例如「外資調升目標價 [0:20]」。
- 全文約 350 至 450 字，2 至 3 段（不論逐字稿長短，篇幅都維持在這個範圍）。
- 只寫逐字稿中明確出現的內容，數字與人名必須與原文一致，不得推測。

逐字稿：
{tx}"""

PROMPT_PROSE_500F = """以下是一場會議或節目的完整逐字稿（含時間戳與講者）。請用繁體中文寫出一篇綜合性的會議摘要：
- 用連貫的段落行文，不要條列。每段圍繞一個主題，2 至 3 段，依主題而非時間組織。
- 每一句只陳述一個事實（一個數字、一個主張或一個決定），句子簡短（約 15 至 40 字）。
- 每一句結尾都必須附上一個時間錨點「[分:秒]」，錨點是逐字稿中直接說出該事實的那一行的時間，例如「外資調升台積電目標價至 3500 元 [0:20]。」沒有錨點的句子不要寫。
- 全文約 350 至 450 字（不論逐字稿長短，篇幅都維持在這個範圍）。
- 只寫逐字稿中明確出現的內容，不得推測，並遵守：
  1. 只有逐字稿中明確表示同意、通過或定案時，才寫「決定」「達成共識」；僅是提議、假設或「可能要」的，寫成「提出」「建議」「討論」。
  2. 只有逐字稿中明確說出的講者、部門或職稱才可作為主詞；不確定是誰說的，就不寫主詞（用「會中提到」）。
  3. 數字、金額、日期、百分比必須逐字照抄原文，不要換算、不要湊整。
  4. 不要把不同時間點的內容合併成同一句，寧可少寫，不要補充背景。

逐字稿：
{tx}"""

# 2026-11-07 500F without the filler subject (36 % of sentences started with it)
PROMPT_PROSE_BAL = """以下是一場會議或節目的完整逐字稿（含時間戳與講者）。請用繁體中文，依時間順序從頭到尾寫出這場會議的重點，每句一行：
- 必須涵蓋整場會議直到結束，前段、中段、後段都要有；會議越長，句子越多。
- 每一句只陳述一個事實（一個數字、一個主張或一個決定），句子簡短（約 15 至 40 字），結尾附上時間錨點「[分:秒]」，錨點是逐字稿中直接說出該事實的那一行的時間。
- 只寫逐字稿中明確出現的內容，不得推測；不寫客套、程序性的話，也不要重複。
- 只有明確表示同意、通過或定案時，才寫「決定」「達成共識」；否則寫「提出」「建議」「討論」。
- 不要寫 S1、S2 這類講者代號，直接陳述內容；只有逐字稿中明確說出姓名、部門或職稱時才可作為主詞。
- 數字、金額、日期、百分比必須逐字照抄原文。

逐字稿：
{tx}"""

PROMPT_PROSE_500G = """以下是一場會議或節目的完整逐字稿（含時間戳與講者）。請用繁體中文寫出一篇綜合性的會議摘要：
- 用連貫的段落行文，不要條列。每段圍繞一個主題，2 至 3 段，依主題而非時間組織。
- 每一句只陳述一個事實（一個數字、一個主張或一個決定），句子簡短（約 15 至 40 字）。
- 每一句結尾都必須附上一個時間錨點「[分:秒]」，錨點是逐字稿中直接說出該事實的那一行的時間，例如「外資調升台積電目標價至 3500 元 [0:20]。」沒有錨點的句子不要寫。
- 全文約 350 至 450 字（不論逐字稿長短，篇幅都維持在這個範圍）。
- 只寫逐字稿中明確出現的內容，不得推測，並遵守：
  1. 只有逐字稿中明確表示同意、通過或定案時，才寫「決定」「達成共識」；僅是提議、假設或「可能要」的，寫成「提出」「建議」「討論」。
  2. 只有逐字稿中明確說出的講者、部門或職稱才可作為主詞；不確定是誰說的，就直接陳述內容，不寫主詞，也不要寫「會中提到」。
  3. 數字、金額、日期、百分比必須逐字照抄原文，不要換算、不要湊整。
  4. 不要把不同時間點的內容合併成同一句，寧可少寫，不要補充背景。

逐字稿：
{tx}"""
PROMPT_PROSE_500B = """以下是一場會議或節目的完整逐字稿（含時間戳與講者）。請用繁體中文寫出一篇綜合性的會議摘要：
- 用連貫的段落行文，不要條列。每段圍繞一個主題，說明來龍去脈、主要論點、關鍵數字、決定與待辦事項，並整合跨時段的相關內容，不要依時間逐條流水帳。
- 在對應的句子後面附上時間錨點，格式為「[分:秒]」，錨點必須是逐字稿中直接說出該句內容的那一行的時間，例如「外資調升目標價 [0:20]」。一個錨點只對應一個事實。
- 全文約 350 至 450 字，2 至 3 段（不論逐字稿長短，篇幅都維持在這個範圍）。
- 只寫逐字稿中明確出現的內容，不得推測，並遵守：
  1. 只有逐字稿中明確表示同意、通過或定案時，才寫「決定」「達成共識」；僅是提議、假設或「可能要」的，寫成「提出」「建議」「討論」。
  2. 只有逐字稿中明確說出的講者、部門或職稱才可作為主詞；不確定是誰說的，就不寫主詞（用「會中提到」）。
  3. 數字、金額、日期、百分比必須逐字照抄原文，不要換算、不要湊整。
  4. 每句只寫該時間點附近確實出現的內容，寧可少寫，不要補充背景。

逐字稿：
{tx}"""

# 2026-11-09 issue #1 (readability): topic sections + decisions/actions, length proportional to the meeting (400 + 600 chars/h, cap 2200)
PROMPT_PROSE_RD = """以下是一場會議或節目的完整逐字稿（含時間戳與講者）。請用繁體中文寫出一份易讀的會議摘要，讓沒參加會議的人能快速掌握重點：
- 依主題組織，不要依時間順序逐段記錄。分成 3 至 6 個主題，每個主題以獨立一行「## 主題名稱」開頭，下面用 2 至 5 句連貫的段落說明該主題的重點與結論。
- 最後加上「## 決定」與「## 待辦」兩節，各用「- 」條列；逐字稿中沒有明確的決定或待辦，就只寫「- 無」。
- 句子要通順完整，一句可以包含同一段討論中相關的兩個事實，但不要堆砌細節；不寫客套、程序性的話，也不要重複。
- 每一句（含條列）結尾都附上一個時間錨點「[分:秒]」，錨點是逐字稿中直接說出該事實的那一行的時間，例如「外資調升台積電目標價至 3500 元 [0:20]。」
- 全文約 {len} 字（依會議長度而定），內容要涵蓋會議的前段、中段與後段。
- 只寫逐字稿中明確出現的內容，不得推測，並遵守：
  1. 只有逐字稿中明確表示同意、通過或定案時，才寫「決定」「達成共識」；僅是提議、假設或「可能要」的，寫成「提出」「建議」「討論」。
  2. 只有逐字稿中明確說出的講者、部門或職稱才可作為主詞；不確定是誰說的，就不寫主詞。
  3. 數字、金額、日期、百分比必須逐字照抄原文，不要換算、不要湊整。
  4. 不要把不同時間點的內容合併成同一句，寧可少寫，不要補充背景。

逐字稿：
{tx}"""


RD_PLAN_LINE = "- 動筆前先寫一行計畫，列出各主題及其在逐字稿中的時間範圍，格式為「# 計畫：主題一（起–迄）；主題二（起–迄）」，空一行後再寫摘要。"
PROMPT_PROSE_RD_PLAN = PROMPT_PROSE_RD.replace("\n\n逐字稿：", "\n" + RD_PLAN_LINE + "\n\n逐字稿：")   # 2026-11-10 plan-before-summary SFT arm (user request)


def rd_plan(y):
    """Plan line built by code from a sectioned summary: each topic title with the first–last time it cites."""
    sec = lambda t: sum(int(x) * 60 ** i for i, x in enumerate(reversed(t.split(":"))))
    items = []
    for blk in re.split(r"^## ", y, flags=re.M)[1:]:
        title = blk.splitlines()[0].strip()
        if title in ("決定", "待辦"):
            continue
        an = sorted(re.findall(r"\[(\d+(?::\d+)+)\]", blk), key=sec)
        items.append(f"{title}（{an[0]}–{an[-1]}）" if an else title)
    return "# 計畫：" + "；".join(items)


def split_plan(prose):
    """(plan line or None, summary without it)."""
    m = re.match(r"\s*(# ?計畫[：:][^\n]*)\n+", prose)
    return (m.group(1), prose[m.end():].strip()) if m else (None, prose)


def rd_len(tx):
    """Target length range from the transcript duration: 800 + 500 chars per hour, cap 2200 (old rule: RD_LEN_A=400 RD_LEN_B=600)."""
    sec = lambda t: sum(int(x) * 60 ** i for i, x in enumerate(reversed(t.split(":"))))
    ts = [sec(m) for m in re.findall(r"^\[(\d+(?::\d+)+)\]", tx, re.M)] or [0]
    A, B = int(os.environ.get("RD_LEN_A", 800)), float(os.environ.get("RD_LEN_B", 500))   # 2026-11-10 unified rule (was 400 + 600/h): 3-seed cov msdev .46 -> .57, ali20 .74 -> .78, q12 equal; contradiction equal
    L = min(2200, round((A + B * (max(ts) - min(ts)) / 3600) / 100) * 100)
    L = round(L * float(os.environ.get("RD_LEN_SCALE", 1)) / 100) * 100   # 2026-11-10 user: longer allowed for dense meetings if readable
    return f"{L - 100} 至 {L + 100}"


def rd_grammar(tx, spread=False):
    """Sentences per topic from the duration target: ~55 chars per sentence over ~5 topics, 2..6 minimum, +2 maximum.
    spread (2026-11-10, coverage lever): K topics fixed by duration (3..6, one per ~20 min); the anchors of topic i are
    literal transcript timestamps of the i-th time slice, so every part of the meeting is cited. RD_SENT_MINLEN: min chars/sentence."""
    L = int(rd_len(tx).split()[0]) + 100; ml = int(os.environ.get("RD_SENT_MINLEN", 8))
    sc = float(os.environ.get("RD_SENT_CHARS", 36))
    tail = ('items ::= "- 無" | item ("\\n" item){0,7}\n'
            'item ::= "- " [^\\n\\[\\]。#]{4,120} anchor "。"?\n'
            'anchor ::= "[" [0-9]{1,3} ":" [0-9]{2} (":" [0-9]{2})? "]"\n')
    if not spread:
        lo = max(2, min(int(os.environ.get("RD_LO_MAX", 8)), round(L / sc / float(os.environ.get("RD_TOPIC_DIV", 5))))); hi = lo + int(os.environ.get("RD_HI_EXTRA", 2))
        tr = os.environ.get("RD_TOPICS", "3,6")   # 2026-11-10 dense meetings: more topics rather than longer topics
        return (f'root ::= topic{{{tr}}} "## 決定\\n\\n" items "\\n\\n## 待辦\\n\\n" items\n'
                f'topic ::= "## " [^\\n#\\[ 決待] [^\\n#\\[]{{1,29}} "\\n\\n" sent{{{lo},{hi}}} "\\n\\n"\n'
                f'sent ::= [^\\n\\[\\]。#]{{{ml},120}} anchor "。"\n' + tail)
    sec = lambda t: sum(int(x) * 60 ** i for i, x in enumerate(reversed(t.split(":"))))
    ts = list(dict.fromkeys(re.findall(r"^\[(\d+(?::\d+)+)\]", tx, re.M))); t0, t1 = sec(ts[0]), max(map(sec, ts))
    K = max(3, min(6, round((t1 - t0) / 1200))); lo = max(2, min(10, round(L / sc / K))); hi = lo + 2
    g = 'root ::= ' + " ".join(f"topic{i}" for i in range(K)) + ' "## 決定\\n\\n" items "\\n\\n## 待辦\\n\\n" items\n'
    for i in range(K):
        sl = [t for t in ts if min(K - 1, (sec(t) - t0) * K // max(1, t1 - t0)) == i] or ts
        g += (f'topic{i} ::= "## " [^\\n#\\[ 決待] [^\\n#\\[]{{1,29}} "\\n\\n" sent{i}{{{lo},{hi}}} "\\n\\n"\n'
              f'sent{i} ::= [^\\n\\[\\]。#]{{{ml},120}} "[" a{i} "]。"\n'
              f'a{i} ::= ' + " | ".join(f'"{t}"' for t in sl) + "\n")
    return g + tail


def quarter_hint(tx, mode):
    """2026-11-07 Spark puts 79-94 % of its anchors in the first quarter of long meetings: give it the meeting's time plan, computed by code."""
    sec = lambda t: sum(int(x) * 60 ** i for i, x in enumerate(reversed(t.split(":"))))
    fmt = lambda v: f"{v // 60}:{v % 60:02d}"
    lines = tx.splitlines()
    ts = [(i, sec(m.group(1))) for i, l in enumerate(lines) if (m := re.match(r"\[(\d+(?::\d+)+)\]", l))]
    t0, t1 = ts[0][1], max(t for _, t in ts)
    cut = [t0 + (t1 - t0) * k // 4 for k in range(5)]
    if mode == "markers":
        k = 1
        for i, t in ts:
            while k < 4 and t >= cut[k]:
                lines[i] = f"=== 第 {k + 1} 段開始 ===\n" + lines[i]; k += 1
        lines[ts[0][0]] = "=== 第 1 段開始 ===\n" + lines[ts[0][0]]
    plan = "；".join(f"第 {k + 1} 段 {fmt(cut[k])} 至 {fmt(cut[k + 1])}" for k in range(4))
    return f"（會議依時間分為四段：{plan}。摘要必須平均涵蓋這四段，每段寫的句子數量大致相同，寫完一段就接著寫下一段，直到第 4 段。）\n\n" + "\n".join(lines)


ap = argparse.ArgumentParser()
ap.add_argument("--url"); ap.add_argument("--model", default="bonsai2b")
ap.add_argument("--split"); ap.add_argument("--transcripts"); ap.add_argument("--out")
ap.add_argument("--max-tokens", type=int, default=2000)
ap.add_argument("--parallel", type=int, default=1)
ap.add_argument("--temperature", type=float, default=0.2); ap.add_argument("--seed", type=int, default=-1)
ap.add_argument("--think", action="store_true", help="enable_thinking (reasoning kept out of prose by llama-server)")
ap.add_argument("--len1k", action="store_true", help="PROMPT_1K (~1000-token summary, fixed length)")
ap.add_argument("--prose500", action="store_true", help="PROMPT_PROSE_500 (~500 tokens)")
ap.add_argument("--prose500f", action="store_true", help="PROMPT_PROSE_500F (one fact + one anchor per sentence)")
ap.add_argument("--quarters", choices=["", "ranges", "markers"], default="", help="tell the model the four time ranges of the meeting; markers also tags them in the transcript")
ap.add_argument("--len-range", default="", help="replace the 350 至 450 length of the prompt, e.g. '800 至 1000'")
ap.add_argument("--prose-bal", action="store_true", help="PROMPT_PROSE_BAL (time-ordered, whole meeting, length grows with the meeting)")
ap.add_argument("--prose-rd", action="store_true", help="PROMPT_PROSE_RD (issue #1: topic sections, decisions/actions, proportional length)")
ap.add_argument("--rd-plan", action="store_true", help="2026-11-10 with --prose-rd: PROMPT_PROSE_RD_PLAN (one plan line before the summary; saved as 'plan', removed from 'prose'); the grammar gets the plan line")
ap.add_argument("--rd-spread", action="store_true", help="2026-11-10 with --rd-grammar: one topic per time slice, anchors restricted to that slice (coverage lever)")
ap.add_argument("--rd-grammar", action="store_true", help="2026-11-10 with --prose-rd: GBNF grammar that fixes the RD shape (3-6 topics, a duration-based number of one-anchor sentences per topic, 決定/待辦). rdsft1 stopped each topic after 1 sentence at T=0.2")
ap.add_argument("--prose500g", action="store_true", help="PROMPT_PROSE_500G (500F, no filler subject)")
ap.add_argument("--prose500b", action="store_true", help="PROMPT_PROSE_500B (500 + attribution/decision/number rules)")
ap.add_argument("--prose1k", action="store_true", help="PROMPT_PROSE_1K (synthesized prose with [m:ss] anchors)")
ap.add_argument("--len500", action="store_true", help="PROMPT_500 (~500-token summary)")
a = ap.parse_args()
if a.len500:
    PROMPT = PROMPT_500
if a.len1k:
    PROMPT = PROMPT_1K
if a.prose500:
    PROMPT = PROMPT_PROSE_500
if a.prose500f:
    PROMPT = PROMPT_PROSE_500F
if a.prose_bal:
    PROMPT = PROMPT_PROSE_BAL
if a.prose500g:
    PROMPT = PROMPT_PROSE_500G
if a.prose500b:
    PROMPT = PROMPT_PROSE_500B
if a.prose_rd:
    PROMPT = PROMPT_PROSE_RD_PLAN if a.rd_plan else PROMPT_PROSE_RD
if a.len_range:
    assert "350 至 450" in PROMPT
    PROMPT = PROMPT.replace("350 至 450", a.len_range)
if a.prose1k:
    PROMPT = PROMPT_PROSE_1K
os.makedirs(a.out, exist_ok=True)


def haiku(sid):   # --url haiku: Claude Haiku low effort via the claude CLI (eval/haiku_chat.py)
    o = os.path.join(a.out, sid + ".json")
    if os.path.exists(o):
        return
    sys.path.insert(0, os.path.dirname(__file__)); from haiku_chat import HaikuChat
    tx = open(os.path.join(a.transcripts, sid + ".txt")).read(); t0 = time.time()
    prose = HaikuChat()([{"role": "user", "content": PROMPT.format(tx=tx, len=rd_len(tx)) if a.prose_rd else PROMPT.format(tx=tx)}])
    if not prose:
        print(sid, "ERROR empty", flush=True); return
    json.dump({"prose": prose, "wall": time.time() - t0}, open(o, "w"), ensure_ascii=False)
    print(sid, f"{time.time()-t0:.0f}s", flush=True)


if a.url == "haiku":
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(haiku, json.load(open(a.split))["heldout"]))
    sys.exit(0)
for sid in json.load(open(a.split))["heldout"]:
    o = os.path.join(a.out, sid + ".json")
    if os.path.exists(o):
        continue
    tx = open(os.path.join(a.transcripts, sid + ".txt")).read()
    if a.quarters:
        tx = quarter_hint(tx, a.quarters)
    t0 = time.time()
    body = {
        "model": a.model, "messages": [{"role": "user", "content": (PROMPT.format(tx=tx, len=rd_len(tx)) if a.prose_rd else PROMPT.format(tx=tx))}],
        "temperature": a.temperature, "seed": a.seed, "max_tokens": a.max_tokens, "dry_multiplier": 0.8, "dry_penalty_last_n": 512, "repeat_penalty": 1.05, "repeat_last_n": 256,
        "chat_template_kwargs": {"enable_thinking": a.think}}
    if a.rd_grammar:
        body["grammar"] = rd_grammar(tx, a.rd_spread)
        if a.rd_plan:
            body["grammar"] = body["grammar"].replace("root ::= ", 'root ::= "# 計畫：" pitem ("；" pitem){2,5} "\\n\\n" ', 1) + 'pitem ::= [^\\n#；（）\\[]{2,30} "（" pt "–" pt "）"\npt ::= [0-9]{1,3} ":" [0-9]{2} (":" [0-9]{2})?\n'
            pf = os.path.join(os.environ.get("RD_PLAN_DIR", "/nonexistent"), sid + ".txt")   # 2026-11-11 diagnostic: forced plan line (is the plan the bottleneck of the position bias?)
            if os.path.exists(pf):
                fp = open(pf, encoding="utf-8").read().strip(); g = rd_grammar(tx, a.rd_spread)
                body["grammar"] = re.sub(r"topic\{[\d,]+\}", "topic{%d}" % (fp.count("；") + 1), g, 1).replace("root ::= ", "root ::= " + json.dumps(fp, ensure_ascii=False) + ' "\\n\\n" ', 1)
    if a.max_tokens <= 0:   # 0 = no cap: the server uses the whole remaining context
        body.pop("max_tokens")
    r = requests.post(a.url + "/chat/completions", json=body, timeout=36000).json()
    if "choices" not in r:
        print(sid, "ERROR", str(r)[:500], flush=True); continue
    m = r["choices"][0]["message"]; prose = (m.get("content") or "").strip()
    if not prose:
        print(sid, "EMPTY", r["choices"][0].get("finish_reason"), r.get("usage"), flush=True); continue
    plan, prose = split_plan(prose) if a.rd_plan else (None, prose)
    json.dump({"prose": prose, "plan": plan, "usage": r.get("usage"), "timings": r.get("timings"),
               "wall": time.time() - t0}, open(o, "w"), ensure_ascii=False)
    print(sid, r.get("usage"), f"{time.time()-t0:.0f}s", flush=True)
