"""The shipped summarizer: read every window once, note it, then write one summary.

Chosen over the fuller reading agent after measuring both with a 35B teacher on 39 real sessions.
The agent's extra actions did not earn their complexity: LOOKBACK fired 0.03 times per session and
REVISE never, while coverage came out level (agent 0.64 empty windows of 6.36, map-reduce 0.59;
thirds cited 92% vs 90%). The simple shape also needs no retries and invents fewer citations
(0.08 per session vs 0.31), and it is far easier to distil into a 2B student and to re-implement
in Kotlin on the phone.

Shape:
  1. The controller cuts the transcript into in-order windows and reads each exactly once.
     There is no stop action, so a window can never be skipped.
  2. Each window must produce notes. A window with real speech is not offered the NOTHING-NEW
     exit: given prior context the teacher would otherwise answer NOTHING-NEW while describing the
     votes and decisions it had just read, and the window was lost.
  3. Windows are independent. Carrying notes forward is what caused that suppression, and it
     bought nothing measurable, so cross-window linking happens once, in the synthesis step,
     which sees every note.
  4. Synthesis writes ~6 numbered points, each citing a timestamp that exists in the transcript,
     with at least one point from each third of the meeting.
"""
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from summarizer.ingest import Line, extract_citations, format_ts, parse_line, resolve_citation

TAGS = ("DECISION", "ACTION", "NUMBER", "OPEN-ISSUE")

SYSTEM_PROMPT = """你是會議閱讀助理。你會依序看到一場會議逐字稿的片段（WINDOW），逐字稿為語音辨識結果，可能有錯字。
每個片段只看一次，看完後寫下筆記；所有片段讀完後，會用這些筆記寫出會議摘要。

每行逐字稿格式：[時間] 講者: 內容
注意：講者標籤（S1、S2）由語音辨識自動判斷，常把不同的人標成同一位，不可據此判斷是誰發言。
請從發言內容找出身分（例如「主席」「某某委員」「某部長」），無法確定時寫「發言者」，不要猜測。

針對目前片段，以繁體中文輸出 1 到 6 則筆記，每則自成一行，格式為：
- [時間] 發言者身分: 內容

規則：
1. 每則筆記開頭的時間必須完全照抄本片段中真實存在的一行時間，不可自行編造。
2. 先寫最重要的，不要依照發言順序填滿：決議與結果（通過、否決、保留、撤回、審查完畢、交付協商）
   優先於提案內容，提案內容優先於一般討論，一般討論優先於程序與寒暄。
   本片段若出現任何結論或裁示，必須記下，即使它出現在片段的最後一行。
3. 內容要寫出實際發生的事（誰主張什麼、是否通過或否決、金額、日期、條號），保留前因後果，不要只寫主題。
4. 屬於決議、待辦、數字或未決爭點時，可在時間前加上 (DECISION)、(ACTION)、(NUMBER) 或 (OPEN-ISSUE)。
5. 辨識錯字請依上下文判斷；無法判斷就照原文並註明。
只輸出筆記，不要輸出其他說明文字。"""

# v2 teacher prompt: the same task, with a rule added for each defect class that cost repairs in v1.
# v1 is kept verbatim above so the v1 corpus stays reproducible; PipelineConfig.prompt_version picks.
#
#   rule 5 (reversed)  v1 told the teacher to fix recognition errors from context. That produced the
#                      largest v1 defect class: the teacher recognised a garbled legislator or agency
#                      and wrote the real one (馬文軍 -> 馬文君, 行業監督院 -> 監察院). The student reads
#                      the garbled transcript, so a corrected name is a target it can never produce.
#   rule 6             outcomes attached to the wrong case in dense budget readings.
#   rule 7             a proposal figure reported as the adopted result.
#   rule 8             a guess stated as fact once the note reached the summary.
SYSTEM_PROMPT_V2 = SYSTEM_PROMPT.replace(
    "5. 辨識錯字請依上下文判斷；無法判斷就照原文並註明。",
    "5. 人名、黨團、機關名稱、案號、條號、金額、日期一律照抄逐字稿的寫法，即使明顯是語音辨識錯字也不要改成你認為正確的寫法"
    "（例如逐字稿寫「馬文軍」就寫馬文軍，不要改成真實人物的名字）。逐字稿沒有說出的名字不要自己補上；不確定是誰時寫「發言者」或「某委員」。\n"
    "6. 記錄決議或結果時，要寫出逐字稿同一段落中說出的案號或條號（例如「第27案」「第七條」），並且只寫該案自己的結果，"
    "不可把前後其他案的金額或結果接到這一案。提案人姓名以逐字稿中緊接在該案之後的名字為準。\n"
    "7. 金額要分清楚是「提案」「協商中」還是「決議」：提案宣讀時的數字寫「提案…」，最後通過的寫「決議…」；"
    "若協商過程中數字有更改，以最後定案的數字為決議，並可註明原提案數字。\n"
    "8. 內容不清楚或是推測時，保留「疑似」「可能」等字眼，不可寫成確定的事實。")

SYNTHESIS_RULES_V2 = ("4. 人名、機關、案號、金額照抄筆記的寫法，不要改成你認為正確的寫法。\n"
                      "5. 筆記中註明「疑似」「可能」「推測」的內容，摘要中也要保留這些字眼。\n"
                      "6. 金額要寫決議通過的數字，不要把提案數字寫成結果；每個結果都要對應到正確的案號。\n")

# Tolerant by design. Models write the note line several ways -- "- [t] x", "NOTE: [t] x",
# "1. [t] x", or a bare "[t] x" -- and a parser that accepts only one shape silently deletes good
# content: a strict version rejected 153 valid notes per session while the model was reading the
# meeting correctly. Accept any leading marker and full-width punctuation; require only the timestamp.
_NOTE_RE = re.compile(
    r"^(?:NOTE\s*[:：]|[-*•]|\d+[.、)])?\s*"
    r"(?:[\(（](DECISION|ACTION|NUMBER|OPEN-ISSUE)[\)）]\s*)?"
    r"[\[［](\d{1,2}(?::\d{2}){1,2})[\]］]\s*(.+)$")
_NOTHING_RE = re.compile(r"^NOTHING-NEW\s*[:：]\s*(.*)$")
_FORMAT_ECHOES = ("帶有前因後果", "完整句子筆記", "身分或姓名", "發言者身分")


EVIDENCE = re.compile(r"〔據「.*?」〕\s*")
EVIDENCE_RULE = "每則筆記在時間之後，先以〔據「…」〕照抄逐字稿中支持該筆記的原文（一句，不超過 60 字），再寫筆記。\n\n"


@dataclass
class Note:
    id: int
    window: int
    ts: str
    text: str
    tag: Optional[str] = None

    def render(self) -> str:
        tag = f"({self.tag}) " if self.tag else ""
        return f"note {self.id}: {tag}[{self.ts}] {self.text}"


@dataclass
class ParsedTurn:
    notes: List[Tuple[Optional[str], str, str]] = field(default_factory=list)  # (tag, ts, text)
    nothing_new: Optional[str] = None
    rejected: List[str] = field(default_factory=list)


def parse_turn(text: str) -> ParsedTurn:
    """Unrecognised lines are kept in `rejected` so a parser bug shows up instead of losing data."""
    out = ParsedTurn()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line in ("NOTE:", "NOTE：") or line.startswith(("<think>", "</think>")):
            continue
        if m := _NOTHING_RE.match(line):
            out.nothing_new = m.group(1).strip()
        elif m := _NOTE_RE.match(line):
            out.notes.append((m.group(1), m.group(2), m.group(3).strip()))
        else:
            out.rejected.append(line)
    return out


def uncoverable_thirds(lines: List[Line], min_han: int = 50) -> set:
    """Thirds that contain no speech worth summarising, and so cannot be required.

    ASR failure is not summary failure. One session in this corpus opens with 11 lines holding six
    Han characters between them and "You can't do that" repeated 294 times; demanding that a
    summary cite that stretch would only invite something invented to satisfy the check.
    """
    n = len(lines)
    if n < 3:
        return set()
    out = set()
    for t in (0, 1, 2):
        seg = lines[t * n // 3:(t + 1) * n // 3]
        han = sum(len(re.findall(r"[\u4e00-\u9fff]", l.text)) for l in seg)
        if han < min_han:
            out.add(t)
    return out


def make_windows(lines: List[Line], budget_tokens: int,
                 count_tokens: Callable[[str], int]) -> List[List[Line]]:
    """Greedy in-order packing on line boundaries; a single over-long line gets its own window."""
    windows: List[List[Line]] = []
    current: List[Line] = []
    used = 0
    for line in lines:
        cost = count_tokens(line.render()) + 1
        if current and used + cost > budget_tokens:
            windows.append(current)
            current, used = [], 0
        current.append(line)
        used += cost
    if current:
        windows.append(current)
    return windows


class ChatClient:
    """OpenAI-compatible chat endpoint (vLLM or llama.cpp). Thinking off: it burns the budget."""

    def __init__(self, base_url: str, model: str, max_tokens: int = 800, temperature: float = 0.0,
                 no_chat_template_kwargs: bool = False):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        # Mistral's tokenizer-mode server rejects chat_template_kwargs outright (it has no jinja
        # template to pass them to), unlike every Qwen-style server we otherwise target.
        self.no_chat_template_kwargs = no_chat_template_kwargs

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model, "messages": messages, "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if not self.no_chat_template_kwargs:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    return json.load(resp)["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                # A 400 is usually one oversized prompt, not a broken server: losing the whole
                # session to it would silently shrink the evaluation.
                detail = e.read()[:300].decode("utf-8", "replace")
                if attempt == 2 or e.code != 400:
                    print(f"  chat call failed ({e.code}): {detail}", flush=True)
                    return ""
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == 2:
                    print(f"  chat call failed ({type(e).__name__})", flush=True)
                    return ""
                time.sleep(2 * (attempt + 1))
        return ""


@dataclass
class PipelineConfig:
    window_tokens: int = 4000
    evidence: bool = False      # evidence-first notes: quote the transcript, then the note
    keep_evidence: bool = False  # pass the 〔據「...」〕 quote on to reduce with its note
    acts: bool = False          # annotated extracts: 〈speech act〉 before the span
    extract: bool = False       # extractive notes: speaker + verbatim span (distill/build_extractive_rows.py)
    nothing_new_min_chars: int = 200   # a window with more speech than this must produce notes
    max_notes_per_window: int = 6   # 4 truncated a window before its concluding line
    summary_points: int = 6
    max_point_chars: int = 130      # one 335-character point took a summary to 971
    max_summary_chars: int = 620
    prompt_version: str = "v1"      # "v2": the corpus-v2 teacher prompt, see SYSTEM_PROMPT_V2


class NotesPipeline:
    def __init__(self, chat: Callable[[List[Dict[str, str]]], str],
                 count_tokens: Callable[[str], int], config: PipelineConfig = PipelineConfig()):
        self.chat, self.count_tokens, self.cfg = chat, count_tokens, config

    def run(self, transcript: str) -> Dict:
        lines = [parse_line(l) for l in transcript.splitlines() if l.strip()]
        windows = make_windows(lines, self.cfg.window_tokens, self.count_tokens)
        notes: List[Note] = []
        log: List[Dict] = []

        for k, window in enumerate(windows, 1):
            block = "\n".join(l.render() for l in window)
            speech_chars = sum(len(l.text) for l in window)
            allow_nothing_new = speech_chars < self.cfg.nothing_new_min_chars
            system = SYSTEM_PROMPT_V2 if self.cfg.prompt_version == "v2" else SYSTEM_PROMPT
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": self._window_prompt(k, len(windows), block,
                                                                        allow_nothing_new)}]
            reply = self.chat(messages)
            turn = parse_turn(reply)
            log.append({"window": k, "messages": messages, "reply": reply, "rejected": turn.rejected})
            added = self._accept(turn, k, window, notes, lines)

            if added == 0 and not allow_nothing_new:
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content":
                              f"本片段有約 {speech_chars} 字的發言，必須寫下筆記。"
                              "請依格式輸出，每則以「- [時間] 」開頭，時間照抄本片段中的一行。"}]
                reply = self.chat(messages)
                turn = parse_turn(reply)
                log.append({"window": k, "messages": list(messages), "reply": reply,
                            "rejected": turn.rejected, "retry": "no-notes"})
                self._accept(turn, k, window, notes, lines)

        summary, summary_log = self._write(lines, notes)
        log.extend(summary_log)
        return {
            "mode": "notes",
            "windows": len(windows),
            "notes": [n.__dict__ for n in notes],
            "notes_tokens": self.count_tokens("\n".join(n.render() for n in notes)),
            "lookbacks": 0,
            "windows_without_notes": [k for k in range(1, len(windows) + 1)
                                      if not any(n.window == k for n in notes)],
            "summary": summary,
            "log": log,
        }

    def _window_prompt(self, k: int, total: int, block: str, allow_nothing_new: bool) -> str:
        rule = ("" if allow_nothing_new else
                "本片段有實質發言，必須寫下筆記，不得輸出 NOTHING-NEW。\n\n")
        from distill.build_extractive_rows import ACT_RULE, EXTRACT_RULE
        ev = ((EVIDENCE_RULE if self.cfg.evidence else "") + (EXTRACT_RULE if self.cfg.extract else "")
              + (ACT_RULE if self.cfg.acts else ""))
        return f"WINDOW {k}/{total}\n\n{rule}{ev}TRANSCRIPT\n{block}"

    def _accept(self, turn: ParsedTurn, k: int, window: List[Line], notes: List[Note],
                all_lines: List[Line]) -> int:
        """Keep a note whose anchor resolves anywhere in the transcript; drop invented ones.

        Measured: of 100 anchors a 2B student produced outside its window, none were absent from
        the transcript -- they were real earlier lines. Rejecting those would have scored the model
        as hallucinating when it had not.
        """
        added = 0
        for tag, ts, text in turn.notes[:self.cfg.max_notes_per_window]:
            if any(marker in text for marker in _FORMAT_ECHOES):
                turn.rejected.append(f"copied the format description [{ts}] {text}")
                continue
            if resolve_citation(ts, window) is None:
                if resolve_citation(ts, all_lines) is None:
                    turn.rejected.append(f"invented timestamp [{ts}] {text}")
                    continue
                turn.rejected.append(f"out-of-window timestamp [{ts}] {text}")
            # An evidence-first student quotes the transcript before its note (〔據「...」〕, see
            # distill/build_evidence_rows.py). The quote is scaffolding for the note, not content:
            # reduce sees the note alone.
            if not self.cfg.keep_evidence:
                text = EVIDENCE.sub("", text).strip()
            notes.append(Note(id=len(notes) + 1, window=k,
                              ts=format_ts(parse_line(f"[{ts}] x").start_s), text=text, tag=tag))
            added += 1
        return added

    def _write(self, lines: List[Line], notes: List[Note]) -> Tuple[List[str], List[Dict]]:
        notes_block = "\n".join(n.render() for n in notes)
        # Asking for more points than the notes support invites invention.
        n_points = max(1, min(self.cfg.summary_points, (len(notes) + 1) // 2 + 1))
        prompt = (f"以下是整場會議依序寫下的筆記：\n\n{notes_block}\n\n"
                  f"請寫出 {n_points} 點繁體中文會議摘要（只寫筆記中有的內容，不要推測立場或動機）：\n"
                  f"0. 每點 60 到 100 字（不含時間標記），不要把多個議題塞進同一點；寧可寫得精簡，也不要超長。\n"
                  f"1. 每點一行，以「1.」「2.」編號開頭，句尾附上出處時間，格式 [時間]，時間必須來自筆記。\n"
                  f"2. 優先寫決議、待辦與負責人、關鍵數字、爭議與未決事項，保留誰主張什麼。\n"
                  f"3. 會議前、中、後三段都要有內容。\n"
                  + (SYNTHESIS_RULES_V2 if self.cfg.prompt_version == "v2" else "")
                  + "只輸出編號清單。")
        messages = [{"role": "user", "content": prompt}]
        log: List[Dict] = []
        best: Tuple[List[str], List[str]] = ([], ["沒有編號清單"])
        for _ in range(2):
            reply = self.chat(messages)
            points = [p.strip() for p in reply.splitlines() if re.match(r"^\s*\d+[.、]", p)]
            problems = self._check_summary(points, lines)
            log.append({"step": "write", "messages": list(messages), "reply": reply, "problems": problems})
            if points and (not best[0] or len(problems) <= len(best[1])):
                best = (points, problems)
            if not problems:
                break
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": "請修正以下問題後重新輸出完整清單：\n" + "\n".join(problems)}]
        return best[0], log

    def _check_summary(self, points: List[str], lines: List[Line]) -> List[str]:
        """Thirds are measured by position in the transcript, not by clock time, so a long silent
        stretch cannot make a third impossible to cover."""
        if not points:
            return ["沒有編號清單"]
        problems, covered, n = [], set(), max(1, len(lines))
        total = len(re.sub(r"[\[［][^\]］]*[\]］]|\s", "", "".join(points)))
        if total > self.cfg.max_summary_chars:
            problems.append(f"總長 {total} 字，超過 {self.cfg.max_summary_chars} 字，請刪減最長的幾點")
        for i, point in enumerate(points, 1):
            body_len = len(re.sub(r"[\[［][^\]］]*[\]］]|\s", "", point))
            if body_len > self.cfg.max_point_chars:
                problems.append(f"第 {i} 點 {body_len} 字，過長，請拆開或精簡")
            cited = extract_citations(point)
            resolved = [(c, resolve_citation(c, lines)) for c in cited]
            indices = [idx for _, idx in resolved if idx is not None]
            if not indices:
                problems.append(f"第 {i} 點沒有有效的出處時間")
            # A point citing several timestamps used to pass on the strength of any one of them
            # resolving, so an invented timestamp sitting beside a real one was invisible. Corpus
            # v2 carried 8 of those through six review rounds (e.g. a [2:48:00] on a transcript
            # ending 2:47:32) before a full-corpus sweep caught them. Flag each bad one.
            for c, idx in resolved:
                if idx is None:
                    problems.append(f"第 {i} 點的出處時間 [{c}] 不存在於逐字稿")
            for idx in indices:
                covered.add(min(2, 3 * idx // n))
        missing = {0, 1, 2} - covered - uncoverable_thirds(lines)
        if missing:
            names = {0: "前段", 1: "中段", 2: "後段"}
            problems.append("缺少會議" + "、".join(names[m] for m in sorted(missing)) + "的內容")
        return problems
