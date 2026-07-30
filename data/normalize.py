"""Normalize raw corpora into one intermediate utterance-level schema.

Output: data/normalized/{source}.jsonl, one meeting per line:

    {
      "id": "qmsum-train-012",
      "source": "qmsum",            # qmsum | meetingbank | dialogsum | vcsum
      "lang": "en" | "zh-TW",
      "split": "train" | "val" | "test",
      "utterances": [{"speaker": "Project Manager" | null, "text": "..."}],
      "refs": {"summary": "...", "topic": "...",            # whatever the corpus provides
               "queries": [{"query": "...", "answer": "..."}]},
      "meta": {...}
    }

No timestamps and no final string rendering here — that happens in render.py once the
transcript format scheme is fixed (configs/transcript_format.py). zh text is converted
to Traditional (OpenCC s2twp) at this stage; the original stays in meta["orig_script"].
"""
import json
import re
import sys
from pathlib import Path

from opencc import OpenCC

HERE = Path(__file__).parent
RAW = HERE / "raw"
OUT = HERE / "normalized"
OUT.mkdir(exist_ok=True)

s2twp = OpenCC("s2twp")  # simplified -> traditional with Taiwan phrasing


def write(name: str, rows: list) -> None:
    path = OUT / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_utt = sum(len(r["utterances"]) for r in rows)
    print(f"{name}: {len(rows)} meetings, {n_utt} utterances -> {path}")


# --------------------------------------------------------------------------- qmsum
# AMI/ICSI annotation markers ({disfmarker}, {vocalsound}, {gap}, {pause}, …) never
# appear in a real ASR transcript — drop them, then collapse the spaced-out punctuation.
_MARKER = re.compile(r"\{[a-z_]+\}")
_SPACED_PUNCT = re.compile(r"\s+([,.!?;:'])")


def _clean_qmsum(text: str) -> str:
    text = _MARKER.sub(" ", text)
    text = _SPACED_PUNCT.sub(r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def qmsum() -> None:
    rows = []
    for split_dir, split in [("train", "train"), ("val", "val"), ("test", "test")]:
        for p in sorted((RAW / "qmsum/data/ALL" / split_dir).glob("*.json")):
            d = json.loads(p.read_text())
            utts = []
            for t in d["meeting_transcripts"]:
                text = _clean_qmsum(t["content"])
                if text and re.search(r"\w", text):
                    utts.append({"speaker": t["speaker"].strip() or None, "text": text})
            general = d.get("general_query_list", [])
            specific = d.get("specific_query_list", [])
            rows.append({
                "id": f"qmsum-{split}-{p.stem}",
                "source": "qmsum",
                "lang": "en",
                "split": split,
                "utterances": utts,
                "refs": {
                    "summary": general[0]["answer"] if general else None,
                    "queries": [{"query": q["query"], "answer": q["answer"]}
                                for q in general[1:] + specific],
                    "topics": [t["topic"] for t in d.get("topic_list", [])],
                },
                "meta": {"file": p.name},
            })
    write("qmsum", rows)


# --------------------------------------------------------------------- meetingbank
_SENT = re.compile(r"(?<=[.!?])\s+")


def meetingbank() -> None:
    rows = []
    for split_file, split in [("train", "train"), ("validation", "val"), ("test", "test")]:
        for line in (RAW / f"meetingbank/{split_file}.jsonl").open():
            d = json.loads(line)
            text = d["transcript"].strip()
            # Flat ASR-style text, no speakers: sentence-split so chunking/rendering has
            # line granularity; speaker stays null (renderer emits an untagged line).
            utts = [{"speaker": None, "text": s.strip()}
                    for s in _SENT.split(text) if s.strip()]
            rows.append({
                "id": f"meetingbank-{split}-{d['uid']}",
                "source": "meetingbank",
                "lang": "en",
                "split": split,
                "utterances": utts,
                "refs": {"summary": d["summary"].strip() or None},
                "meta": {},
            })
    write("meetingbank", rows)


# ---------------------------------------------------------------------- dialogsum
_PERSON = re.compile(r"^#(Person\d+)#$")


def dialogsum() -> None:
    rows = []
    for split_file, split in [("train", "train"), ("validation", "val"), ("test", "test")]:
        for line in (RAW / f"dialogsum/{split_file}.jsonl").open():
            d = json.loads(line)
            utts = []
            for ln in d["dialogue"].split("\n"):
                ln = ln.strip()
                if not ln:
                    continue
                if ":" in ln:
                    spk, txt = ln.split(":", 1)
                    m = _PERSON.match(spk.strip())
                    spk = m.group(1).replace("Person", "Person ") if m else spk.strip()
                    utts.append({"speaker": spk, "text": txt.strip()})
                else:
                    utts.append({"speaker": None, "text": ln})
            rows.append({
                "id": f"dialogsum-{split}-{d['id']}",
                "source": "dialogsum",
                "lang": "en",
                "split": split,
                "utterances": utts,
                "refs": {"summary": d["summary"].strip(), "topic": d.get("topic", "").strip() or None},
                "meta": {},
            })
    write("dialogsum", rows)


# -------------------------------------------------------------------------- vcsum
def vcsum() -> None:
    # HF mirror layout: `context` = the transcript (one utterance per newline, NO speaker
    # tags), `discussion` = per-segment reference summaries, `summary` = overall summary.
    rows = []
    for split_file, split in [("train", "train"), ("dev", "val"), ("test", "test")]:
        for line in (RAW / f"vcsum/{split_file}.jsonl").open():
            d = json.loads(line)
            utts = [{"speaker": None, "text": s2twp.convert(ln.strip())}
                    for ln in d["context"].split("\n") if ln.strip()]
            if not utts:
                continue
            rows.append({
                "id": f"vcsum-{split}-{d['id']}",
                "source": "vcsum",
                "lang": "zh-TW",
                "split": split,
                "utterances": utts,
                "refs": {
                    "summary": s2twp.convert(d["summary"].strip()) if d.get("summary") else None,
                    "segment_summaries": [s2twp.convert(x) for x in (d.get("discussion") or [])] or None,
                    "agenda": [s2twp.convert(a) for a in (d.get("agenda") or [])] or None,
                },
                "meta": {"orig_script": "zh-Hans", "av_num": d.get("av_num")},
            })
    write("vcsum", rows)


if __name__ == "__main__":
    which = sys.argv[1:] or ["qmsum", "meetingbank", "dialogsum", "vcsum"]
    for name in which:
        globals()[name]()
