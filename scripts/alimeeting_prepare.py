"""Convert AliMeeting far-field TextGrid transcripts into this project's transcript-line format.

AliMeeting (OpenSLR-119, CC BY-SA 4.0) ships one TextGrid per meeting, one tier per speaker, each
tier a list of {xmin, xmax, text} intervals in Simplified Chinese. This merges all tiers into a
single time-ordered transcript in the same "[H:MM:SS] Sn: text" contract as data/v2/transcripts,
converted to Traditional Chinese with the project's own s2twp converter, so the existing pipeline
(v2_generate.py, distill/v2_gates.py, the QA protocol) can run over it unmodified.

Deliberately audio-free: only textgrid_dir is extracted from each archive, never audio_dir.
"""
import argparse
import glob
import json
import os
import sys
import tarfile

import textgrid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from summarizer.ingest import format_ts, opencc_s2twp  # noqa: E402


def extract_textgrids(archive: str, part: str, out_dir: str) -> None:
    """Pull only .../{part}_Ali_far/textgrid_dir/*.TextGrid out of the tar.gz -- never audio.

    Eval/Test wrap both mic types in one archive (Eval_Ali/Eval_Ali_far/...); Train ships far and
    near as separate archives with no outer wrapper (Train_Ali_far/... directly) -- match on the
    suffix so both layouts work without knowing which one a given archive uses."""
    suffix_dir = f"{part}_Ali_far/textgrid_dir/"
    with tarfile.open(archive) as tf:
        members = [m for m in tf.getmembers()
                   if suffix_dir in m.name and m.name.endswith(".TextGrid")]
        tf.extractall(out_dir, members=members)


def convert_session(tg_path: str, convert) -> tuple:
    tg = textgrid.TextGrid.fromFile(tg_path)
    events = []
    speaker_order = []
    for tier in tg.tiers:
        if tier.name not in speaker_order:
            speaker_order.append(tier.name)
        label = f"S{speaker_order.index(tier.name) + 1}"
        for iv in tier.intervals:
            text = (iv.mark or "").strip()
            if not text:
                continue
            events.append((iv.minTime, label, text))
    events.sort(key=lambda e: e[0])
    lines = [f"[{format_ts(t)}] {spk}: {convert(text)}" for t, spk, text in events]
    duration = max((iv.maxTime for tier in tg.tiers for iv in tier.intervals), default=0)
    return lines, duration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="/media/big-disk-2/alimeeting/raw")
    ap.add_argument("--work-dir", default="/media/big-disk-2/alimeeting/extract")
    ap.add_argument("--out-transcripts", default="data/alimeeting/transcripts")
    ap.add_argument("--out-manifest", default="data/alimeeting/manifest.json")
    ap.add_argument("--parts", nargs="+", default=["Eval", "Test"])
    args = ap.parse_args()
    os.makedirs(args.out_transcripts, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_manifest), exist_ok=True)

    convert = opencc_s2twp()
    sessions = []
    # A part is "done" once every session it contains has a transcript on disk AND this marker
    # exists -- checked here, not by whether the (deletable) extract/raw dirs still have content,
    # so re-running this script never re-triggers extraction for a part already converted, and
    # never risks deleting a not-yet-extracted archive because a DIFFERENT part's step ran first.
    done_marker = lambda part: os.path.join(args.out_transcripts, f".done_{part}")
    for part in args.parts:
        if os.path.exists(done_marker(part)):
            part_sessions = json.load(open(done_marker(part)))
            print(f"{part}: already converted (marker present, {len(part_sessions)} sessions), "
                  f"skipping", flush=True)
            sessions.extend(part_sessions)
            continue
        # Eval/Test ship one combined archive; Train ships far/near separately with no wrapper dir.
        archive_name = f"{part}_Ali_far.tar.gz" if part == "Train" else f"{part}_Ali.tar.gz"
        archive = os.path.join(args.raw_dir, archive_name)
        pattern = os.path.join(args.work_dir, "**", f"{part}_Ali_far", "textgrid_dir", "*.TextGrid")
        tg_paths = sorted(glob.glob(pattern, recursive=True))
        if not tg_paths:
            print(f"extracting textgrid_dir from {archive} ...", flush=True)
            extract_textgrids(archive, part, args.work_dir)
            tg_paths = sorted(glob.glob(pattern, recursive=True))
        part_sessions = []
        for tg_path in tg_paths:
            stem = os.path.splitext(os.path.basename(tg_path))[0]  # R8001_M8004
            sid = f"alimeeting_{stem}"
            lines, duration = convert_session(tg_path, convert)
            dest = os.path.join(args.out_transcripts, f"{sid}.txt")
            open(dest, "w", encoding="utf-8").write("\n".join(lines) + "\n")
            part_sessions.append({
                "session_id": sid, "source_part": part, "room_meeting": stem,
                "duration_s": round(duration), "lines": len(lines),
                "source": "AliMeeting (OpenSLR-119, CC BY-SA 4.0), far-field track",
            })
            print(f"{sid}: {len(lines)} lines, {duration / 60:.0f} min", flush=True)
        json.dump(part_sessions, open(done_marker(part), "w", encoding="utf-8"), ensure_ascii=False)
        sessions.extend(part_sessions)
        # Safe now: this part's transcripts are on disk and its marker is written, so the source
        # archive (and whatever this part alone extracted into work_dir) is no longer needed --
        # deleting it here, immediately after THIS part's own success, can never destroy a
        # different part's not-yet-extracted archive the way deferring cleanup to the end of a
        # multi-part run did.
        if os.path.exists(archive):
            os.remove(archive)
        part_workdir = os.path.join(args.work_dir, f"{part}_Ali") if part != "Train" else None
        if part_workdir and os.path.isdir(part_workdir):
            import shutil
            shutil.rmtree(part_workdir)
        elif part == "Train":
            train_workdir = os.path.join(args.work_dir, "Train_Ali_far")
            if os.path.isdir(train_workdir):
                import shutil
                shutil.rmtree(train_workdir)

    json.dump({"sessions": sessions}, open(args.out_manifest, "w", encoding="utf-8"),
               ensure_ascii=False, indent=1)
    print(f"{len(sessions)} sessions -> {args.out_transcripts}, manifest -> {args.out_manifest}")


if __name__ == "__main__":
    main()
