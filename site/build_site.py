"""Build a static site for reviewing the dataset in a browser.

The point is judgement, not decoration: every summary point sits above the transcript lines it
cites, so a reviewer can check a claim without searching. Anything that was changed after the
teacher produced it is stated on the page, and a third that was exempted from a rule says why.
A dataset that hides its repairs is harder to trust than one that lists them.

Output is plain files under site/dist, openable with no server.
"""
import argparse
import glob
import html
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.validate_teacher import check  # noqa: E402
from summarizer.ingest import parse_line, resolve_citation  # noqa: E402
from summarizer.pipeline import uncoverable_thirds  # noqa: E402

CSS = """
:root { --bg:#fbfaf8; --fg:#1a1a1a; --muted:#6b6b6b; --line:#e3e0da; --accent:#7a3b2e;
        --ok:#2f6b4f; --warn:#8a6d1f; --bad:#a33; --cite:#f3ede3; --card:#fff; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.65 "Iowan Old Style",
       "Source Serif Pro", Georgia, "Noto Serif CJK TC", "PingFang TC", serif; }
header { border-bottom:1px solid var(--line); padding:26px 32px 20px; background:var(--card); }
h1 { margin:0 0 3px; font-size:22px; letter-spacing:-.01em; }
h2 { font-size:17px; margin:26px 0 12px; }
.sub { color:var(--muted); font-size:14px; }
main { max-width:1200px; margin:0 auto; padding:26px 32px 80px; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th,td { text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); vertical-align:top; }
th { font-weight:600; color:var(--muted); font-size:12px; text-transform:uppercase;
     letter-spacing:.06em; }
td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; }
tr:hover td { background:#f5f2ec; }
a { color:var(--accent); text-decoration:none; }
a:hover { text-decoration:underline; }
.pill { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px;
        white-space:nowrap; }
.pass { background:#e6f0ea; color:var(--ok); }
.warn { background:#f6efda; color:var(--warn); }
.fail { background:#f6e6e6; color:var(--bad); }
.stats { display:flex; flex-wrap:wrap; gap:26px; margin:14px 0 4px; }
.stat b { display:block; font-size:21px; font-weight:600; letter-spacing:-.01em; }
.stat span { color:var(--muted); font-size:12.5px; text-transform:uppercase;
             letter-spacing:.05em; }
.point { border:1px solid var(--line); border-radius:6px; margin:0 0 18px; background:var(--card); }
.point > .txt { padding:14px 16px; }
.point > .ev { border-top:1px dashed var(--line); background:var(--cite); padding:10px 16px;
               font-size:13.5px; color:#333; }
.ev .ln { margin:2px 0; }
.ev .hit { font-weight:600; }
.ts { color:var(--accent); font-variant-numeric:tabular-nums; }
.meta { color:var(--muted); font-size:13px; margin:0 0 20px; }
.notes { font-size:13.5px; columns:2; column-gap:28px; }
.notes div { break-inside:avoid; margin:0 0 7px; }
.tag { font-size:11px; color:var(--warn); }
details { margin:18px 0; }
.prose { background:#fff; border:1px solid var(--line); border-radius:6px; padding:4px 18px; line-height:1.9; }
.prose p { margin:14px 0; }
.note { color:#8a6d3b; font-size:13px; border-top:1px dashed var(--line); padding-top:8px; }
summary { cursor:pointer; color:var(--accent); font-size:14px; }
pre { white-space:pre-wrap; font:13px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;
      background:var(--card); border:1px solid var(--line); padding:14px; border-radius:6px;
      max-height:460px; overflow:auto; }
.banner { border-radius:6px; padding:11px 15px; font-size:14px; margin:0 0 16px; }
.banner.repair { background:#f6efda; border:1px solid #e4d5ab; }
.banner.exempt { background:#eef1f5; border:1px solid #d5dbe4; }
.banner ul { margin:6px 0 0; padding-left:20px; }
nav { font-size:14px; }
nav a { margin-right:16px; }
"""

TS_RE = re.compile(r"[\[［](\d{1,2}(?::\d{2}){1,2})[\]］]")
THIRD_NAME = {0: "前段", 1: "中段", 2: "後段"}


def esc(s):
    return html.escape(str(s) if s is not None else "")


def body_len(p):
    return len(re.sub(r"[\[［][^\]］]*[\]］]|\s", "", p))


def page(title, body, nav_here=None):
    links = [("index.html", "sessions"), ("method.html", "how this was made")]
    nav = " ".join(f"<a href='{h}'>{'· ' if h == nav_here else ''}{t}</a>" for h, t in links)
    # The training monitor is a separate live server on port 8322. Point at it with whatever host
    # this page was reached by, so the link works by tailnet name or by IP alike.
    nav += (" <a id='monitor-link' href='#'>training monitor ↗</a>"
            "<script>document.getElementById('monitor-link').href="
            "location.protocol+'//'+location.hostname+':8322/';</script>")
    return (f"<!doctype html><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><style>{CSS}</style>"
            f"<header><h1>{esc(title)}</h1><nav class='sub'>{nav}</nav></header><main>{body}</main>")


def fmt_repair(r):
    """Repairs are recorded two ways: early ones as a sentence, semantic-review ones as a record."""
    if isinstance(r, str):
        return r
    where = "第 %s 點" % r.get("point") if r.get("point") else {
        "prose": "紀要", "note": "筆記", "note-ts": "筆記時間", "derivability": "全篇"
    }.get(r.get("surface") or r.get("stage"), "全篇")
    return "%s（%s）：%s" % (where, r.get("kind", ""), r.get("why", ""))


def render_session(session, run, lines, manifest, problems, rows_for):
    by_ts = {}
    for i, l in enumerate(lines):
        by_ts.setdefault(l.render().split("]")[0].lstrip("["), i)

    out = []
    m = manifest.get(session, {})
    out.append(f"<p class='meta'>{esc(m.get('meeting_name', '')[:200])}<br>"
               f"{esc(m.get('committee', ''))} · {esc(m.get('date', ''))} · "
               f"{m.get('duration_s', 0)//60} min · {len(lines)} transcript lines · "
               f"{run['windows']} windows · {len(run['notes'])} notes · "
               f"{rows_for} training rows · "
               f"<a href='{esc(m.get('ivod_url', '#'))}'>IVOD source</a></p>")

    if run.get("repairs"):
        items = "".join(f"<li>{esc(fmt_repair(r))}</li>" for r in run["repairs"])
        leaks = sum(1 for r in run["repairs"]
                    if not isinstance(r, str) and r.get("kind") == "REFERENCE LEAK")
        note = ("Each reuses the teacher's own wording or the transcript's; no summary prose was "
                "written by hand.")
        if leaks:
            note += (f" {leaks} of them replaced a correctly-spelled name or figure with the "
                     f"transcript's own garbled version, so that the summary stays derivable from "
                     f"what the model can actually read.")
        out.append(f"<div class='banner repair'><b>{len(run['repairs'])} correction"
                   f"{'s' if len(run['repairs']) > 1 else ''} after generation.</b> {note}"
                   f"<ul>{items}</ul></div>")
    exempt = uncoverable_thirds(lines)
    if exempt:
        names = "、".join(THIRD_NAME[t] for t in sorted(exempt))
        out.append(f"<div class='banner exempt'><b>{esc(names)} exempt from the coverage rule.</b> "
                   f"That stretch of transcript contains no intelligible speech — an ASR failure, "
                   f"not a summary failure — so nothing was invented to cover it.</div>")
    if problems:
        out.append(f"<div class='banner'><b>Validation failed:</b> {esc('; '.join(problems))}</div>")

    if run.get("prose"):
        paras = "".join(f"<p>{esc(x)}</p>" for x in run["prose"].split("\n\n") if x.strip())
        flag = ""
        if run.get("prose_problems"):
            flag = (f"<div class='note'>Does not meet the length rule: "
                    f"{esc('; '.join(run['prose_problems']))}</div>")
        out.append(f"<h2>Meeting note</h2><div class='prose'>{paras}{flag}</div>")

    out.append("<h2>The same meeting as checkable points</h2>")
    for point in run["summary"]:
        ev = []
        for c in TS_RE.findall(point):
            idx = by_ts.get(c)
            if idx is None:
                ev.append(f"<div class='ln'><span class='ts'>[{esc(c)}]</span> "
                          f"<i>no such line in the transcript</i></div>")
                continue
            for j in range(max(0, idx - 1), min(len(lines), idx + 3)):
                cls = "ln hit" if j == idx else "ln"
                ev.append(f"<div class='{cls}'>{esc(lines[j].render())}</div>")
        out.append(f"<div class='point'><div class='txt'>{esc(point)} "
                   f"<span class='sub'>({body_len(point)} chars)</span></div>"
                   f"<div class='ev'>{''.join(ev) or '<i>no citation</i>'}</div></div>")

    notes = "".join(
        f"<div><span class='ts'>[{esc(n['ts'])}]</span> "
        f"{('<span class=tag>(' + esc(n['tag']) + ')</span> ') if n.get('tag') else ''}"
        f"{esc(n['text'])}</div>" for n in run["notes"])
    out.append(f"<details><summary>All {len(run['notes'])} intermediate notes "
               f"(these are the per-window training targets)</summary>"
               f"<div class='notes'>{notes}</div></details>")
    out.append(f"<details><summary>Full transcript ({len(lines)} lines)</summary>"
               f"<pre>{esc(chr(10).join(l.render() for l in lines))}</pre></details>")
    return page(session, "".join(out))


METHOD = """
<h2>What this dataset is</h2>
<p>Thirty-nine Legislative Yuan committee sessions of 1.5–2.5 hours, recorded from the public IVOD
archive, transcribed on-device-style with VibeVoice-ASR-Streaming-1.5B, and summarised by a teacher
model. Each session carries the intermediate per-window notes as well as the final summary, because
the notes are what a phone-sized student is trained to produce.</p>

<h2>How the summary is produced</h2>
<p>Map-reduce over 4k-token windows. The controller reads every window once, in order, and there is
no stop action, so a window cannot be skipped. Each window must yield notes; a window with real
speech is not offered a “nothing new” exit, because given prior context the teacher would otherwise
decline while describing the votes it had just read. Windows are independent — cross-window linking
happens once, in the synthesis step, which sees every note.</p>
<p>Map-reduce was chosen by measurement, not preference. A reading agent with lookback and revise
actions was built and tested first; at 4k windows its optional actions fired 0.03 times per session
and revise never fired at all, while coverage came out level. The simpler pipeline needs no retries
and invents fewer citations.</p>

<h2>Why this teacher</h2>
<p>Twenty of seventy models on the gateway answer at all and answer in Traditional Chinese. Thirteen
cheap-tier candidates were screened on identical sessions and scored on how much of their output
survives validation. minimax-m3 and deepseek-v4-flash tied on quality; minimax-m3 was 11× faster for
the same work (7.2 minutes against 79.7 for four sessions), which decides it when a corpus has to be
generated. One candidate was rejected for writing Simplified characters into Traditional output;
another for emitting its English reasoning as the answer.</p>

<h2>What is checked</h2>
<p>Every summary must cite timestamps that exist in the transcript, reach each third of the meeting,
stay inside a length band, avoid repeated points, and leave no window unnoted. A third containing no
intelligible speech is exempt, because ASR failure is not summary failure and demanding coverage of
noise only invites invention.</p>

<h2>What was repaired, and what was not</h2>
<p>Six sessions initially failed the structural checks. Over-long points were split at the teacher's
own semicolons with citations dealt out in order. One uncovered third was filled from a note the
teacher had already written but the synthesis had not selected. Two were genuine gaps in the notes,
so the same teacher was replayed on exactly those lines and the summary rewritten. One was left
alone: its first third holds six Han characters across eleven lines, with a single English sentence
repeated 294 times.</p>
<p>Passing those checks says a summary is well-formed, not that it is true. So every session was then
read against its transcript, line by line, over twelve rounds of review — 132 defects found and
repaired, until a full round came back clean. The counts by kind are below, and each session page lists its own.</p>
<p>No summary prose was written by hand. Every repair reuses the teacher's own wording or the
transcript's, and every repaired session records what changed and why.</p>

<h2>What the review found</h2>
<p>The mechanical checks cannot see a point that is fluent, correctly cited and wrong. These are the
classes that turned up when people read the summaries against the source:</p>
<ul>
<li><b>Wrong pairing (17 misattributions, 1 fabrication).</b> In a budget meeting dozens of numbered
cases each get their own 減列 or 凍結 outcome, and the synthesis pairs a case with a neighbour's
result. Every component is real; only the pairing is wrong, which is why no format check catches it.</li>
<li><b>Proposal reported as outcome (17 number errors).</b> A figure read out when a case was
<i>proposed</i>, presented as what was <i>adopted</i> — 2億9476萬1千元 where the committee resolved on
改凍結100萬元.</li>
<li><b>Citation drift (16).</b> A timestamp that exists but points somewhere else: the right fact
sourced to a line about a different agency, or to the moment an article was deferred for printing
rather than where it passed.</li>
<li><b>Reference leaks (14).</b> The teacher recognising a garbled name and writing the real one —
馬文軍 as 馬文君, 行業監督院 as 監察院, 489億 as 48.9億. Every one is true of the world, which is
why eight rounds of review approved them. See below.</li>
<li><b>Reversals (6).</b> An outcome inverted: an article recorded as passing per the Executive Yuan
version when the transcript says 維持原規, or a party leader placed in the opposing caucus and made
to attack his own side.</li>
<li><b>Unsupported inference (6).</b> A note correcting the transcript rather than reporting it — an
inferred fiscal year that contradicted the rest of its own session, a room number split into a floor
and a room. Twice a note marked its guess 推測 or 應為 and the summary dropped the hedge, so an
ambiguous reading arrived as fact.</li>
</ul>

<h2>Why the summaries keep the transcript's mistakes</h2>
<p>A summary here may not contain a fact that is absent from its transcript, even when the fact is
correct. The student reads the noisy ASR and nothing else, so a training row that maps 馬文軍 to
馬文君 teaches it to supply a name it cannot see — to guess a plausible public figure whenever the
audio is unclear. That is the hallucination this project exists to prevent, taught deliberately.</p>
<p>So where the transcript says 中小兵 and the member is really 邱志偉, the summary says 中小兵. A
second, cleaner transcript of the same audio (the Legislative Yuan's own whisperx output) is used to
decide which reading of a garbled line is right, and never to add anything our transcript lost.
<code>eval/check_derivable.py</code> enforces this mechanically: it reports any name, organisation or
figure in the gold that never occurs in the session's transcript. That count is now zero.</p>

<h2>Known limits</h2>
<ul>
<li>The ASR is rough on names and institutions; 經費稽核委員會 appears as 經費清害委員會 in places. That
bounds how precise any summary can be.</li>
<li>Speaker labels are unreliable — one session produced 178 of 180 labels as a single speaker — so
the teacher is told to identify speakers from content, not from the tag.</li>
<li>Coverage against a human key-fact list is not yet measured. That list does not exist, and it is
the gate that decides whether these summaries are good, as opposed to merely well-formed.</li>
<li>Twelve rounds of review ended when a full pass over all 39 sessions found nothing new. That
is the stopping rule, not a proof: round 8 found a defect class the seven rounds before it had
passed over entirely, so the honest claim is that the known failure modes have been swept, not that
none remain. Name substitutions, the most frequent class, are now also checked mechanically by
<code>eval/check_near_miss_names.py</code>, which recovers 25 of the leaks the rounds found by hand
and reports none left.</li>
<li>Five sessions from 2017–18 predate the Legislative Yuan's published transcripts, so their
repairs rest on internal evidence alone and could not be cross-checked against a second ASR.</li>
</ul>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/gold/w4000")
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--manifest", default="data/ivod_eval_manifest.json")
    ap.add_argument("--train-rows", default="data/train/gold_rows.jsonl")
    ap.add_argument("--teacher", default="minimax-m3")
    ap.add_argument("--out", default="site/dist")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    manifest = {f"ivod_{s['ivod_id']}": s
                for s in json.load(open(args.manifest, encoding="utf-8"))["sessions"]}
    rows_by_session = Counter()
    if os.path.exists(args.train_rows):
        for line in open(args.train_rows, encoding="utf-8"):
            rows_by_session[json.loads(line)["session"]] += 1

    rows = []
    for path in sorted(glob.glob(os.path.join(args.run_dir, "*.json"))):
        if path.endswith(".score.json"):
            continue
        session = os.path.splitext(os.path.basename(path))[0]
        tpath = os.path.join(args.transcripts, f"{session}.txt")
        if not os.path.exists(tpath):
            continue
        run = json.load(open(path, encoding="utf-8"))
        lines = [parse_line(l) for l in open(tpath, encoding="utf-8").read().splitlines() if l.strip()]
        problems = check(run, lines, 0, 650, 0)
        open(os.path.join(args.out, f"{session}.html"), "w", encoding="utf-8").write(
            render_session(session, run, lines, manifest, problems, rows_by_session[session]))
        rows.append({
            "session": session, "problems": problems, "lines": len(lines),
            "windows": run["windows"], "notes": len(run["notes"]),
            "points": len(run["summary"]),
            "chars": sum(body_len(p) for p in run["summary"]),
            "cites": sum(1 for p in run["summary"] for _ in TS_RE.findall(p)),
            "rows": rows_by_session[session],
            "repairs": len(run.get("repairs") or []),
            "exempt": sorted(uncoverable_thirds(lines)),
            "minutes": manifest.get(session, {}).get("duration_s", 0) // 60,
            "committee": manifest.get(session, {}).get("committee", ""),
        })

    passed = sum(1 for r in rows if not r["problems"])
    chars = sorted(r["chars"] for r in rows)
    repaired = sum(1 for r in rows if r["repairs"])
    stats = [
        (len(rows), "sessions"),
        (f"{passed}/{len(rows)}", "pass validation"),
        (sum(r['minutes'] for r in rows) // 60, "hours of meeting"),
        (sum(r["notes"] for r in rows), "notes"),
        (sum(r["rows"] for r in rows), "training rows"),
        (sum(r["repairs"] for r in rows), "defects found and fixed"),
        (f"{repaired}/{len(rows)}", "sessions corrected"),
    ]
    head = ["<div class='stats'>"]
    for v, label in stats:
        head.append(f"<div class='stat'><b>{esc(v)}</b><span>{esc(label)}</span></div>")
    head.append("</div>")
    head.append(f"<p class='meta'>Teacher <b>{esc(args.teacher)}</b>, map-reduce over 4k-token "
                f"windows. Click a session to read its meeting note and check every summary point "
                f"against the transcript lines it cites. "
                f"<a href='method.html'>How this was made, and what the review found</a>.</p>")
    head.append("<p class='meta'>Every summary passes the format checks, so the numbers above say "
                "little on their own. What the sessions were then read for, and what that turned "
                "up, is on the <a href='method.html'>method page</a> — including why a summary "
                "keeps the transcript's wrong names instead of correcting them.</p>")

    body = head + ["<table><tr><th>session</th><th>committee</th><th class='n'>min</th>"
                   "<th class='n'>lines</th><th class='n'>win</th><th class='n'>notes</th>"
                   "<th class='n'>pts</th><th class='n'>chars</th><th class='n'>cites</th>"
                   "<th class='n'>rows</th><th>status</th></tr>"]
    for r in sorted(rows, key=lambda r: (bool(r["problems"]), -r["repairs"], r["session"])):
        if r["problems"]:
            status = f"<span class='pill fail'>{esc('; '.join(r['problems'])[:56])}</span>"
        else:
            status = "<span class='pill pass'>pass</span>"
        if r["repairs"]:
            status += f" <span class='pill warn'>{r['repairs']} repair"\
                      f"{'s' if r['repairs'] > 1 else ''}</span>"
        if r["exempt"]:
            status += f" <span class='pill warn'>" \
                      f"{esc('、'.join(THIRD_NAME[t] for t in r['exempt']))} exempt</span>"
        body.append(
            f"<tr><td><a href='{r['session']}.html'>{r['session']}</a></td>"
            f"<td>{esc(r['committee'])}</td><td class='n'>{r['minutes']}</td>"
            f"<td class='n'>{r['lines']}</td><td class='n'>{r['windows']}</td>"
            f"<td class='n'>{r['notes']}</td><td class='n'>{r['points']}</td>"
            f"<td class='n'>{r['chars']}</td><td class='n'>{r['cites']}</td>"
            f"<td class='n'>{r['rows']}</td><td>{status}</td></tr>")
    body.append("</table>")

    open(os.path.join(args.out, "index.html"), "w", encoding="utf-8").write(
        page("Meeting summarizer — dataset", "".join(body), nav_here="index.html"))
    open(os.path.join(args.out, "method.html"), "w", encoding="utf-8").write(
        page("How this dataset was made", METHOD, nav_here="method.html"))
    print(f"{len(rows)} sessions -> {args.out}/index.html  ({passed} pass, {len(rows)-passed} fail, "
          f"{sum(r['repairs'] for r in rows)} repairs recorded)")


if __name__ == "__main__":
    main()
