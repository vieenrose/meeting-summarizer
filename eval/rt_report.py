"""One table for the realtime-agent bake-off (scripts/rt_judge.sh): per run under runs/student/rt-*,
faithfulness of notes and minutes (judge_prose_tx), coverage (judge_prose), protocol health, and
the phone timing recomputed from each trace's token counts with the model's measured Reno7 speeds.
"""
import glob
import json
import os

# Reno7 CPU, 8 threads, Q4_0, llama-bench pp512 / tg32 (tok/s).
PHONE = {"q35-2b": (40.4, 8.8), "lfm25-2.6b": (35.0, 8.0), "minicpm5-2b": (33.5, 9.2),
         "gemma4-e2b": (34.6, 7.0), "lfm25-1.2b": (78.1, 18.2), "lfm25-8b-a1b": (24.5, 9.0),
         "apodex-2b": (34.8, 8.3)}  # Qwen3.5-2B Q4_K_M: same architecture


def tx(name):
    p = f"reports/judge_prose_tx_{name}.json"
    if not os.path.exists(p):
        return None
    rows = [r for r in json.load(open(p, encoding="utf-8"))["rows"]
            if r["verdict"] in ("supported", "contradicted", "unsupported")]
    n = max(1, len(rows))
    return (sum(r["verdict"] == "contradicted" for r in rows) / n, sum(r["verdict"] == "unsupported" for r in rows) / n, len(rows))


def timing(trace, end, pp, tg):
    clock = worst = 0.0
    for t in trace:
        if "pp" not in t:
            continue
        clock = max(clock, t["arrive"]) + t["pp"] / pp + t["tg"] / tg
        if t["window"] != "overview":
            worst = max(worst, clock - t["arrive"])
    return worst, clock - end


def timing_chunked(trace, end, pp, tg):
    """Incremental prefill: a window's transcript tokens are prefilled while it is being spoken
    (spread evenly between the previous window's close and its own), so at close only the
    remaining prefill, the check turn and the decode are left. One CPU, tasks in arrival order."""
    clock = worst = 0.0
    prev_close = 0.0
    for t in trace:
        if "pp" not in t:
            continue
        if "reply" in t and t["window"] != "overview":   # a reading turn: its window streams in
            # prefill starts as the window starts arriving; it only runs late when it cannot keep up
            clock = max(max(clock, prev_close) + t["pp"] / pp, t["arrive"]) + t["tg"] / tg
            prev_close = t["arrive"]
        else:                                            # check turn, lookback, overview: at close
            clock = max(clock, t["arrive"]) + t["pp"] / pp + t["tg"] / tg
        if t["window"] != "overview":
            worst = max(worst, clock - t["arrive"])
    return worst, clock - end


def main():
    print(f"{'run':<16}{'notes/s':>8}{'notes contr':>12}{'unsup':>7}{'min contr':>10}{'unsup':>7}{'cov':>6}"
          f"{'parse':>7}{'lag max':>9}{'after':>7}{'lag chunked':>12}{'after':>7}")
    for d in sorted(glob.glob("runs/student/rt-*")):
        tag = os.path.basename(d)[3:]
        recs = [json.load(open(f, encoding="utf-8")) for f in glob.glob(f"{d}/ivod_*.json")]
        if not recs:
            continue
        notes = sum(len(r["notes"]) for r in recs) / len(recs)
        p = [r["timing"]["protocol"] for r in recs]
        parse = sum(x["actions"] for x in p) / max(1, sum(x["lines"] for x in p))
        nt, mt = tx(f"nj25-rt-{tag}"), tx(f"rt-{tag}")
        cov_p = f"reports/judge_prose_rt-{tag}.json"
        cov = json.load(open(cov_p))["coverage"] if os.path.exists(cov_p) else None
        lag = after = lagc = afterc = ""
        speed = PHONE.get(tag, PHONE["q35-2b"] if tag == "q38-27b" else None)  # 27B: its token volume at 2B speed
        if speed:
            ts = [timing(r["trace"], r["timing"]["meeting_s"], *speed) for r in recs]
            tc = [timing_chunked(r["trace"], r["timing"]["meeting_s"], *speed) for r in recs]
            lag, after = f"{max(t[0] for t in ts) / 60:.1f}m", f"{max(t[1] for t in ts) / 60:.1f}m"
            lagc, afterc = f"{max(t[0] for t in tc) / 60:.1f}m", f"{max(t[1] for t in tc) / 60:.1f}m"
        f = lambda x, i: f"{x[i]:.0%}" if x else "-"  # noqa: E731
        print(f"{tag:<16}{notes:>8.1f}{f(nt, 0):>12}{f(nt, 1):>7}{f(mt, 0):>10}{f(mt, 1):>7}"
              f"{(f'{cov:.2f}' if cov is not None else '-'):>6}{parse:>7.0%}{lag:>9}{after:>7}{lagc:>12}{afterc:>7}")


if __name__ == "__main__":
    main()
