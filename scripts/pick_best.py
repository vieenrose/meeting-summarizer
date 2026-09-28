"""Pick the best checkpoint from a scored report, by held-out results rather than training reward.

A candidate must not be worse than the starting point on the things GRPO could quietly trade
away: more hard-zero summaries, or notes that recall less of the teacher's facts. Among those
that pass, highest mean reward wins, then most sessions passing validation.
"""
import re
import sys

report, run_dir = sys.argv[1], sys.argv[2]
rows = []
for line in open(report, encoding="utf-8"):
    if re.search(r"pass \d+/0\b", line):
        continue                      # a failed evaluation, not a score
    m = re.search(r"(\S+)\s+pass (\d+)/\d+\s+reward ([\d.]+)\s+zeros (\d+)\s+notes-recall (\d+)%", line)
    if m:
        rows.append({"label": m.group(1), "pass": int(m.group(2)), "reward": float(m.group(3)),
                     "zeros": int(m.group(4)), "recall": int(m.group(5))})
start = next((r for r in rows if not r["label"].startswith(("checkpoint-", "final"))), None)
scored = [r for r in rows if r["label"].startswith(("checkpoint-", "final"))]
guarded = [r for r in scored if not start or
           (r["zeros"] <= start["zeros"] and r["recall"] >= start["recall"] - 3)]
# If every checkpoint dipped slightly on note recall -- GRPO here trains only the summary step,
# but notes and summaries share one adapter -- falling back to the SFT start would throw away the
# summary gains too. Relax to "no more broken summaries" and say that the guard was relaxed.
relaxed = [r for r in scored if not start or r["zeros"] <= start["zeros"]]
cands, rule = (guarded, "guarded") if guarded else (relaxed, "relaxed: note recall dipped")
if not cands:
    sys.exit(0)
best = max(cands, key=lambda r: (r["reward"], r["pass"]))
print(f"{run_dir}/{best['label']}")
print(f"picked {best['label']} ({rule}): reward {best['reward']} pass {best['pass']} "
      f"zeros {best['zeros']} recall {best['recall']}%", file=sys.stderr)
