#!/usr/bin/env bash
# Judge one realtime-agent run (runs/student/rt-<tag>): 25 sampled notes per session (nj25-rt-<tag>)
# and the minutes with judge_prose_tx, coverage with judge_prose, then minutes_report and rt_report.
set -uo pipefail
cd /home/luigi/meeting-summarizer
REPORT=reports/rt_bakeoff.txt
for tag in "$@"; do
python3 - $tag <<'P'
import glob, json, os, random, sys
src = f'runs/student/rt-{sys.argv[1]}'; dst = f'runs/student/nj25-rt-{sys.argv[1]}'
os.makedirs(dst, exist_ok=True)
for f in glob.glob(f'{src}/ivod_*.json'):
    s = os.path.basename(f); r = json.load(open(f)); rng = random.Random(s + '25')
    ns = rng.sample(r['notes'], min(25, len(r['notes'])))
    prose = ''.join(n['text'].rstrip('。') + f" [{n['ts']}]。" for n in ns)
    json.dump({'notes': r['notes'], 'prose': prose}, open(f'{dst}/{s}', 'w'), ensure_ascii=False)
P
done
TX=$(for t in "$@"; do echo -n "runs/student/rt-$t runs/student/nj25-rt-$t "; done)
CV=$(for t in "$@"; do echo -n "runs/student/rt-$t "; done)
JUDGE_SCRIPT=judge_prose_tx DIRS="$TX" bash scripts/v2_judge.sh >> logs/rt_bakeoff_judge.log 2>&1
JUDGE_SCRIPT=judge_prose DIRS="$CV" bash scripts/v2_judge.sh >> logs/rt_bakeoff_judge.log 2>&1
for t in "$@"; do echo "== rt-$t"; python3 eval/minutes_report.py --run rt-$t; done | tee -a $REPORT
python3 eval/rt_report.py | tee -a $REPORT
echo "JUDGED $*" | tee -a $REPORT
