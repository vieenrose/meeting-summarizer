#!/usr/bin/env bash
# Round 2: the screen's survivors on a 20-session stratified subset, all in parallel.
# n=20 separates a large gap (90% vs 50%) but not 90% vs 80%; see the sample-size note in the
# report. Candidates are whatever cleared the 4-session screen, passed in via CANDIDATES.
set -uo pipefail
cd /home/luigi/meeting-summarizer
SESSIONS=$(cat data/subset20.txt)
CANDIDATES=${CANDIDATES:?set CANDIDATES}
while pgrep -f "run_singlepass|run_teacher" >/dev/null; do sleep 20; done
for model in $CANDIDATES; do
  (
    timeout 5400 python3 eval/run_teacher.py --backend zen --model "$model" \
      --tokenizer Qwen/Qwen3.6-35B-A3B-FP8 --window-tokens 4000 --out "runs/round2/${model}" \
      --parallel 5 --max-output-tokens 24000 --only $SESSIONS >> "logs/round2_${model}.log" 2>&1
    echo "done $model"
  ) &
done
wait
echo "round 2 finished"
