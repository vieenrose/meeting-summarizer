#!/usr/bin/env bash
# Same screen as teacher_bakeoff.sh, but every candidate runs at once.
#
# The candidates are independent models behind one gateway, so running them in sequence made the
# screen take as long as the sum of all of them when it need only take as long as the slowest.
# Each candidate still runs its own sessions with modest concurrency, to stay polite per model.
set -uo pipefail
ROOT=/home/luigi/meeting-summarizer; cd $ROOT
SESSIONS=${SESSIONS:-$(cat data/bakeoff_sessions.txt)}
CANDIDATES=${CANDIDATES:-$(cat data/teacher_candidates.txt)}
mkdir -p logs runs/bakeoff reports

run_one() {
  local model=$1
  timeout 1800 python3 eval/run_singlepass.py --backend zen --model "$model" \
    --tokenizer Qwen/Qwen3.6-35B-A3B-FP8 --out "runs/bakeoff/${model}/singlepass" --parallel 4 \
    --max-output-tokens 24000 --only $SESSIONS >> "logs/bakeoff_${model}_singlepass.log" 2>&1
  timeout 1800 python3 eval/run_teacher.py --backend zen --model "$model" \
    --tokenizer Qwen/Qwen3.6-35B-A3B-FP8 --window-tokens 4000 --out "runs/bakeoff/${model}" \
    --parallel 4 --max-output-tokens 24000 --only $SESSIONS >> "logs/bakeoff_${model}_notes.log" 2>&1
  echo "done $model"
}

for model in $CANDIDATES; do run_one "$model" & done
wait
echo "all candidates finished"
