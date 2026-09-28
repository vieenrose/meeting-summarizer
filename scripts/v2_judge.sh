#!/usr/bin/env bash
# Judge stage of the v2 SFT evaluation (split out of v2_sft_night.sh after its baseline evals were rerun).
set -uo pipefail
cd /home/luigi/meeting-summarizer
REPORT=reports/v2_sft_night.txt
~/.venvs/cu13/bin/python -m vllm.entrypoints.openai.api_server \
  --model bahadirakdemir/gemma-4-31B-it-text-fp8 --served-model-name judge --tensor-parallel-size 2 \
  --max-model-len 24000 --gpu-memory-utilization 0.85 --max-num-seqs 8 --port 8700 \
  > logs/judge_v2_night.log 2>&1 &
JUDGE=$!
until curl -sf localhost:8700/v1/models >/dev/null; do
  kill -0 $JUDGE 2>/dev/null || { echo "judge died" | tee -a $REPORT; exit 1; }
  sleep 15
done
for d in ${DIRS:-runs/v2/w4000}; do
  python3 eval/${JUDGE_SCRIPT:-judge_prose}.py --candidate $d --judge-url http://127.0.0.1:8700/v1 --judge-model judge ${JUDGE_EXTRA:-} \
    --out reports/${JUDGE_SCRIPT:-judge_prose}_${JUDGE_TAG:-}$(basename $d .jsonl).json 2>&1 | tail -1 | tee -a $REPORT
done
kill $JUDGE; sleep 20; kill -9 $JUDGE 2>/dev/null
echo JUDGE DONE | tee -a $REPORT
