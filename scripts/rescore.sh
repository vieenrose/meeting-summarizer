#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/luigi/meeting-summarizer; cd $ROOT
STATUS=$ROOT/logs/overnight_status.txt
say() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
gpus_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -lt 2000 ]; }
pkill -f "[v]llm.entrypoints.openai.api_server" 2>/dev/null; pkill -f "[l]lama-server" 2>/dev/null
until gpus_free; do sleep 10; done
say "=== S: rescore with asymmetric judge context"
nohup /home/luigi/.venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model baidu/ERNIE-4.5-21B-A3B-PT --served-model-name judge --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 --max-model-len 32768 --max-num-seqs 16 --trust-remote-code \
  --enable-prefix-caching --port 8000 > logs/judge_rescore.log 2>&1 &
waited=0
until curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null; do
  pgrep -f "[v]llm.entrypoints.openai.api_server" >/dev/null || { say "S: judge died"; exit 1; }
  sleep 10; waited=$((waited+10)); [ $waited -gt 2400 ] && { say "S: timeout"; exit 1; }
done
for d in runs/final2/mapreduce/w4000 runs/final2/agent/w4000; do
  say "S: $d"
  python3 eval/score.py --run-dir "$d" --keyfacts data/keyfacts_silver_core \
    --judge-url http://127.0.0.1:8000/v1 --judge-model judge 2>&1 | tail -2 | tee -a "$STATUS"
done
pkill -f "[v]llm.entrypoints.openai.api_server" 2>/dev/null
say "=== S finished"
