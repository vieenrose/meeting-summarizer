#!/usr/bin/env bash
# Pass 6: the results that decide the remaining open questions.
#   L  student bake-off on deployment quantizations (llama.cpp, GPU offload for throughput)
#   M  CPU-only memory probe against the 2.5 GB ceiling
#   N  judged scoring of every pipeline against the reduced (interpretable) silver facts
set -uo pipefail
ROOT=/home/luigi/meeting-summarizer
cd $ROOT
STATUS=$ROOT/logs/overnight_status.txt
say() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
gpus_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -lt 2000 ]; }
stop_servers() { pkill -f "[v]llm.entrypoints.openai.api_server" 2>/dev/null; pkill -f "[l]lama-server" 2>/dev/null; until gpus_free; do sleep 10; done; }

while pgrep -f "[h]f download" >/dev/null; do sleep 30; done

say "=== L: student bake-off on deployment quantizations"
bash scripts/overnight3.sh >> logs/overnight3c.log 2>&1 || say "L: failed"

say "=== M: CPU memory probe (2.5 GB ceiling)"
stop_servers
bash scripts/cpu_probe.sh >> logs/cpu_probe.log 2>&1 || say "M: failed"
grep -E "^===|peak RSS|MISSING" reports/cpu_probe.txt 2>/dev/null | tee -a "$STATUS"

say "=== N: judged scoring against reduced silver facts"
stop_servers
nohup /home/luigi/.venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model baidu/ERNIE-4.5-21B-A3B-PT --served-model-name judge --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 --max-model-len 32768 --max-num-seqs 16 --trust-remote-code \
  --enable-prefix-caching --port 8000 > logs/judge_ernie3.log 2>&1 &
waited=0
until curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null; do
  pgrep -f "[v]llm.entrypoints.openai.api_server" >/dev/null || { say "N: judge died"; tail -10 logs/judge_ernie3.log | tee -a "$STATUS"; exit 1; }
  sleep 10; waited=$((waited + 10)); [ $waited -gt 2400 ] && { say "N: timeout"; exit 1; }
done
say "N: judge ready"
for d in runs/final2/mapreduce/w4000 runs/final2/agent/w4000 runs/gguf_minicpm5/w4000 runs/gguf_gemma4e2b/w4000; do
  [ -d "$d" ] || continue
  say "N: scoring $d"
  python3 eval/score.py --run-dir "$d" --keyfacts data/keyfacts_silver_core \
    --judge-url http://127.0.0.1:8000/v1 --judge-model judge 2>&1 | tail -2 | tee -a "$STATUS"
done
stop_servers
say "=== pass 6 finished"
