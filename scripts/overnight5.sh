#!/usr/bin/env bash
# Pass 5: the two stages that could not run earlier, once their weights exist.
#   H2  student bake-off on deployment quantizations (llama.cpp), retried
#   G2  silver key facts + judged scoring, with ERNIE-4.5-21B-A3B as judge
#       (GLM-4-32B needs 64 GB of weights and does not fit two 31 GB cards; Gemma cannot be
#        served by this vLLM build; ERNIE is Chinese-native and shares no family with the Qwen
#        teacher or either student candidate)
set -uo pipefail
ROOT=/home/luigi/meeting-summarizer
cd $ROOT
STATUS=$ROOT/logs/overnight_status.txt
say() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
gpus_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -lt 2000 ]; }
stop_servers() { pkill -f "[v]llm.entrypoints.openai.api_server" 2>/dev/null; pkill -f "[l]lama-server" 2>/dev/null; until gpus_free; do sleep 10; done; }

while pgrep -f "[o]vernight4.sh" >/dev/null; do sleep 60; done
while pgrep -f "[h]f download" >/dev/null; do sleep 60; done

say "=== H2: student bake-off retry"
bash scripts/overnight3.sh >> logs/overnight3b.log 2>&1 || say "H2: failed"

say "=== G2: judged scoring with ERNIE-4.5-21B-A3B"
stop_servers
nohup /home/luigi/.venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model baidu/ERNIE-4.5-21B-A3B-PT --served-model-name judge --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 --max-model-len 16384 --max-num-seqs 16 --trust-remote-code \
  --enable-prefix-caching --port 8000 > logs/judge_ernie.log 2>&1 &
waited=0
until curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null; do
  pgrep -f "[v]llm.entrypoints.openai.api_server" >/dev/null || { say "G2: judge died"; tail -12 logs/judge_ernie.log | tee -a "$STATUS"; exit 1; }
  sleep 10; waited=$((waited + 10)); [ $waited -gt 2400 ] && { say "G2: judge timeout"; exit 1; }
done
say "G2: judge ready"
python3 eval/silver_keyfacts.py --base-url http://127.0.0.1:8000/v1 --model judge \
  --transcripts data/transcripts --out data/keyfacts_silver >> logs/silver_keyfacts.log 2>&1 \
  || say "G2: silver key facts FAILED"
say "G2: silver key facts for $(ls data/keyfacts_silver/*.json 2>/dev/null | wc -l) sessions"
for d in runs/final2/agent/w4000 runs/final2/mapreduce/w4000 runs/gguf_minicpm5/agent/w4000 runs/gguf_gemma4e2b/agent/w4000; do
  [ -d "$d" ] || continue
  say "G2: scoring $d"
  python3 eval/score.py --run-dir "$d" --keyfacts data/keyfacts_silver \
    --judge-url http://127.0.0.1:8000/v1 --judge-model judge 2>&1 | tail -2 | tee -a "$STATUS"
done
stop_servers
say "=== overnight pass 5 finished"
