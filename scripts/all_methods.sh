#!/usr/bin/env bash
# Single-pass control + iterative refine + judged comparison, in one run.
set -uo pipefail
ROOT=/home/luigi/meeting-summarizer; cd $ROOT
STATUS=$ROOT/logs/overnight_status.txt
say() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
gpus_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -lt 2000 ]; }
stop_servers() { pkill -f "[v]llm.entrypoints.openai.api_server" 2>/dev/null; pkill -f "[l]lama-server" 2>/dev/null; until gpus_free; do sleep 10; done; }
wait_server() {
  local log=$1 waited=0
  sleep 20
  while ! curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null; do
    if ! pgrep -f "[v]llm.entrypoints.openai.api_server" >/dev/null; then
      say "server died, see $log"; tail -8 "$log" | tee -a "$STATUS"; return 1
    fi
    sleep 10; waited=$((waited+10))
    if [ $waited -gt 2400 ]; then say "server timeout"; return 1; fi
  done
  return 0
}
serve() {  # serve <log> <extra args...>
  local log=$1; shift
  stop_servers
  nohup /home/luigi/.venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server "$@" --port 8000 > "$log" 2>&1 &
  wait_server "$log"
}

say "=== X: single-pass control (64k)"
if serve logs/teacher_64k.log --model Qwen/Qwen3.6-35B-A3B-FP8 --served-model-name Qwen/Qwen3.6-35B-A3B \
     --tensor-parallel-size 2 --gpu-memory-utilization 0.88 --max-model-len 65536 --max-num-seqs 4 \
     --enable-prefix-caching --limit-mm-per-prompt '{"image": 0, "video": 0}'; then
  say "X: teacher ready"
  python3 eval/run_singlepass.py --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.6-35B-A3B \
    --tokenizer Qwen/Qwen3.6-35B-A3B-FP8 --out runs/singlepass/w64k --parallel 3 \
    >> logs/singlepass.log 2>&1 || say "X: FAILED"
  say "X: $(ls runs/singlepass/w64k/*.json 2>/dev/null | wc -l) sessions"
fi

say "=== Y: iterative refine (deployable alternative)"
if serve logs/teacher_refine.log --model Qwen/Qwen3.6-35B-A3B-FP8 --served-model-name Qwen/Qwen3.6-35B-A3B \
     --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 16384 --max-num-seqs 16 \
     --enable-prefix-caching --limit-mm-per-prompt '{"image": 0, "video": 0}' \
     --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'; then
  say "Y: teacher ready"
  python3 eval/run_refine.py --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.6-35B-A3B \
    --tokenizer Qwen/Qwen3.6-35B-A3B-FP8 --out runs/refine/w4000 --parallel 6 \
    >> logs/refine.log 2>&1 || say "Y: FAILED"
  say "Y: $(ls runs/refine/w4000/*.json 2>/dev/null | wc -l) sessions"
fi

say "=== Z: judged comparison of all methods"
if serve logs/judge_all.log --model baidu/ERNIE-4.5-21B-A3B-PT --served-model-name judge \
     --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 32768 --max-num-seqs 16 \
     --trust-remote-code --enable-prefix-caching; then
  for d in runs/final2/mapreduce/w4000 runs/refine/w4000 runs/singlepass/w64k; do
    [ -d "$d" ] && [ -n "$(ls $d/*.json 2>/dev/null)" ] || { say "Z: $d empty"; continue; }
    say "Z: scoring $d"
    python3 eval/score.py --run-dir "$d" --keyfacts data/keyfacts_silver_core \
      --judge-url http://127.0.0.1:8000/v1 --judge-model judge 2>&1 | tail -2 | tee -a "$STATUS"
  done
fi
stop_servers
python3 eval/report.py runs/final2/mapreduce/w4000 runs/refine/w4000 runs/singlepass/w64k \
  --out reports/method_shootout.json 2>&1 | tee -a "$STATUS"
say "=== all methods finished"
