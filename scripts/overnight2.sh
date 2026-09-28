#!/usr/bin/env bash
# Second overnight pass, queued behind the first:
#   F  re-run the reading agent at 4k with the coverage fix (NOTHING-NEW withdrawn on windows with
#      speech; a window the notes suppressed is re-asked without them) and compare against map-reduce
#   G  silver key facts + judged coverage/faithfulness with GLM-4-32B, a family shared with neither
#      the Qwen teacher nor either student candidate
set -uo pipefail
ROOT=/home/luigi/meeting-summarizer
cd $ROOT
STATUS=$ROOT/logs/overnight_status.txt
say() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

gpus_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -lt 2000 ]; }
stop_server() { pkill -f "[v]llm.entrypoints.openai.api_server" 2>/dev/null; until gpus_free; do sleep 10; done; }

serve_vllm() {  # serve_vllm <model> <served-name> <log> [extra args...]
  local model=$1 name=$2 log=$3; shift 3
  until gpus_free; do sleep 20; done
  nohup /home/luigi/.venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$model" --served-model-name "$name" --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 --max-model-len 16384 --max-num-seqs 16 \
    --enable-prefix-caching --limit-mm-per-prompt '{"image": 0, "video": 0}' --port 8000 "$@" \
    > "$log" 2>&1 &
  local waited=0
  until curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null; do
    pgrep -f "[v]llm.entrypoints.openai.api_server" >/dev/null || { say "server died, see $log"; tail -15 "$log" | tee -a "$STATUS"; return 1; }
    sleep 10; waited=$((waited + 10)); [ $waited -gt 2400 ] && { say "server timeout"; return 1; }
  done
  say "server ready ($log)"
}

# Wait for the first pass and for the judge weights.
while pgrep -f "[o]vernight.sh" >/dev/null; do sleep 60; done
while pgrep -f "[h]f download zai-org/GLM-4-32B-0414" >/dev/null; do sleep 60; done
say "=== F: agent re-run with coverage fix"

stop_server
if serve_vllm Qwen/Qwen3.6-35B-A3B-FP8 Qwen/Qwen3.6-35B-A3B logs/teacher_server2.log \
     --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'; then
  python3 eval/run_teacher.py --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.6-35B-A3B \
    --tokenizer Qwen/Qwen3.6-35B-A3B-FP8 --transcripts data/transcripts \
    --window-tokens 4000 --mode agent --out runs/teacher_v2 --parallel 8 \
    >> logs/teacher_runs_v2.log 2>&1 || say "F: agent v2 FAILED"
  stop_server
  python3 eval/report.py runs/teacher_v2/agent/w4000 runs/teacher/agent/w4000 runs/teacher/mapreduce/w4000 \
    --out reports/coverage_fix.json 2>&1 | tee -a "$STATUS"
else
  say "F: teacher server failed"
fi

say "=== G: silver key facts and judged scoring (judge GLM-4-32B)"
stop_server
if serve_vllm zai-org/GLM-4-32B-0414 judge logs/judge_glm.log; then
  python3 eval/silver_keyfacts.py --base-url http://127.0.0.1:8000/v1 --model judge \
    --transcripts data/transcripts --out data/keyfacts_silver \
    >> logs/silver_keyfacts.log 2>&1 || say "G: silver key facts FAILED"
  say "G: silver key facts for $(ls data/keyfacts_silver/*.json 2>/dev/null | wc -l) sessions"
  for d in runs/teacher_v2/agent/w4000 runs/teacher/agent/w4000 runs/teacher/mapreduce/w4000; do
    [ -d "$d" ] || continue
    say "G: scoring $d"
    python3 eval/score.py --run-dir "$d" --keyfacts data/keyfacts_silver \
      --judge-url http://127.0.0.1:8000/v1 --judge-model judge 2>&1 | tail -2 | tee -a "$STATUS"
  done
  stop_server
else
  say "G: judge server failed"
fi
say "=== overnight pass 2 finished"
