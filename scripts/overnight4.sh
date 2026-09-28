#!/usr/bin/env bash
# Final overnight pass: one clean comparison with every fix in place, so the numbers are
# comparable. Earlier passes each ran with different code (NOTHING-NEW still offered, the
# copyable prompt placeholder, out-of-window citations discarded), so their agent-vs-map-reduce
# gap mixes protocol effects with harness bugs.
#
#   I  teacher agent and map-reduce at 4k on all sessions, final code
#   J  protocol metrics across every run
set -uo pipefail
ROOT=/home/luigi/meeting-summarizer
cd $ROOT
STATUS=$ROOT/logs/overnight_status.txt
say() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
gpus_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -lt 2000 ]; }
stop_servers() { pkill -f "[v]llm.entrypoints.openai.api_server" 2>/dev/null; pkill -f "[l]lama-server" 2>/dev/null; until gpus_free; do sleep 10; done; }

while pgrep -f "[o]vernight3.sh" >/dev/null; do sleep 60; done
say "=== I: clean teacher comparison (final code)"
stop_servers
nohup /home/luigi/.venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.6-35B-A3B-FP8 --served-model-name Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 16384 --max-num-seqs 16 \
  --enable-prefix-caching --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}' --port 8000 \
  > logs/teacher_server3.log 2>&1 &
waited=0
until curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null; do
  pgrep -f "[v]llm.entrypoints.openai.api_server" >/dev/null || { say "I: server died"; tail -12 logs/teacher_server3.log | tee -a "$STATUS"; exit 1; }
  sleep 10; waited=$((waited + 10)); [ $waited -gt 2400 ] && { say "I: server timeout"; exit 1; }
done
say "I: teacher ready"

for mode in agent mapreduce; do
  python3 eval/run_teacher.py --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.6-35B-A3B \
    --tokenizer Qwen/Qwen3.6-35B-A3B-FP8 --transcripts data/transcripts \
    --window-tokens 4000 --mode $mode --out runs/final2 --parallel 8 \
    >> logs/final_runs.log 2>&1 || say "I: $mode FAILED"
  say "I: $mode done"
done
stop_servers

say "=== J: protocol metrics, all runs"
python3 eval/report.py runs/final2/agent/w4000 runs/final2/mapreduce/w4000 \
  runs/teacher/agent/w4000 runs/teacher/mapreduce/w4000 runs/teacher_v2/agent/w4000 \
  --out reports/final_comparison.json 2>&1 | tee -a "$STATUS"
say "=== overnight pass 4 finished"
