#!/usr/bin/env bash
# Overnight Stage 0 pipeline. Each stage is idempotent and guarded: a stage that fails is logged
# and the run stops rather than feeding bad inputs to the next stage. Only one model server holds
# the GPUs at a time.
#
#   A  wait for transcription of every session
#   B  teacher (Qwen3.6-35B-A3B FP8 + MTP): reading agent and map-reduce at 4k / 2k / 8k windows
#   C  protocol metrics report (mechanical, no judge)
#   D  silver key facts + coverage/faithfulness scoring with an independent judge (Gemma-4-31B)
#   E  untrained student bake-off (MiniCPM5-2B vs Gemma-4-E2B) on a subset, same protocol
set -uo pipefail
ROOT=/home/luigi/meeting-summarizer
cd $ROOT
mkdir -p runs logs reports
STATUS=$ROOT/logs/overnight_status.txt
say() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

gpus_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -lt 2000 ]; }
wait_gpus() { until gpus_free; do sleep 20; done; }

stop_server() {
  pkill -f "[v]llm.entrypoints.openai.api_server" 2>/dev/null
  until gpus_free; do sleep 10; done
}

# start_server <log> <extra vllm args...>; waits until /v1/models answers, or gives up.
start_server() {
  local log=$1; shift
  wait_gpus
  nohup "$@" > "$log" 2>&1 &
  local waited=0
  until curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null; do
    pgrep -f "[v]llm.entrypoints.openai.api_server" >/dev/null || { say "server died, see $log"; tail -20 "$log" | tee -a "$STATUS"; return 1; }
    sleep 10; waited=$((waited + 10))
    [ $waited -gt 1800 ] && { say "server timeout after 30 min"; return 1; }
  done
  say "server ready ($log)"
}

serve_vllm() {  # serve_vllm <model> <served-name> <log> [extra args...]
  local model=$1 name=$2 log=$3; shift 3
  start_server "$log" /home/luigi/.venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$model" --served-model-name "$name" --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 --max-model-len 16384 --max-num-seqs 16 \
    --enable-prefix-caching --limit-mm-per-prompt '{"image": 0, "video": 0}' --port 8000 "$@"
}

# ---- A: transcription ------------------------------------------------------
say "=== A: waiting for transcription"
while pgrep -f "[t]ranscribe_watch.sh" >/dev/null; do sleep 60; done
N_AUDIO=$(ls data/audio/*.m4a 2>/dev/null | wc -l)
N_TXT=$(ls data/transcripts/*.txt 2>/dev/null | wc -l)
say "A: $N_TXT transcripts of $N_AUDIO sessions"
[ "$N_TXT" -lt "$N_AUDIO" ] && { say "A: incomplete, stopping"; exit 1; }
python3 - <<'PY' | tee -a "$STATUS"
import glob, json
short = []
for p in sorted(glob.glob("data/transcripts/*.chunks.json")):
    d = json.load(open(p, encoding="utf-8"))
    live = sum(1 for c in d["chunks"] if c.strip() not in ("", "[Silence]", "[Music]"))
    if live / max(1, len(d["chunks"])) < 0.15:
        short.append((p.split("/")[-1][:-12], round(live / len(d["chunks"]), 3)))
print(f"A: {len(short)} sessions under 15% speech (kept, flagged): {short}")
PY

# ---- B: teacher runs -------------------------------------------------------
say "=== B: teacher runs"
stop_server
serve_vllm Qwen/Qwen3.6-35B-A3B-FP8 Qwen/Qwen3.6-35B-A3B logs/teacher_server.log \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}' || exit 1
for mode in agent mapreduce; do
  for w in 4000 2000 8000; do
    say "B: teacher $mode w$w"
    python3 eval/run_teacher.py --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.6-35B-A3B \
      --tokenizer Qwen/Qwen3.6-35B-A3B-FP8 --transcripts data/transcripts \
      --window-tokens $w --mode $mode --out runs/teacher --parallel 8 \
      >> logs/teacher_runs.log 2>&1 || say "B: $mode w$w FAILED (see logs/teacher_runs.log)"
  done
done
stop_server
say "B: done"

# ---- C: protocol metrics ---------------------------------------------------
say "=== C: protocol metrics"
python3 eval/report.py runs/teacher/agent/w* runs/teacher/mapreduce/w* \
  --out reports/protocol_metrics.json 2>&1 | tee -a "$STATUS"

# ---- D: silver key facts + judged scoring ----------------------------------
say "=== D: silver key facts and scoring (judge: Gemma-4-31B, independent of Qwen teacher)"
stop_server
if serve_vllm google/gemma-4-31B-it judge logs/judge_server.log; then
  python3 eval/silver_keyfacts.py --base-url http://127.0.0.1:8000/v1 --model judge \
    --transcripts data/transcripts --out data/keyfacts_silver \
    >> logs/silver_keyfacts.log 2>&1 || say "D: silver key facts FAILED"
  for d in runs/teacher/agent/w4000 runs/teacher/mapreduce/w4000 runs/teacher/agent/w2000 runs/teacher/agent/w8000; do
    [ -d "$d" ] || continue
    say "D: scoring $d"
    python3 eval/score.py --run-dir "$d" --keyfacts data/keyfacts_silver \
      --judge-url http://127.0.0.1:8000/v1 --judge-model judge 2>&1 | tail -3 | tee -a "$STATUS"
  done
  stop_server
else
  say "D: judge server failed, skipping scoring"
fi

# ---- E: untrained student bake-off ----------------------------------------
say "=== E: untrained student bake-off on 8 sessions"
mkdir -p data/subset && ls data/transcripts/*.txt | head -8 | xargs -I{} cp {} data/subset/ 2>/dev/null
for spec in "openbmb/MiniCPM5-2B minicpm5" "google/gemma-4-E2B-it-qat-q4_0-unquantized gemma4e2b"; do
  set -- $spec
  model=$1 tag=$2
  stop_server
  if serve_vllm "$model" "$tag" "logs/student_${tag}.log"; then
    say "E: $tag agent w4000"
    python3 eval/run_teacher.py --base-url http://127.0.0.1:8000/v1 --model "$tag" \
      --tokenizer "$model" --transcripts data/subset --window-tokens 4000 --mode agent \
      --out "runs/student_$tag" --parallel 4 >> "logs/student_${tag}_runs.log" 2>&1 \
      || say "E: $tag FAILED"
  fi
done
stop_server
python3 eval/report.py runs/student_*/agent/w4000 runs/teacher/agent/w4000 \
  --transcripts data/transcripts --out reports/student_bakeoff.json 2>&1 | tee -a "$STATUS"
say "=== overnight pipeline finished"
