#!/usr/bin/env bash
# Bake-off of ~2B Q4_0 models on the realtime reading agent (eval/realtime_agent.py), same 10
# held-out sessions (data/split_rt_bakeoff.json). Two models at a time, one per GPU (upstream llama.cpp: the PrismML fork loops on context checkpoints). A model that
# always thinks is detected by the agent itself and switched to --nothink-prefill.
# Phone speeds: Reno7 CPU, 8 threads, llama-bench pp512 / tg32.
set -uo pipefail
cd /home/luigi/meeting-summarizer
M=~/Bonsai-demo/models
MODELS=(
  "q35-2b|$M/q35/Qwen3.5-2B-Q4_0.gguf|40.4|8.8"
  "lfm25-2.6b|$M/lfm/LFM2.5-2.6B-Q4_0.gguf|35.0|8.0"
  "minicpm5-2b|$M/small/minicpm5/MiniCPM5-2B-Q4_0.gguf|33.5|9.2"
  "gemma4-e2b|$M/small/gemma4e2b/gemma-4-E2B_q4_0-it.gguf|34.6|7.0"
  "lfm25-1.2b|$M/lfm/LFM2.5-1.2B-Instruct-Q4_0.gguf|78.1|18.2"
  "lfm25-8b-a1b|$M/lfm/LFM2.5-8B-A1B-Q4_0.gguf|24.5|9.0"
)
run_model() {   # gpu port spec
  local gpu=$1 port=$2 tag gguf pp tg extra=""
  IFS='|' read -r tag gguf pp tg <<< "$3"
  local out=runs/student/rt-$tag
  [ "$(ls $out 2>/dev/null | wc -l)" -ge 10 ] && return
  CUDA_VISIBLE_DEVICES=$gpu ~/llama.cpp/build-cuda/bin/llama-server -m $gguf -ngl 99 -c 131072 -np 4 --jinja \
    --port $port --alias rt > logs/rt_server_$tag.log 2>&1 &
  local srv=$!
  until curl -sf localhost:$port/health >/dev/null; do
    kill -0 $srv 2>/dev/null || { echo "$tag: server died"; return; }
    sleep 5
  done
  python3 eval/realtime_agent.py --url http://127.0.0.1:$port/v1 --model rt --parallel 4 $extra \
    --split data/split_rt_bakeoff.json --phone-pp $pp --phone-tg $tg --out $out 2>&1 \
    | grep -v -i warning | sed "s/^/$tag /"
  kill $srv; sleep 5; kill -9 $srv 2>/dev/null
}
for ((i = 0; i < ${#MODELS[@]}; i += 2)); do
  run_model 0 8110 "${MODELS[i]}" &
  [ -n "${MODELS[i+1]:-}" ] && run_model 1 8111 "${MODELS[i+1]}" &
  wait
done
echo "GEN DONE" | tee -a reports/rt_bakeoff.txt
