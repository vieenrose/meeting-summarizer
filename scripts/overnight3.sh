#!/usr/bin/env bash
# Third overnight pass: student bake-off through llama.cpp, on the quantizations the phone will run.
#
# vLLM cannot serve Gemma-4 in this environment (transformers raises on its per-layer head_dim), and
# serving bf16 would answer the wrong question anyway: the phone runs a 2-bit QAT GGUF for Gemma and
# a 4-bit GGUF for MiniCPM5, and the literature says quantization amplifies exactly the failure modes
# under test. So both students are served by llama.cpp from their deployment artifacts. GPU offload
# is used here for throughput only — protocol quality is what this measures, never speed, which has
# to be measured on the Reno7.
set -uo pipefail
ROOT=/home/luigi/meeting-summarizer
LLAMA=/home/luigi/llama.cpp/build-cuda/bin/llama-server
HUB=/home/luigi/.cache/huggingface/hub
cd $ROOT
STATUS=$ROOT/logs/overnight_status.txt
say() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

gpus_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -lt 2000 ]; }
stop_servers() {
  pkill -f "[v]llm.entrypoints.openai.api_server" 2>/dev/null
  pkill -f "[l]lama-server" 2>/dev/null
  until gpus_free; do sleep 10; done
}

find_gguf() { find -L "$HUB" -name "$1" -type f 2>/dev/null | head -1; }

serve_gguf() {  # serve_gguf <gguf> <log> [extra args...]
  local model=$1 log=$2; shift 2
  until gpus_free; do sleep 20; done
  nohup $LLAMA -m "$model" --host 127.0.0.1 --port 8000 -ngl 999 -c 16384 \
    --parallel 4 --no-warmup "$@" > "$log" 2>&1 &
  local waited=0
  until curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null; do
    pgrep -f "[l]lama-server" >/dev/null || { say "llama-server died, see $log"; tail -12 "$log" | tee -a "$STATUS"; return 1; }
    sleep 10; waited=$((waited + 10)); [ $waited -gt 900 ] && { say "llama-server timeout"; return 1; }
  done
  say "llama-server ready ($log)"
}

while pgrep -f "[h]f download" >/dev/null; do sleep 30; done
say "=== H: student bake-off on deployment quantizations (llama.cpp)"

GEMMA=$(find_gguf "gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf")
GEMMA_MTP=$(find_gguf "mtp-gemma-4-E2B-it.gguf")
MINICPM=$(find_gguf "MiniCPM5-2B-Q4_K_M.gguf")
say "H: gemma=${GEMMA:-MISSING} minicpm=${MINICPM:-MISSING}"

run_student() {  # run_student <tag> <hf tokenizer id>
  local tag=$1 tokenizer=$2
  python3 eval/run_teacher.py --base-url http://127.0.0.1:8000/v1 --model "$tag" \
    --tokenizer "$tokenizer" --transcripts data/subset --window-tokens 4000 \
    --out "runs/gguf_$tag" --parallel 2 >> "logs/gguf_${tag}.log" 2>&1 || say "H: $tag FAILED"
}

if [ -n "$MINICPM" ]; then
  stop_servers
  serve_gguf "$MINICPM" logs/gguf_minicpm_server.log && run_student minicpm5 openbmb/MiniCPM5-2B
fi
if [ -n "$GEMMA" ]; then
  stop_servers
  # The MTP drafter only speeds up generation; it changes no output, so it is left off here.
  serve_gguf "$GEMMA" logs/gguf_gemma_server.log && run_student gemma4e2b google/gemma-4-E2B-it
fi
stop_servers

python3 eval/report.py runs/gguf_*/w4000 runs/final2/mapreduce/w4000 \
  --out reports/gguf_bakeoff.json 2>&1 | tee -a "$STATUS"
say "=== overnight pass 3 finished"
