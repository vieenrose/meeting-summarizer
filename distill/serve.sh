#!/usr/bin/env bash
# Teacher server for SINGLE-PASS distillation: Qwen3.5-9B Q6_K via llama-server,
# GPU 1 ONLY (GPU 0 reserved). 4 slots × 32k context; big ubatch for fast prefill of
# hour-long transcripts. OpenAI-compatible API on :8088.
set -euo pipefail
MODEL=$(ls ~/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/*/*Q6_K*.gguf | head -1)
export CUDA_VISIBLE_DEVICES=1
exec ~/llama.cpp/build/bin/llama-server \
  -m "$MODEL" \
  --alias teacher \
  -ngl 99 \
  -c $((4 * 40960)) \
  -np 4 \
  -b 4096 -ub 2048 \
  --flash-attn on \
  --jinja \
  --port 8088 \
  --host 127.0.0.1
