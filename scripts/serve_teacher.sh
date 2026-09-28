#!/usr/bin/env bash
# Teacher: Qwen3.6-35B-A3B FP8 on both RTX 5090s with its built-in MTP head for speculative decoding.
# gpu-memory-utilization leaves ~6 GB on GPU 0 for VibeVoice transcription running alongside.
# Venv and CUDA 12.8 build per memory note host-cuda-lxc (vLLM 0.19.1+cu128, torch 2.10).
set -euo pipefail
PORT="${PORT:-8000}"
exec /home/luigi/.venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --served-model-name Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "${GPU_UTIL:-0.78}" \
  --max-model-len "${MAX_LEN:-16384}" \
  --max-num-seqs "${MAX_SEQS:-16}" \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}' \
  --port "$PORT"
