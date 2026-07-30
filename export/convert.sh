#!/usr/bin/env bash
# Fine-tuned Qwen3-0.6B checkpoint -> .litertlm bundle for VoxSumDroid (LiteRT-LM).
#
# Uses ai-edge-torch (litert_torch) qwen3 example: builds multi-signature tflite
# (prefill+decode, kv cache 4096 = app nCtx) and packs a .litertlm with the HF tokenizer
# + chat template. Quant recipes: dynamic_int8 (safe default, ~586MB on 0.6B) or the
# mixed int4 recipe for the smaller artifact. Exact flag names can drift with
# litert-torch releases — check `--help` on first run.
#
# Env: needs its own venv (torch versions may differ from the training env):
#   uv venv export/.venv && uv pip install --python export/.venv -e ./ai-edge-torch
set -euo pipefail
CKPT=${1:?usage: convert.sh <checkpoint_dir> [out_dir]}
OUT=${2:-$(dirname "$0")/out}
mkdir -p "$OUT"
cd "$(dirname "$0")/.."

export/.venv/bin/python -m litert_torch.generative.examples.qwen.convert_v3_to_tflite \
  --checkpoint_path "$CKPT" \
  --model_size=0.6b \
  --output_path "$OUT" \
  --output_name_prefix=voxsum-qwen3-0.6b \
  --output_format=litertlm \
  --hf_tokenizer_model_path "$CKPT" \
  --kv_cache_max_len=32768 \
  --quantize=dynamic_int8

# Desktop smoke test (litert-lm CLI):
#   uvx litert-lm run "$OUT"/voxsum-qwen3-0.6b*.litertlm --prompt="Summarize: ..."
