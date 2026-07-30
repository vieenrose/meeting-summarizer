#!/usr/bin/env bash
# Fine-tuned Qwen3-0.6B checkpoint -> .litertlm bundle for VoxSumDroid (LiteRT-LM).
# Two steps (this litert-torch version has no output_format flag):
#   1. multi-signature tflite (prefill sigs up to 4096, kv cache 32768, dynamic int8)
#   2. pack .litertlm with the HF tokenizer + native qwen3 template metadata
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
  --prefill_seq_lens 8 --prefill_seq_lens 64 --prefill_seq_lens 128 \
  --prefill_seq_lens 256 --prefill_seq_lens 512 --prefill_seq_lens 1024 \
  --prefill_seq_lens 2048 --prefill_seq_lens 4096 \
  --kv_cache_max_len=32768 \
  --mask_as_input=True \
  --quantize=dynamic_int8

TFLITE=$(ls -t "$OUT"/voxsum-qwen3-0.6b*.tflite | head -1)
export/.venv/bin/python export/pack_litertlm.py "$TFLITE" "$CKPT" "$OUT/voxsum-qwen3-0.6b.litertlm"
ls -la "$OUT"

# Desktop smoke test (litert-lm CLI):
#   uvx litert-lm run "$OUT"/*.litertlm --prompt="Summarize: ..."
