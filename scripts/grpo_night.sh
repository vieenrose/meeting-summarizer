#!/usr/bin/env bash
# Overnight GRPO on the summary step, data-parallel on both RTX 5090s.
#
# Waits for the rebalanced SFT adapter and its held-out evaluation, then trains GRPO from that
# adapter with two processes -- each GPU generates and trains, doubling samples per step over the
# single-GPU smoke runs. Checkpoints every 25 steps so any of them can be evaluated.
set -uo pipefail
ROOT=/home/luigi/meeting-summarizer; cd $ROOT
ADAPTER=runs/sft/gemma4-e2b-lora-s5/final
until [ -f $ADAPTER/adapter_model.safetensors ]; do sleep 30; done
while pgrep -f "run_student_hf.py|sft_gemma.py" >/dev/null; do sleep 30; done
echo "$(date '+%F %T') starting GRPO from $ADAPTER"
PYTORCH_ALLOC_CONF=expandable_segments:True \
  ~/.venvs/vllm/bin/python -m accelerate.commands.launch --num_processes 2 --mixed_precision bf16 \
  distill/grpo_synth.py --adapter $ADAPTER --out runs/grpo/gemma4-synth --epochs 20
echo "$(date '+%F %T') GRPO finished"
