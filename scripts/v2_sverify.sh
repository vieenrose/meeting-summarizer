#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
source ~/.venvs/vllm/bin/activate
CUDA_VISIBLE_DEVICES=0 python3 distill/sft_gemma.py --base openbmb/MiniCPM5-2B \
  --rows data/train/v2_gold_rows_all.jsonl data/train/alimeeting_gold_rows.jsonl data/train/sentence_verifier_rows.jsonl \
  --synth-repeat 3 --epochs 2 --out runs/sft/minicpm5-sverify > logs/sft_sverify.log 2>&1
echo SVERIFY_TRAINED
