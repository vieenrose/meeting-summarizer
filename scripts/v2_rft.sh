#!/usr/bin/env bash
set -uo pipefail
cd /home/luigi/meeting-summarizer
R=reports/v2_rft.txt; say() { echo "$(date '+%F %T') $*" | tee -a $R; }
source ~/.venvs/vllm/bin/activate
say "SFT MiniCPM5 on gold + 495 RFT notes rows"
CUDA_VISIBLE_DEVICES=0 python3 distill/sft_gemma.py --base openbmb/MiniCPM5-2B \
  --rows data/train/v2_gold_rows_all.jsonl data/train/alimeeting_gold_rows.jsonl data/train/rft_best.jsonl \
  --synth-repeat 3 --epochs 2 --out runs/sft/minicpm5-v2-rft > logs/sft_minicpm5_rft.log 2>&1
say "eval_loss: $(grep -o "'eval_loss': '[0-9.a-z]*'" logs/sft_minicpm5_rft.log | tr '\n' ' ')"
deactivate
CUDA_VISIBLE_DEVICES=0 timeout 7200 ~/vllm019/bin/python eval/run_student_vllm.py --base openbmb/MiniCPM5-2B \
  --adapter runs/sft/minicpm5-v2-rft/final --out runs/student/v2-minicpm5-rft --max-model-len 16384 \
  > logs/eval_v2-minicpm5-rft.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill $p; done; sleep 10
python3 eval/score_v2.py runs/student/v2-minicpm5 runs/student/v2-minicpm5-rft | tee -a $R
DIRS=runs/student/v2-minicpm5-rft bash scripts/v2_judge.sh >> logs/v2_judge4.log 2>&1
grep -h "v2-minicpm5" reports/v2_sft_night.txt | tail -2 | tee -a $R
say "RFT DONE"
