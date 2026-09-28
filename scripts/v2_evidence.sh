#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python; R=reports/v2_evidence.txt
source ~/.venvs/vllm/bin/activate
CUDA_VISIBLE_DEVICES=0 python3 distill/sft_gemma.py --base openbmb/MiniCPM5-2B --rows data/train/evidence_rows.jsonl \
  --synth-repeat 3 --epochs 2 --out runs/sft/minicpm5-evidence > logs/sft_evidence.log 2>&1
deactivate
echo "eval_loss $(grep -o "'eval_loss': '[0-9.a-z]*'" logs/sft_evidence.log | tr '\n' ' ')" | tee -a $R
CUDA_VISIBLE_DEVICES=0 timeout 7200 $PY eval/run_student_vllm.py --base openbmb/MiniCPM5-2B --adapter runs/sft/minicpm5-evidence/final \
  --evidence --out runs/student/v2-minicpm5-evidence --max-model-len 16384 > logs/eval_evidence.log 2>&1
for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done
echo EVIDENCE_GEN_DONE | tee -a $R
