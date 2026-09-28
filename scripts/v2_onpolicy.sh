#!/usr/bin/env bash
set -uo pipefail
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python; R=reports/v2_onpolicy.txt
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True $PY distill/dpo_notes.py --pairs data/train/onpolicy_pairs.jsonl \
  --epochs 3 --out runs/dpo/minicpm5-notes-onpolicy > logs/dpo_onpolicy.log 2>&1
grep "preference pairs" logs/dpo_onpolicy.log | tee -a $R
grep -o "'rewards/accuracies': '[0-9.]*'" logs/dpo_onpolicy.log | tail -1 | tee -a $R
CUDA_VISIBLE_DEVICES=0 timeout 7200 $PY eval/run_student_vllm.py --base runs/merged/runs__sft__minicpm5-v2-ivod-ali__final \
  --adapter runs/dpo/minicpm5-notes-onpolicy/final --out runs/student/v2-minicpm5-dpo-onpolicy --max-model-len 16384 > logs/eval_dpo_onpolicy.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 15
python3 eval/score_v2.py runs/student/v2-minicpm5-dpo-onpolicy | tee -a $R
JUDGE_SCRIPT=judge_prose_tx DIRS=runs/student/v2-minicpm5-dpo-onpolicy bash scripts/v2_judge.sh >> logs/v2_judge_onpolicy.log 2>&1
grep sentences reports/v2_sft_night.txt | tail -1 | tee -a $R
echo ONPOLICY DONE | tee -a $R
