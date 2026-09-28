#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python
CUDA_VISIBLE_DEVICES=0 timeout 7200 $PY eval/run_student_vllm.py --base openbmb/MiniCPM5-2B --adapter runs/sft/minicpm5-v2-ivod-ali/final \
  --out runs/student/v2-minicpm5-w2k --max-model-len 16384 --window-tokens 2000 > logs/eval_w2k.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 15
python3 eval/score_v2.py runs/student/v2-minicpm5-w2k | tee -a reports/v2_w2k.txt
JUDGE_SCRIPT=judge_prose_tx DIRS=runs/student/v2-minicpm5-w2k bash scripts/v2_judge.sh >> logs/v2_judge_w2k.log 2>&1
grep sentences reports/v2_sft_night.txt | tail -1 | tee -a reports/v2_w2k.txt
