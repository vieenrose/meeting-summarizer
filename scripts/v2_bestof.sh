#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python
CUDA_VISIBLE_DEVICES=0 $PY eval/sample_prose.py --run-dirs runs/student/v2-minicpm5 --adapter runs/sft/minicpm5-v2-ivod-ali/final \
  --out-prefix runs/student/bo4-s > logs/bo4_sample.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 10
JUDGE_SCRIPT=judge_prose_tx DIRS="runs/student/bo4-s0 runs/student/bo4-s1 runs/student/bo4-s2 runs/student/bo4-s3" bash scripts/v2_judge.sh >> logs/v2_judge_bo4.log 2>&1
echo BO4 DONE > reports/v2_bo4.txt
