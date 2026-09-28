#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python; R=reports/v2_xact2.txt
source ~/.venvs/vllm/bin/activate
CUDA_VISIBLE_DEVICES=0 python3 distill/sft_gemma.py --base google/gemma-4-E4B-it --rows data/train/extract_act_reduce_fn_clean.jsonl \
  --epochs 3 --out runs/sft/e4b-xact-reduce2 > logs/sft_e4b_xact2.log 2>&1
deactivate
echo "eval_loss $(grep -o "'eval_loss': '[0-9.a-z]*'" logs/sft_e4b_xact2.log | tr '\n' ' ')" | tee -a $R
RED=$(ls -d runs/sft/e4b-xact-reduce2/final 2>/dev/null || ls -d runs/sft/e4b-xact-reduce2/checkpoint-* | tail -1)
CUDA_VISIBLE_DEVICES=0 $PY eval/prose_from_gold.py --base google/gemma-4-E4B-it --adapter $RED \
  --notes-dir runs/student/v2-minicpm5-xact --out runs/student/v2-xact-e4b2 > logs/pfg_xact2.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 10; rm -rf runs/merged/runs__sft__e4b*
python3 eval/prose_guard.py --run-dir runs/student/v2-xact-e4b2 --out runs/student/v2-xact-e4b2-guard | tee -a $R
python3 eval/score_v2.py runs/student/v2-xact-e4b2 | tee -a $R
JUDGE_SCRIPT=judge_prose_tx DIRS="runs/student/v2-xact-e4b2 runs/student/v2-xact-e4b2-guard" bash scripts/v2_judge.sh >> logs/v2_judge_xact2.log 2>&1
JUDGE_SCRIPT=judge_prose DIRS="runs/student/v2-xact-e4b2" bash scripts/v2_judge.sh >> logs/v2_judge_xact2.log 2>&1
grep -E "sentences|coverage" reports/v2_sft_night.txt | tail -3 | tee -a $R
echo XACT2 DONE | tee -a $R
