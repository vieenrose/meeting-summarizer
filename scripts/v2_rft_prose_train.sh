#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python; R=reports/v2_rft_prose.txt; AD=runs/sft/minicpm5-rftprose/final
source ~/.venvs/vllm/bin/activate
CUDA_VISIBLE_DEVICES=0 python3 distill/sft_gemma.py --base openbmb/MiniCPM5-2B \
  --rows data/train/v2_gold_rows_all.jsonl data/train/alimeeting_gold_rows.jsonl data/train/rft_prose_rows.jsonl:2 \
  --synth-repeat 3 --epochs 2 --out runs/sft/minicpm5-rftprose > logs/sft_rftprose.log 2>&1
deactivate
echo "eval_loss $(grep -o "'eval_loss': '[0-9.a-z]*'" logs/sft_rftprose.log | tr '\n' ' ')" | tee -a $R
CUDA_VISIBLE_DEVICES=0 timeout 7200 $PY eval/run_student_vllm.py --base openbmb/MiniCPM5-2B --adapter $AD \
  --out runs/student/v2-minicpm5-rftprose --max-model-len 16384 > logs/eval_rftprose.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 10
python3 eval/score_v2.py runs/student/v2-minicpm5-rftprose | tee -a $R
python3 eval/prose_guard.py --run-dir runs/student/v2-minicpm5-rftprose --out runs/student/v2-minicpm5-rftprose-guard | tee -a $R
JUDGE_SCRIPT=judge_prose_tx DIRS="runs/student/v2-minicpm5-rftprose runs/student/v2-minicpm5-rftprose-guard" bash scripts/v2_judge.sh >> logs/v2_judge_rftprose.log 2>&1
grep sentences reports/v2_sft_night.txt | tail -2 | tee -a $R
echo RFTPROSE DONE | tee -a $R
