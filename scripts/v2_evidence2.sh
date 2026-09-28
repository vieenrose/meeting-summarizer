#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python; R=reports/v2_evidence2.txt; AD=runs/sft/minicpm5-evidence2/final
source ~/.venvs/vllm/bin/activate
CUDA_VISIBLE_DEVICES=0 python3 distill/sft_gemma.py --base openbmb/MiniCPM5-2B --rows data/train/evidence_rows.jsonl \
  --synth-repeat 3 --epochs 2 --out runs/sft/minicpm5-evidence2 > logs/sft_evidence2.log 2>&1
deactivate
echo "eval_loss $(grep -o "'eval_loss': '[0-9.a-z]*'" logs/sft_evidence2.log | tr '\n' ' ')" | tee -a $R
CUDA_VISIBLE_DEVICES=0 timeout 7200 $PY eval/run_student_vllm.py --base openbmb/MiniCPM5-2B --adapter $AD \
  --evidence --out runs/student/v2-minicpm5-evidence2 --max-model-len 16384 > logs/eval_evidence2.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 10
CUDA_VISIBLE_DEVICES=0 $PY eval/prose_from_gold.py --base openbmb/MiniCPM5-2B --adapter $AD --evidence \
  --out runs/student/v2-minicpm5-evidence2-goldnotes > logs/pfg_evidence2.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 10
D="runs/student/v2-minicpm5-evidence2 runs/student/v2-minicpm5-evidence2-goldnotes"
python3 eval/score_v2.py $D | tee -a $R
JUDGE_SCRIPT=judge_prose_tx DIRS="$D" bash scripts/v2_judge.sh >> logs/v2_judge_ev2.log 2>&1
grep sentences reports/v2_sft_night.txt | tail -2 | tee -a $R
echo EV2 DONE | tee -a $R
