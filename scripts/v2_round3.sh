#!/usr/bin/env bash
# DPO (synthetic negatives) and verify-pass SFT: held-out eval, then transcript-grounded judge.
set -uo pipefail
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python; R=reports/v2_round3.txt
say() { echo "$(date '+%F %T') $*" | tee -a $R; }
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
freegpu() { for p in $(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader); do kill $p; done; sleep 10; }
(
  while alive logs/dpo_notes_synth.pid; do sleep 60; done
  say "DPO done: $(grep -o "'rewards/accuracies': '[0-9.]*'" logs/dpo_notes_synth.log | tail -1)"
  CUDA_VISIBLE_DEVICES=1 timeout 7200 $PY eval/run_student_vllm.py --base runs/merged/runs__sft__minicpm5-v2-ivod-ali__final \
    --adapter runs/dpo/minicpm5-notes-synth/final --out runs/student/v2-minicpm5-dpo --max-model-len 16384 > logs/eval_dpo.log 2>&1
  tail -1 logs/eval_dpo.log | tee -a $R; freegpu 1
) &
A=$!
(
  while alive logs/sft_minicpm5_verify.pid; do sleep 60; done
  say "verify SFT done: $(grep -o "'eval_loss': '[0-9.a-z]*'" logs/sft_minicpm5_verify.log | tr '\n' ' ')"
  AD=runs/sft/minicpm5-v2-verify/final
  CUDA_VISIBLE_DEVICES=0 timeout 7200 $PY eval/run_student_vllm.py --base openbmb/MiniCPM5-2B --adapter $AD \
    --out runs/student/v2-minicpm5-mt --max-model-len 16384 > logs/eval_mt.log 2>&1
  tail -1 logs/eval_mt.log | tee -a $R; freegpu 0
  CUDA_VISIBLE_DEVICES=0 $PY eval/verify_pass.py --run-dir runs/student/v2-minicpm5-mt --adapter $AD \
    --out runs/student/v2-minicpm5-mt-verified > logs/verify_pass.log 2>&1
  tail -1 logs/verify_pass.log | tee -a $R; freegpu 0
  CUDA_VISIBLE_DEVICES=0 $PY eval/prose_from_gold.py --base openbmb/MiniCPM5-2B --adapter $AD \
    --notes-dir runs/student/v2-minicpm5-mt-verified --out runs/student/v2-minicpm5-mt-verified-prose > logs/prose_verified.log 2>&1
  tail -1 logs/prose_verified.log | tee -a $R; freegpu 0
) &
B=$!
wait $A $B
D="runs/student/v2-minicpm5-dpo runs/student/v2-minicpm5-mt runs/student/v2-minicpm5-mt-verified-prose"
python3 eval/score_v2.py $D | tee -a $R
JUDGE_SCRIPT=judge_prose_tx DIRS="$D" bash scripts/v2_judge.sh >> logs/v2_judge_r3.log 2>&1
grep "sentences" reports/v2_sft_night.txt | tail -3 | tee -a $R
say "ROUND3 DONE"
