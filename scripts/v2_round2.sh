#!/usr/bin/env bash
# After notes-GRPO (GPU0) and MiniCPM5 SFT (GPU1): held-out eval of each, mechanical scores, judge.
set -uo pipefail
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python; REPORT=reports/v2_round2.txt
say() { echo "$(date '+%F %T') $*" | tee -a $REPORT; }
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
evaluate() {  # gpu out adapter [base]
  [ -f "$2/.done" ] && return
  say "eval $2"
  CUDA_VISIBLE_DEVICES=$1 timeout 7200 $PY eval/run_student_vllm.py --adapter "$3" ${4:+--base $4} \
    --out "$2" --max-model-len 16384 > logs/eval_$(basename $2).log 2>&1 && touch "$2/.done"
  tail -1 logs/eval_$(basename $2).log | tee -a $REPORT
  for p in $(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader); do kill $p; done
}
(
  while alive logs/grpo_v2_notes.pid; do sleep 60; done
  for ck in checkpoint-100 checkpoint-200 final; do
    [ -d runs/grpo/g4-v2-notes/$ck ] && evaluate 0 runs/student/v2-grpo-notes-$ck runs/grpo/g4-v2-notes/$ck
  done
) &
A=$!
(
  while alive logs/sft_minicpm5_v2.pid; do sleep 60; done
  say "MiniCPM5 SFT: $(grep -o "'eval_loss': '[0-9.]*'" logs/sft_minicpm5_v2.log | tr '\n' ' ')"
  evaluate 1 runs/student/v2-minicpm5 runs/sft/minicpm5-v2-ivod-ali/final openbmb/MiniCPM5-2B
) &
B=$!
wait $A $B
D=$(ls -d runs/student/v2-grpo-notes-* runs/student/v2-minicpm5 2>/dev/null | tr '\n' ' ')
python3 eval/score_v2.py runs/student/v2-ivod-ali $D | tee -a $REPORT
DIRS="$D" bash scripts/v2_judge.sh >> logs/v2_judge3.log 2>&1
grep -h "coverage" reports/v2_sft_night.txt | tail -$(echo $D | wc -w) | tee -a $REPORT
say "ROUND2 DONE"
