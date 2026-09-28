#!/usr/bin/env bash
# Overnight: GRPO on the summary step (GPU 0) with every checkpoint evaluated as it lands (GPU 1).
#
# Reward on training prompts only says the policy is learning the reward. Whether that carries to
# unseen meetings is what matters, so each checkpoint is run through the full pipeline on the 7
# held-out sessions and scored, and reports/grpo_checkpoints.txt grows one line per checkpoint.
# Merged models are ~10 GB and the disk has ~80 GB free, so each is deleted after scoring.
set -uo pipefail
ROOT=/home/luigi/meeting-summarizer; cd $ROOT
PY=~/vllm019/bin/python
TAG=${TAG:-run1}
STEPS=${STEPS:--1}
RUN=runs/grpo/g4it-vllm-$TAG
REPORT=reports/grpo_checkpoints_$TAG.txt
mkdir -p reports logs runs/grpo
# The notes stage waits on this PID and on the "finished" line below. It used to match process
# command lines for this script's name, and a leftover launcher shell whose command line merely
# contained the name kept it waiting for three hours with both GPUs idle.
echo $$ > runs/grpo/.summary-$TAG.pid
echo "$(date '+%F %T') GRPO from runs/sft/g4it-s5/final" | tee -a $REPORT
$PY eval/score_run.py "it-sft-s5 (start)=runs/student/g4it-sft-s5" | tee -a $REPORT

CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  $PY -m accelerate.commands.launch --num_processes 1 --mixed_precision bf16 \
  distill/grpo_vllm.py --adapter runs/sft/g4it-s5/final --out $RUN --epochs 40 --max-steps $STEPS \
  > logs/grpo_g4it_vllm_$TAG.log 2>&1 &
TRAIN=$!

evaluate() {   # evaluate <adapter dir> <label>
  local adapter=$1 label=$2 out=runs/student/grpo-$TAG-$2
  [ -f "$out/.done" ] && return
  CUDA_VISIBLE_DEVICES=1 timeout 1800 $PY eval/run_student_vllm.py --adapter "$adapter" --out "$out" \
    > logs/eval_grpo_${TAG}_$label.log 2>&1 && touch "$out/.done"
  $PY eval/score_run.py "$label=$out" | sed "s/^/$(date '+%H:%M') /" | tee -a $REPORT
  rm -rf "runs/merged/$(echo ${adapter#/} | tr / _ | sed 's/_/__/g')" runs/merged/*grpo* 2>/dev/null
}

while kill -0 $TRAIN 2>/dev/null; do
  for ck in $(ls -d $RUN/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    [ -f "$ck/adapter_model.safetensors" ] || continue
    evaluate "$ck" "$(basename $ck)"
  done
  sleep 60
done
for ck in $(ls -d $RUN/checkpoint-* $RUN/final 2>/dev/null); do
  [ -f "$ck/adapter_model.safetensors" ] && evaluate "$ck" "$(basename $ck)"
done
echo "$(date '+%F %T') GRPO and evaluation finished" | tee -a $REPORT
