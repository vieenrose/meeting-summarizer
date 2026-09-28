#!/usr/bin/env bash
# Second overnight stage: GRPO on the notes step, from the best summary-GRPO checkpoint.
#
# Notes feed every summary, and the untrained base summarises gold notes far better (0.6-0.8) than
# it summarises its own (0.53), so the notes step is where most of the remaining gap sits. Starts
# only once the summary stage has trained and every checkpoint has been scored.
set -uo pipefail
ROOT=/home/luigi/meeting-summarizer; cd $ROOT
PY=~/vllm019/bin/python
TAG=${TAG:-run1}
REPORT=reports/grpo_checkpoints_$TAG.txt
PIDFILE=runs/grpo/.summary-$TAG.pid
summary_running() {
  grep -q "GRPO and evaluation finished" "$REPORT" 2>/dev/null && return 1
  [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}
until [ -f "$PIDFILE" ] || grep -q "GRPO and evaluation finished" "$REPORT" 2>/dev/null; do sleep 10; done
while summary_running; do sleep 60; done
START=$(python3 scripts/pick_best.py $REPORT runs/grpo/g4it-vllm-$TAG 2>>$REPORT)
[ -n "$START" ] || START=runs/sft/g4it-s5/final
echo "$(date '+%F %T') notes GRPO from $START" | tee -a $REPORT
RUN=runs/grpo/g4it-notes-$TAG
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  $PY -m accelerate.commands.launch --num_processes 1 --mixed_precision bf16 \
  distill/grpo_vllm.py --task notes --adapter "$START" --out $RUN --epochs 3 \
  > logs/grpo_notes_vllm_$TAG.log 2>&1 &
TRAIN=$!

evaluate() {
  local adapter=$1 label=notes-$2 out=runs/student/grpo-$TAG-notes-$2
  [ -f "$out/.done" ] && return
  CUDA_VISIBLE_DEVICES=1 timeout 1800 $PY eval/run_student_vllm.py --adapter "$adapter" --out "$out" \
    > logs/eval_grpo_$label.log 2>&1 && touch "$out/.done"
  $PY eval/score_run.py "$label=$out" | sed "s/^/$(date '+%H:%M') /" | tee -a $REPORT
  rm -rf runs/merged/*grpo* 2>/dev/null
}
while kill -0 $TRAIN 2>/dev/null; do
  for ck in $(ls -d $RUN/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    [ -f "$ck/adapter_model.safetensors" ] && evaluate "$ck" "$(basename $ck)"
  done
  sleep 60
done
for ck in $(ls -d $RUN/checkpoint-* $RUN/final 2>/dev/null); do
  [ -f "$ck/adapter_model.safetensors" ] && evaluate "$ck" "$(basename $ck)"
done
echo "$(date '+%F %T') notes GRPO and evaluation finished" | tee -a $REPORT
