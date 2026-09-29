#!/usr/bin/env bash
# Teacher traces for distilling Gemma-4-E2B: Qwen3.8-27B (NInfer, one instance per GPU) runs the
# tuned realtime protocol (v3x) on the training sessions (data/split_rt_train.json), then the
# judge (Gemma-4-31B) checks every teacher note against the transcript, so that turns with a
# contradicted note can be filtered out of the training set.
set -uo pipefail
cd /home/luigi/meeting-summarizer
OUT=runs/teacher/rt-q38-v3x
ART=~/models/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer
mkdir -p $OUT
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g ~/ninfer/build/apps/ninfer-serve $ART --host 127.0.0.1 --port 812$g --max-context 32768 \
    --kv-capacity auto --max-concurrency 4 --kv-dtype fp8 --device-state-slots 4 --host-state-slots 16 \
    --host-kv-mib 16384 --spec mtp --draft-tokens 3 --lm-head-draft --model-id q38 > logs/ninfer_teacher_$g.log 2>&1 &
  echo $! > logs/ninfer_teacher_$g.pid
done
for g in 0 1; do until curl -sf localhost:812$g/v1/models >/dev/null; do sleep 5; done; done
mapfile -t S < <(python3 -c "import json; print('\n'.join(json.load(open('data/split_rt_train.json'))['train']))")
half=$(( ${#S[@]} / 2 ))
agents=()
for g in 0 1; do
  if [ $g = 0 ]; then part=("${S[@]:0:$half}"); else part=("${S[@]:$half}"); fi
  python3 eval/realtime_agent.py --url http://127.0.0.1:812$g/v1 --model q38 --parallel 4 \
    --harness v3 --no-check --overview none --number-section --only "${part[@]}" --out $OUT \
    > logs/teacher_traces_$g.log 2>&1 &
  agents+=($!)
done
wait "${agents[@]}"                                  # not a bare wait: the servers never exit
for g in 0 1; do kill $(cat logs/ninfer_teacher_$g.pid) 2>/dev/null; done; sleep 20
echo "TRACES $(ls $OUT | wc -l)" | tee -a reports/rt_teacher.txt
# Judge every teacher note (not a sample) against the transcript.
python3 - <<'P'
import glob, json, os
src = 'runs/teacher/rt-q38-v3x'; dst = 'runs/teacher/notes-rt-q38-v3x'
os.makedirs(dst, exist_ok=True)
for f in glob.glob(f'{src}/ivod_*.json'):
    r = json.load(open(f))
    prose = ''.join(n['text'].rstrip('。') + f" [{n['ts']}]。" for n in r['notes'])
    json.dump({'notes': r['notes'], 'prose': prose}, open(f'{dst}/{os.path.basename(f)}', 'w'), ensure_ascii=False)
json.dump({'heldout': json.load(open('data/split_rt_train.json'))['train']}, open('logs/split_rt_train_as_heldout.json', 'w'))
P
JUDGE_SCRIPT=judge_prose_tx DIRS="runs/teacher/notes-rt-q38-v3x" JUDGE_EXTRA="--split logs/split_rt_train_as_heldout.json --parallel 16" \
  bash scripts/v2_judge.sh >> logs/rt_teacher_judge.log 2>&1
grep "notes-rt-q38-v3x" reports/v2_sft_night.txt | tail -1 | tee -a reports/rt_teacher.txt
echo "TEACHER DONE" | tee -a reports/rt_teacher.txt
