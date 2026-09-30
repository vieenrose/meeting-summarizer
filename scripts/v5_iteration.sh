#!/usr/bin/env bash
# Iteration v5, same student (Gemma-4-E2B QAT): PROPOSAL type + strict DECISION/ACTION, and
# teacher traces on IVOD train + AliMeeting (minus the 20 test meetings).
#  1. teacher traces (Qwen3.8-27B, NInfer, one per GPU) with --harness v5
#  2. judge every teacher note against its transcript (IVOD and AliMeeting separately)
#  3. SFT rows, LoRA SFT on 2 GPUs, merge -> Q4_0
#  4. evaluate: IVOD 38 held-out and AliMeeting 20, against the v3 model; section precision
set -uo pipefail
cd /home/luigi/meeting-summarizer
R=reports/v5_iteration.txt
PY=~/.venvs/vllm/bin/python
OUT=runs/teacher/rt-q38-v5
DEPLOY5="--harness v5 --no-check --overview none --number-section --read-max-tokens 400 --restart-journal-tokens 2500 --ctx 8192"
DEPLOY3="--harness v3 --no-check --overview none --number-section --read-max-tokens 400 --restart-journal-tokens 2500 --ctx 8192"
ART=~/models/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer
echo "start $(date)" | tee -a $R
srv=()
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g ~/ninfer/build/apps/ninfer-serve $ART --host 127.0.0.1 --port 812$g --max-context 32768 \
    --kv-capacity auto --max-concurrency 4 --kv-dtype fp8 --spec mtp --draft-tokens 3 --lm-head-draft --model-id q38 \
    > logs/ninfer_v5_$g.log 2>&1 &
  srv+=($!)
done
for g in 0 1; do until curl -sf localhost:812$g/v1/models >/dev/null; do sleep 5; done; done
mapfile -t S < <(python3 -c "import json; print('\n'.join(json.load(open('data/split_rt_train_v5.json'))['train']))")
half=$(( ${#S[@]} / 2 )); ag=()
for g in 0 1; do
  if [ $g = 0 ]; then part=("${S[@]:0:$half}"); else part=("${S[@]:$half}"); fi
  python3 eval/realtime_agent.py --url http://127.0.0.1:812$g/v1 --model q38 --parallel 4 \
    --transcripts data/v2/transcripts,data/alimeeting/transcripts $DEPLOY5 --only "${part[@]}" --out $OUT \
    > logs/teacher_v5_$g.log 2>&1 &
  ag+=($!)
done
wait "${ag[@]}"
kill "${srv[@]}"; sleep 30
echo "teacher traces $(ls $OUT | wc -l)" | tee -a $R
python3 - <<'P'
import glob, json, os
for src, dst in [('runs/teacher/rt-q38-v5', 'runs/teacher/notes-rt-q38-v5')]:
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(f'{src}/*.json'):
        r = json.load(open(f))
        json.dump({'notes': r['notes'], 'prose': ''.join(n['text'].rstrip('。') + f" [{n['ts']}]。" for n in r['notes'])},
                  open(f'{dst}/{os.path.basename(f)}', 'w'), ensure_ascii=False)
P
~/.venvs/cu13/bin/python -m vllm.entrypoints.openai.api_server --model bahadirakdemir/gemma-4-31B-it-text-fp8 \
  --served-model-name judge --tensor-parallel-size 2 --max-model-len 24000 --gpu-memory-utilization 0.85 \
  --max-num-seqs 8 --port 8700 > logs/judge_v5_teacher.log 2>&1 &
JU=$!
until curl -sf localhost:8700/v1/models >/dev/null; do sleep 15; done
for c in "ivod|data/v2/transcripts" "ali|data/alimeeting/transcripts"; do
  IFS='|' read -r n t <<< "$c"
  python3 eval/judge_prose_tx.py --candidate runs/teacher/notes-rt-q38-v5 --split logs/split_v5_${n}_as_heldout.json \
    --transcripts $t --judge-url http://127.0.0.1:8700/v1 --judge-model judge --parallel 16 \
    --out reports/judge_prose_tx_notes-rt-q38-v5-$n.json 2>&1 | tail -1 | tee -a $R
done
kill $JU; sleep 20
$PY distill/build_agent_sft.py --teacher $OUT --system v5 \
  --judged reports/judge_prose_tx_notes-rt-q38-v5-ivod.json,reports/judge_prose_tx_notes-rt-q38-v5-ali.json \
  --out data/train/agent_sft_rows_v5.jsonl 2>&1 | grep -v -i warn | tail -1 | tee -a $R
CUDA_VISIBLE_DEVICES=0,1 $PY -m torch.distributed.run --nproc_per_node 2 distill/sft_agent.py \
  --rows data/train/agent_sft_rows_v5.jsonl --epochs 1 --out runs/sft/agent/lora-v5 > logs/sft_v5.log 2>&1 \
  || { echo "sft failed" | tee -a $R; exit 1; }
grep -E "val loss|train segments" logs/sft_v5.log | tee -a $R
CUDA_VISIBLE_DEVICES=0 $PY distill/merge_agent_lora.py --adapter runs/sft/agent/lora-v5/epoch0 --out runs/sft/agent/ft-v5-q4_0.gguf \
  > logs/merge_v5.log 2>&1 || { echo "merge failed" | tee -a $R; exit 1; }
V5=$PWD/runs/sft/agent/ft-v5-q4_0.gguf; V3=$PWD/runs/sft/agent/ft-ep0f16-q4_0.gguf
ALI="TRANSCRIPTS=data/alimeeting/transcripts GOLD=runs/v2/ali-gold SPLIT=data/split_ali20.json PREFIX=ali"
env MODEL=$V5 TRANSCRIPTS=data/alimeeting/transcripts GOLD=runs/v2/ali-gold SPLIT=data/split_ali20.json PREFIX=ali \
  bash scripts/rt_dev.sh ft-v5 $DEPLOY5 > logs/eval_ali_v5.log 2>&1
env MODEL=$V3 TRANSCRIPTS=data/alimeeting/transcripts GOLD=runs/v2/ali-gold SPLIT=data/split_ali20.json PREFIX=ali \
  bash scripts/rt_dev.sh ft-v3-v5prompt $DEPLOY5 > logs/eval_ali_v3v5prompt.log 2>&1
env MODEL=$V5 PREFIX=h38 SPLIT=data/split_rt_heldout38.json bash scripts/rt_dev.sh ft-v5 $DEPLOY5 > logs/eval_h38_v5.log 2>&1
# section precision of the v3 baselines (their generation and other judgments already exist)
python3 eval/section_precision.py --candidate runs/student/ali-gemma4-ft --split data/split_ali20.json \
  --transcripts data/alimeeting/transcripts --judge-url http://127.0.0.1:8700/v1 --out reports/section_precision_ali-gemma4-ft.json 2>&1 | tail -2 | tee -a $R
python3 eval/section_precision.py --candidate runs/student/h38-gemma4-e2b-ft-ep0-ctx8k --split data/split_rt_heldout38.json \
  --judge-url http://127.0.0.1:8700/v1 --out reports/section_precision_h38-v3.json 2>&1 | tail -2 | tee -a $R
for f in logs/eval_ali_v5.log logs/eval_ali_v3v5prompt.log logs/eval_h38_v5.log; do echo "== $f"; grep -E "^runs/student|^decision|precision" $f; done | tee -a $R
python3 eval/rt_report.py "ali-*" | tee -a $R
python3 eval/rt_report.py "h38-*" | tee -a $R
kill $(cat logs/judge_dev.pid) $(cat logs/dev_student.pid) 2>/dev/null
echo "V5 DONE $(date)" | tee -a $R
