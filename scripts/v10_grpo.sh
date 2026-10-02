#!/usr/bin/env bash
# v10: v9's recipe rebalanced. v9 (40 % reading, 40 % titles, beta 0.04) gained titles 4.05 -> 4.21
# and prose, but lost reading coverage 0.94 -> 0.90. v10 keeps the pairwise title reward with 60 %
# reading, 20 % prose, 20 % titles, and a stronger KL to v8 (beta 0.08). GPU 0 trains, GPU 1 serves the judge (NInfer).
# Then merge and evaluate the three tasks on the held-out sets; v8 numbers are the control.
set -uo pipefail
cd /home/luigi/meeting-summarizer
until grep -q "G350 DONE" reports/small_students.txt 2>/dev/null && ! pgrep -f "^bash scripts/g350_rerun.sh" >/dev/null; do sleep 60; done
kill $(cat logs/judge_dev.pid) 2>/dev/null
R=reports/v10_grpo.txt
PY=~/.venvs/vllm/bin/python
DEPLOY="--harness v5 --no-check --overview none --number-section --read-max-tokens 400 --restart-journal-tokens 2500 --ctx 8192"
gpu_free() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ]; do sleep 5; done; }
echo "start $(date)" | tee -a $R
gpu_free
ART=~/models/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer
CUDA_VISIBLE_DEVICES=1 ~/ninfer/build/apps/ninfer-serve $ART --host 127.0.0.1 --port 8121 --max-context 32768 \
  --kv-capacity auto --max-concurrency 8 --kv-dtype fp8 --spec mtp --draft-tokens 3 --lm-head-draft --model-id judge \
  > logs/ninfer_grpo9.log 2>&1 &
NI=$!
until curl -sf localhost:8121/v1/models >/dev/null; do sleep 5; done
CUDA_VISIBLE_DEVICES=0 $PY distill/grpo_multi.py --adapter runs/sft/agent/grpo-v8/step150/policy --out runs/sft/agent/grpo-v10 \
  --steps 100 --beta 0.08 --task-weights read=0.6,prose=0.2,title=0.2 --pairwise-title --log grpo_v10_log.jsonl \
  > logs/grpo_v10.log 2>&1 || { echo "grpo failed" | tee -a $R; kill -9 $NI; exit 1; }
kill -9 $NI; gpu_free
tail -n 5 logs/grpo_v10.log | tee -a $R
CUDA_VISIBLE_DEVICES=0 $PY distill/merge_agent_lora.py --adapter runs/sft/agent/grpo-v10/step100/policy --out runs/sft/agent/ft-v10-q4_0.gguf \
  > logs/merge_v10.log 2>&1 || { echo "merge failed" | tee -a $R; exit 1; }
M=$PWD/runs/sft/agent/ft-v10-q4_0.gguf
env MODEL=$M TRANSCRIPTS=data/alimeeting/transcripts GOLD=runs/v2/ali-gold SPLIT=data/split_ali20.json PREFIX=ali \
  bash scripts/rt_dev.sh ft-v10 $DEPLOY > logs/eval_ali_v10.log 2>&1
env MODEL=$M PREFIX=h38 SPLIT=data/split_rt_heldout38.json bash scripts/rt_dev.sh ft-v10 $DEPLOY > logs/eval_h38_v10.log 2>&1
kill $(cat logs/dev_student.pid) 2>/dev/null; sleep 10
# conversions and titles of the same journals as v5-v8; the judge (8700) is still up from rt_dev.sh
CUDA_VISIBLE_DEVICES=1 ~/llama.cpp/build-cuda/bin/llama-server -m $M -ngl 99 -c 65536 -np 4 --jinja --port 8140 --alias rt > logs/v10_conv.srv.log 2>&1 &
ST=$!
until curl -sf localhost:8140/health >/dev/null; do sleep 3; done
for n in h38 ali; do
  python3 distill/convert_targets.py --journals runs/student/$n-ft-v5 --urls http://127.0.0.1:8140/v1 --model rt --out runs/convert/$n-v10 > /dev/null 2>&1
  python3 eval/judge_prose_notes.py --dir runs/convert/$n-v10 --judge-url http://127.0.0.1:8700/v1 \
    --out reports/judge_prose_notes_$n-v10.json 2>&1 | tail -1 | tee -a $R
done
python3 eval/title_eval.py --run runs/student/h38-ft-v5 --split data/split_rt_heldout38.json --gold runs/v2/w4000 \
  --titler http://127.0.0.1:8140/v1:rt --judge http://127.0.0.1:8700/v1:judge --tag h38/v10 --out reports/titles_h38_v10.json 2>&1 | tail -1 | tee -a $R
python3 eval/title_eval.py --run runs/student/ali-ft-v5 --split data/split_ali20.json --gold runs/v2/ali-gold \
  --transcripts data/alimeeting/transcripts --titler http://127.0.0.1:8140/v1:rt --judge http://127.0.0.1:8700/v1:judge \
  --tag ali/v10 --out reports/titles_ali_v10.json 2>&1 | tail -1 | tee -a $R
kill $ST; kill $(cat logs/judge_dev.pid) 2>/dev/null
for f in logs/eval_ali_v10.log logs/eval_h38_v10.log; do echo "== $f"; grep -E "^runs/student|^decision|precision" $f; done | tee -a $R
python3 eval/rt_report.py "h38-ft-v*" | tee -a $R
python3 eval/rt_report.py "ali-ft-v*" | tee -a $R
echo "V10 DONE $(date)" | tee -a $R
