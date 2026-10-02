#!/usr/bin/env bash
# GRPO on the best 1B-class student, Qwen3.5-0.8B (SFT: IVOD-38 minutes 31 % contradicted). Its
# best-of-6 sample per window is far more faithful than its average sample (eval/headroom.py), so RL
# has room. Same recipe as Gemma v8 (multi-task, 1-5 title score), from its SFT adapter.
# GPU 0: trainer; GPU 1: Qwen3.8-27B judge on NInfer. Then merge and evaluate on IVOD 38.
set -uo pipefail
cd /home/luigi/meeting-summarizer
until grep -q "SCREEN DONE" reports/screen_1b.txt 2>/dev/null && ! pgrep -f "^bash scripts/screen_1b" >/dev/null; do sleep 60; done
R=reports/grpo_q35.txt
PY=~/.venvs/vllm/bin/python
BASE=Qwen/Qwen3.5-0.8B
gpu_free() { kill $(cat logs/judge_dev.pid) 2>/dev/null; kill $(cat logs/dev_student.pid) 2>/dev/null
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ]; do sleep 5; done; }
echo "start $(date)" | tee -a $R
gpu_free
ART=~/models/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer
CUDA_VISIBLE_DEVICES=1 ~/ninfer/build/apps/ninfer-serve $ART --host 127.0.0.1 --port 8121 --max-context 32768 \
  --kv-capacity auto --max-concurrency 8 --kv-dtype fp8 --spec mtp --draft-tokens 3 --lm-head-draft --model-id judge \
  > logs/ninfer_grpo_q35.log 2>&1 &
NI=$!
until curl -sf localhost:8121/v1/models >/dev/null; do sleep 5; done
CUDA_VISIBLE_DEVICES=0 $PY distill/grpo_multi.py --base $BASE --adapter runs/sft/q35-0.8b/lora/epoch1 --out runs/sft/q35-0.8b/grpo \
  --steps 150 --log grpo_q35_log.jsonl > logs/grpo_q35.log 2>&1 || { echo "grpo failed" | tee -a $R; kill -9 $NI; exit 1; }
kill -9 $NI; gpu_free
tail -n 5 logs/grpo_q35.log | tee -a $R
CUDA_VISIBLE_DEVICES=0 $PY distill/merge_agent_lora.py --base $BASE --adapter runs/sft/q35-0.8b/grpo/step150/policy \
  --out runs/sft/q35-0.8b/grpo-q4_0.gguf > logs/merge_q35_grpo.log 2>&1 || { echo "merge failed" | tee -a $R; exit 1; }
M=$PWD/runs/sft/q35-0.8b/grpo-q4_0.gguf
DEPLOY="--harness v5 --no-check --overview none --number-section --read-max-tokens 400 --restart-journal-tokens 2500 --ctx 8192"
env SERVER_ARGS=--skip-chat-parsing MODEL=$M PREFIX=h38 SPLIT=data/split_rt_heldout38.json bash scripts/rt_dev.sh q35-0.8b-grpo $DEPLOY > logs/eval_h38_q35-0.8b-grpo.log 2>&1
kill $(cat logs/dev_student.pid) 2>/dev/null; sleep 10
CUDA_VISIBLE_DEVICES=1 ~/llama.cpp/build-cuda/bin/llama-server -m $M -ngl 99 -c 65536 -np 4 --jinja --skip-chat-parsing --port 8140 --alias rt > logs/q35_grpo_conv.srv.log 2>&1 &
ST=$!
until curl -sf localhost:8140/health >/dev/null; do sleep 3; done
python3 distill/convert_targets.py --journals runs/student/h38-ft-v5 --urls http://127.0.0.1:8140/v1 --model rt --out runs/convert/h38-q35-grpo > /dev/null 2>&1
python3 eval/judge_prose_notes.py --dir runs/convert/h38-q35-grpo --judge-url http://127.0.0.1:8700/v1 --out reports/judge_prose_notes_h38-q35-grpo.json 2>&1 | tail -1 | tee -a $R
python3 eval/title_eval.py --run runs/student/h38-ft-v5 --split data/split_rt_heldout38.json --gold runs/v2/w4000 \
  --titler http://127.0.0.1:8140/v1:rt --judge http://127.0.0.1:8700/v1:judge --tag h38/q35-grpo --out reports/titles_h38_q35-grpo.json 2>&1 | tail -1 | tee -a $R
kill $ST; kill $(cat logs/judge_dev.pid) 2>/dev/null
grep -E "^runs/student|^decision|precision" logs/eval_h38_q35-0.8b-grpo.log | tee -a $R
python3 eval/rt_report.py "h38-*" | grep -E "^run|q35|ft-v8" | tee -a $R
echo "Q35 GRPO DONE $(date)" | tee -a $R
