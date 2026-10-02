#!/usr/bin/env bash
# v11: contrastive DPO on single-span perturbations of the teacher's notes (binding, attribution,
# outcome; distill/build_contrast_pairs.py), from v8, plus the mechanical attribution guard
# (eval/attribution_guard.py) measured on v8 and Qwen3.5-0.8B. Evaluation on IVOD 38.
set -uo pipefail
cd /home/luigi/meeting-summarizer
until grep -qE "Q35 GRPO DONE|grpo failed|merge failed" reports/grpo_q35.txt 2>/dev/null && ! pgrep -f "^bash scripts/grpo_q35.sh" >/dev/null; do sleep 60; done
R=reports/v11_contrast.txt
PY=~/.venvs/vllm/bin/python
gpu_free() { kill $(cat logs/judge_dev.pid) 2>/dev/null; kill $(cat logs/dev_student.pid) 2>/dev/null
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ]; do sleep 5; done; }
echo "start $(date)" | tee -a $R
gpu_free
CUDA_VISIBLE_DEVICES=0,1 $PY -m torch.distributed.run --nproc_per_node 2 distill/dpo_agent.py --pairs data/train/contrast_pairs.jsonl \
  --sft-adapter runs/sft/agent/grpo-v8/step150/policy --out runs/sft/agent/dpo-v11 --beta 0.1 --alpha 0.2 --lr 5e-6 --epochs 1 \
  > logs/dpo_v11.log 2>&1 || { echo "dpo failed" | tee -a $R; exit 1; }
grep -E "^step" logs/dpo_v11.log | tail -3 | tee -a $R
CUDA_VISIBLE_DEVICES=0 $PY distill/merge_agent_lora.py --adapter runs/sft/agent/dpo-v11 --out runs/sft/agent/ft-v11-q4_0.gguf \
  > logs/merge_v11.log 2>&1 || { echo "merge failed" | tee -a $R; exit 1; }
M=$PWD/runs/sft/agent/ft-v11-q4_0.gguf
DEPLOY="--harness v5 --no-check --overview none --number-section --read-max-tokens 400 --restart-journal-tokens 2500 --ctx 8192"
env MODEL=$M PREFIX=h38 SPLIT=data/split_rt_heldout38.json bash scripts/rt_dev.sh ft-v11 $DEPLOY > logs/eval_h38_v11.log 2>&1
kill $(cat logs/dev_student.pid) 2>/dev/null; sleep 10
# the attribution guard on v8 and Qwen3.5-0.8B (judge still up)
J="--judge-url http://127.0.0.1:8700/v1 --judge-model judge --split data/split_rt_heldout38.json"
for run in h38-ft-v8-ag h38-q35-0.8b-ft-ag; do
  python3 eval/judge_prose_tx.py --candidate runs/student/$run $J --transcripts data/v2/transcripts --out reports/judge_prose_tx_$run.json 2>&1 | tail -1 | tee -a $R
  python3 eval/judge_prose.py --candidate runs/student/$run $J --gold runs/v2/w4000 --out reports/judge_prose_$run.json 2>&1 | tail -1 | tee -a $R
done
CUDA_VISIBLE_DEVICES=1 ~/llama.cpp/build-cuda/bin/llama-server -m $M -ngl 99 -c 65536 -np 4 --jinja --port 8140 --alias rt > logs/v11_conv.srv.log 2>&1 &
ST=$!
until curl -sf localhost:8140/health >/dev/null; do sleep 3; done
python3 distill/convert_targets.py --journals runs/student/h38-ft-v5 --urls http://127.0.0.1:8140/v1 --model rt --out runs/convert/h38-v11 > /dev/null 2>&1
python3 eval/judge_prose_notes.py --dir runs/convert/h38-v11 --judge-url http://127.0.0.1:8700/v1 --out reports/judge_prose_notes_h38-v11.json 2>&1 | tail -1 | tee -a $R
python3 eval/title_eval.py --run runs/student/h38-ft-v5 --split data/split_rt_heldout38.json --gold runs/v2/w4000 \
  --titler http://127.0.0.1:8140/v1:rt --judge http://127.0.0.1:8700/v1:judge --tag h38/v11 --out reports/titles_h38_v11.json 2>&1 | tail -1 | tee -a $R
kill $ST
python3 eval/contradiction_types.py --runs h38-ft-v11,h38-ft-v8-ag --per-run 250 --out reports/contradiction_types_v11.json 2>&1 | grep -v Warn | tee -a $R
kill $(cat logs/judge_dev.pid) 2>/dev/null
grep -E "^runs/student|^decision|precision" logs/eval_h38_v11.log | tee -a $R
python3 eval/rt_report.py "h38-ft-v*" | tee -a $R
echo "V11 DONE $(date)" | tee -a $R
