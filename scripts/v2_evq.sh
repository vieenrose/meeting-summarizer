#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python; R=reports/v2_evq.txt
until grep -q CAD_GEN_DONE reports/v2_cad.txt 2>/dev/null; do sleep 60; done
source ~/.venvs/vllm/bin/activate
CUDA_VISIBLE_DEVICES=1 python3 distill/sft_gemma.py --base google/gemma-4-E4B-it --rows data/train/reduce_evq_mix.jsonl \
  --epochs 2 --out runs/sft/e4b-reduce-evq > logs/sft_e4b_evq.log 2>&1
deactivate
AD=$(ls -d runs/sft/e4b-reduce-evq/final 2>/dev/null || ls -d runs/sft/e4b-reduce-evq/checkpoint-* | tail -1)
CUDA_VISIBLE_DEVICES=1 timeout 9000 $PY eval/run_student_vllm.py --base openbmb/MiniCPM5-2B --adapter runs/sft/minicpm5-evidence/final \
  --evidence --keep-evidence --out runs/student/v2-mev-keepq --max-model-len 16384 > logs/eval_mev_keepq.log 2>&1
for p in $(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 10
CUDA_VISIBLE_DEVICES=1 $PY eval/prose_from_gold.py --base google/gemma-4-E4B-it --adapter $AD \
  --notes-dir runs/student/v2-mev-keepq --out runs/student/v2-e4b-evq > logs/pfg_evq.log 2>&1
for p in $(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; rm -rf runs/merged/runs__sft__e4b-reduce-evq* runs/merged/runs__sft__minicpm5-evidence__*
python3 eval/prose_guard.py --run-dir runs/student/v2-e4b-evq --out runs/student/v2-e4b-evq-guard | tee -a $R
echo EVQ_GEN_DONE | tee -a $R
