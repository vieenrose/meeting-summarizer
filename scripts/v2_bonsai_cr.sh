#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
until grep -q "BONSAI DONE" reports/v2_bonsai.txt 2>/dev/null; do sleep 120; done
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 15
B=~/Bonsai-demo/bin/cuda
for g in 0 1; do CUDA_VISIBLE_DEVICES=$g LD_LIBRARY_PATH=$B nohup $B/llama-server -m /home/luigi/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf -ngl 99 -c 131072 -np 1 --jinja   --port 809$g --alias bonsai > logs/bonsai_cr$g.log 2>&1 & echo $! > logs/bonsai_cr$g.pid; done
for g in 0 1; do until curl -sf localhost:809$g/health >/dev/null; do sleep 10; done; done
python3 eval/structured_cr.py --urls http://127.0.0.1:8090/v1,http://127.0.0.1:8091/v1 --out runs/student/v2-bonsai-cr > logs/bonsai_cr.log 2>&1
for g in 0 1; do kill $(cat logs/bonsai_cr$g.pid); done; sleep 20
JUDGE_SCRIPT=judge_prose_tx DIRS="runs/student/v2-bonsai-cr runs/student/v2-bonsai-cr-draft" bash scripts/v2_judge.sh >> logs/v2_judge_bcr.log 2>&1
grep sentences reports/v2_sft_night.txt | tail -2 | tee -a reports/v2_bonsai_cr.txt
echo BONSAI_CR DONE | tee -a reports/v2_bonsai_cr.txt
