#!/usr/bin/env bash
# v8: multi-task GRPO from the v6 adapter (v7 lost coverage) (reading turns, notes->prose, notes->title).
# GPU 0: trainer (sampling + update). GPU 1: Qwen3.8-27B judge on NInfer (rewards).
# Then merge, and evaluate all three tasks on the held-out sets against v5/v6.
set -uo pipefail
cd /home/luigi/meeting-summarizer
until grep -q "V7 DONE" reports/v7_iteration.txt 2>/dev/null && grep -q "h38/v5" reports/title_eval.txt 2>/dev/null && ! pgrep -f "^bash scripts/title_v5_h38.sh" >/dev/null; do sleep 60; done
R=reports/v8_grpo.txt
PY=~/.venvs/vllm/bin/python
DEPLOY7="--harness v5 --no-check --overview none --number-section --read-max-tokens 400 --restart-journal-tokens 2500 --ctx 8192"
echo "start $(date)" | tee -a $R
ART=~/models/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer
CUDA_VISIBLE_DEVICES=1 ~/ninfer/build/apps/ninfer-serve $ART --host 127.0.0.1 --port 8121 --max-context 32768 \
  --kv-capacity auto --max-concurrency 8 --kv-dtype fp8 --spec mtp --draft-tokens 3 --lm-head-draft --model-id judge \
  > logs/ninfer_grpo.log 2>&1 &
NI=$!
until curl -sf localhost:8121/v1/models >/dev/null; do sleep 5; done
CUDA_VISIBLE_DEVICES=0 $PY distill/grpo_multi.py --steps 150 > logs/grpo_v8.log 2>&1 || { echo "grpo failed" | tee -a $R; kill $NI; exit 1; }
kill $NI; sleep 20
tail -n 5 logs/grpo_v8.log | tee -a $R
CUDA_VISIBLE_DEVICES=0 $PY distill/merge_agent_lora.py --adapter runs/sft/agent/grpo-v8/step150/policy --out runs/sft/agent/ft-v8-q4_0.gguf \
  > logs/merge_v8.log 2>&1 || { echo "merge failed" | tee -a $R; exit 1; }
M=$PWD/runs/sft/agent/ft-v8-q4_0.gguf
# task 1: reading
env MODEL=$M TRANSCRIPTS=data/alimeeting/transcripts GOLD=runs/v2/ali-gold SPLIT=data/split_ali20.json PREFIX=ali \
  bash scripts/rt_dev.sh ft-v8 $DEPLOY7 > logs/eval_ali_v8.log 2>&1
env MODEL=$M PREFIX=h38 SPLIT=data/split_rt_heldout38.json bash scripts/rt_dev.sh ft-v8 $DEPLOY7 > logs/eval_h38_v8.log 2>&1
kill $(cat logs/judge_dev.pid) $(cat logs/dev_student.pid) 2>/dev/null; sleep 20
# tasks 2-3: conversions of the same held-out journals as v5/v6, by v6 (GPU 0, again, as a control) and v8 (GPU 1)
for spec in "v6|$PWD/runs/sft/agent/ft-v6-q4_0.gguf|0|8150" "v8|$M|1|8151"; do
  IFS='|' read -r tag m g port <<< "$spec"
  CUDA_VISIBLE_DEVICES=$g ~/llama.cpp/build-cuda/bin/llama-server -m $m -ngl 99 -c 65536 -np 4 --jinja --port $port --alias rt \
    > logs/convert_$tag.srv.log 2>&1 &
  echo $! > logs/convert_$tag.pid
done
for port in 8150 8151; do until curl -sf localhost:$port/health >/dev/null; do sleep 3; done; done
for n in h38 ali; do
  python3 distill/convert_targets.py --journals runs/student/$n-ft-v5 --urls http://127.0.0.1:8150/v1 --model rt --out runs/convert/$n-v6b > /dev/null 2>&1 & p1=$!
  python3 distill/convert_targets.py --journals runs/student/$n-ft-v5 --urls http://127.0.0.1:8151/v1 --model rt --out runs/convert/$n-v8 > /dev/null 2>&1 & p2=$!
  wait $p1 $p2
done
kill $(cat logs/convert_v6.pid) $(cat logs/convert_v8.pid); sleep 10
~/.venvs/cu13/bin/python -m vllm.entrypoints.openai.api_server --model bahadirakdemir/gemma-4-31B-it-text-fp8 \
  --served-model-name judge --tensor-parallel-size 2 --max-model-len 24000 --gpu-memory-utilization 0.72 \
  --max-num-seqs 16 --port 8700 > logs/judge_v8.log 2>&1 &
JU=$!
until curl -sf localhost:8700/v1/models >/dev/null; do sleep 15; done
for n in h38 ali; do for w in v6b v8; do
  python3 eval/judge_prose_notes.py --dir runs/convert/$n-$w --judge-url http://127.0.0.1:8700/v1 \
    --out reports/judge_prose_notes_$n-$w.json 2>&1 | tail -1 | tee -a $R
done; done
for spec in "v6|$PWD/runs/sft/agent/ft-v6-q4_0.gguf" "v8|$M"; do
  IFS='|' read -r tag m <<< "$spec"
  CUDA_VISIBLE_DEVICES=1 ~/llama.cpp/build-cuda/bin/llama-server -m $m -ngl 99 -c 65536 -np 4 --jinja --port 8140 --alias rt > logs/title_$tag.srv.log 2>&1 &
  ST=$!
  until curl -sf localhost:8140/health >/dev/null; do sleep 3; done
  python3 eval/title_eval.py --run runs/student/h38-ft-v5 --split data/split_rt_heldout38.json --gold runs/v2/w4000 \
    --titler http://127.0.0.1:8140/v1:rt --judge http://127.0.0.1:8700/v1:judge --tag h38/$tag --out reports/titles_h38_$tag.json 2>&1 | tail -1 | tee -a $R
  python3 eval/title_eval.py --run runs/student/ali-ft-v5 --split data/split_ali20.json --gold runs/v2/ali-gold \
    --transcripts data/alimeeting/transcripts --titler http://127.0.0.1:8140/v1:rt --judge http://127.0.0.1:8700/v1:judge \
    --tag ali/$tag --out reports/titles_ali_$tag.json 2>&1 | tail -1 | tee -a $R
  kill $ST; sleep 5
done
kill $JU
for f in logs/eval_ali_v8.log logs/eval_h38_v8.log; do echo "== $f"; grep -E "^runs/student|^decision|precision" $f; done | tee -a $R
python3 eval/rt_report.py "h38-ft-v*" | tee -a $R
python3 eval/rt_report.py "ali-ft-v*" | tee -a $R
echo "V8 DONE $(date)" | tee -a $R
