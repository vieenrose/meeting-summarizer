#!/usr/bin/env bash
# Iteration v6, same student: distill the two conversion calls VoxSumDroid makes at the end of a
# meeting (title, prose summary from the journal; eval/conversion_prompts.py), in the same LoRA as
# the v5 reading turns. Both GPUs throughout.
#  1. teacher (27B, NInfer x2) converts the teacher's v5 journals and the student's own journals
#  2. judge every prose sentence against the notes; keep faithful, well-formed targets
#  3. SFT on v5 agent rows + conversion rows (2 GPUs), merge -> Q4_0
#  4. evaluate conversion on held-out journals (v5 student, IVOD 38 + AliMeeting 20): v5 zero-shot,
#     v6, and the 27B; titles scored; and check v6 still reads as well as v5
set -uo pipefail
cd /home/luigi/meeting-summarizer
until grep -q "TITLE DONE" reports/title_eval.txt 2>/dev/null; do sleep 60; done
R=reports/v6_iteration.txt
PY=~/.venvs/vllm/bin/python
DEPLOY5="--harness v5 --no-check --overview none --number-section --read-max-tokens 400 --restart-journal-tokens 2500 --ctx 8192"
ART=~/models/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer
echo "start $(date)" | tee -a $R
ninfer_up() {
  NI=()
  for g in 0 1; do
    CUDA_VISIBLE_DEVICES=$g ~/ninfer/build/apps/ninfer-serve $ART --host 127.0.0.1 --port 812$g --max-context 32768 \
      --kv-capacity auto --max-concurrency 4 --kv-dtype fp8 --spec mtp --draft-tokens 3 --lm-head-draft --model-id q38 \
      > logs/ninfer_v6_$g.log 2>&1 &
    NI+=($!)
  done
  for g in 0 1; do until curl -sf localhost:812$g/v1/models >/dev/null; do sleep 5; done; done
}
judge_up() {
  ~/.venvs/cu13/bin/python -m vllm.entrypoints.openai.api_server --model bahadirakdemir/gemma-4-31B-it-text-fp8 \
    --served-model-name judge --tensor-parallel-size 2 --max-model-len 24000 --gpu-memory-utilization ${1:-0.85} \
    --max-num-seqs 16 --port 8700 > logs/judge_v6.log 2>&1 &
  JU=$!
  until curl -sf localhost:8700/v1/models >/dev/null; do sleep 15; done
}
# 1. teacher conversions (train journals), and the 27B's conversions of the held-out student journals
ninfer_up
python3 distill/convert_targets.py --journals runs/teacher/rt-q38-v5 runs/student/onpol-ft-ep0 \
  --urls http://127.0.0.1:8120/v1,http://127.0.0.1:8121/v1 --out runs/convert/teacher-train 2>&1 | tail -1 | tee -a $R
for c in "h38|runs/student/h38-ft-v5" "ali|runs/student/ali-ft-v5"; do
  IFS='|' read -r n j <<< "$c"
  python3 distill/convert_targets.py --journals $j --urls http://127.0.0.1:8120/v1,http://127.0.0.1:8121/v1 \
    --out runs/convert/$n-q38 2>&1 | tail -1
done
kill "${NI[@]}"; sleep 30
# 2. judge the teacher targets against their notes; build rows
judge_up 0.85
python3 eval/judge_prose_notes.py --dir runs/convert/teacher-train --judge-url http://127.0.0.1:8700/v1 \
  --out reports/judge_prose_notes_teacher-train.json 2>&1 | tail -1 | tee -a $R
kill $JU; sleep 20
$PY distill/build_convert_sft.py --targets runs/convert/teacher-train --judged reports/judge_prose_notes_teacher-train.json \
  --out data/train/convert_rows.jsonl 2>&1 | tail -1 | tee -a $R
cat data/train/agent_sft_rows_v5.jsonl data/train/convert_rows.jsonl > data/train/agent_sft_rows_v6.jsonl
# 3. SFT on both GPUs, merge
CUDA_VISIBLE_DEVICES=0,1 $PY -m torch.distributed.run --nproc_per_node 2 distill/sft_agent.py \
  --rows data/train/agent_sft_rows_v6.jsonl --epochs 1 --out runs/sft/agent/lora-v6 > logs/sft_v6.log 2>&1 \
  || { echo "sft failed" | tee -a $R; exit 1; }
grep -E "val loss|train segments" logs/sft_v6.log | tee -a $R
CUDA_VISIBLE_DEVICES=0 $PY distill/merge_agent_lora.py --adapter runs/sft/agent/lora-v6/epoch0 --out runs/sft/agent/ft-v6-q4_0.gguf \
  > logs/merge_v6.log 2>&1 || { echo "merge failed" | tee -a $R; exit 1; }
# 4. conversions by the students (v5 zero-shot, v6) on the held-out journals, one server per GPU
for spec in "v5|$PWD/runs/sft/agent/ft-v5-q4_0.gguf|0|8150" "v6|$PWD/runs/sft/agent/ft-v6-q4_0.gguf|1|8151"; do
  IFS='|' read -r tag m g port <<< "$spec"
  CUDA_VISIBLE_DEVICES=$g ~/llama.cpp/build-cuda/bin/llama-server -m $m -ngl 99 -c 32768 -np 4 --jinja --port $port --alias rt \
    > logs/convert_$tag.srv.log 2>&1 &
  echo $! > logs/convert_$tag.pid
done
for port in 8150 8151; do until curl -sf localhost:$port/health >/dev/null; do sleep 3; done; done
for n in h38 ali; do
  j=runs/student/$n-ft-v5
  python3 distill/convert_targets.py --journals $j --urls http://127.0.0.1:8150/v1 --model rt --out runs/convert/$n-v5 > /dev/null 2>&1 &
  python3 distill/convert_targets.py --journals $j --urls http://127.0.0.1:8151/v1 --model rt --out runs/convert/$n-v6 > /dev/null 2>&1 &
  wait
done
kill $(cat logs/convert_v5.pid) $(cat logs/convert_v6.pid); sleep 10
judge_up 0.85
for n in h38 ali; do for w in v5 v6 q38; do
  python3 eval/judge_prose_notes.py --dir runs/convert/$n-$w --judge-url http://127.0.0.1:8700/v1 \
    --out reports/judge_prose_notes_$n-$w.json 2>&1 | tail -1 | tee -a $R
done; done
kill $JU; sleep 20
# titles by v6 (v5's are in reports/title_eval.txt), and v6 still reading as well as v5
judge_up 0.72
CUDA_VISIBLE_DEVICES=1 ~/llama.cpp/build-cuda/bin/llama-server -m $PWD/runs/sft/agent/ft-v6-q4_0.gguf -ngl 99 -c 16384 -np 4 \
  --jinja --port 8140 --alias rt > logs/title_v6.srv.log 2>&1 &
ST=$!
until curl -sf localhost:8140/health >/dev/null; do sleep 3; done
python3 eval/title_eval.py --run runs/student/h38-ft-v5 --split data/split_rt_heldout38.json --gold runs/v2/w4000 \
  --titler http://127.0.0.1:8140/v1:rt --judge http://127.0.0.1:8700/v1:judge --tag h38/v6 --out reports/titles_h38_v6.json 2>&1 | tail -1 | tee -a $R
python3 eval/title_eval.py --run runs/student/ali-ft-v5 --split data/split_ali20.json --gold runs/v2/ali-gold \
  --transcripts data/alimeeting/transcripts --titler http://127.0.0.1:8140/v1:rt --judge http://127.0.0.1:8700/v1:judge \
  --tag ali/v6 --out reports/titles_ali_v6.json 2>&1 | tail -1 | tee -a $R
kill $ST $JU; sleep 20
env MODEL=$PWD/runs/sft/agent/ft-v6-q4_0.gguf TRANSCRIPTS=data/alimeeting/transcripts GOLD=runs/v2/ali-gold \
  SPLIT=data/split_ali20.json PREFIX=ali bash scripts/rt_dev.sh ft-v6 $DEPLOY5 > logs/eval_ali_v6.log 2>&1
env MODEL=$PWD/runs/sft/agent/ft-v6-q4_0.gguf PREFIX=h38 SPLIT=data/split_rt_heldout38.json \
  bash scripts/rt_dev.sh ft-v6 $DEPLOY5 > logs/eval_h38_v6.log 2>&1
for f in logs/eval_ali_v6.log logs/eval_h38_v6.log; do echo "== $f"; grep -E "^runs/student|^decision|precision" $f; done | tee -a $R
kill $(cat logs/judge_dev.pid) $(cat logs/dev_student.pid) 2>/dev/null
echo "V6 DONE $(date)" | tee -a $R
