#!/usr/bin/env bash
# Q4_0 loss of the fine-tuned student: the same merged v8 model in f16 vs Q4_0, on the same three
# tasks and held-out sets as v8 (reading IVOD 38 + AliMeeting 20, prose fidelity, titles). If the gap
# is small, quantization-aware LoRA training is not worth it.
set -uo pipefail
cd /home/luigi/meeting-summarizer
until grep -qE "V8 DONE|grpo failed|merge failed" reports/v8_grpo.txt 2>/dev/null && ! pgrep -f "^bash scripts/v8_grpo.sh" >/dev/null; do sleep 60; done
R=reports/quant_check.txt
PY=~/.venvs/vllm/bin/python
DEPLOY="--harness v5 --no-check --overview none --number-section --read-max-tokens 400 --restart-journal-tokens 2500 --ctx 8192"
AD=runs/sft/agent/grpo-v8/step150/policy; [ -d $AD ] || AD=runs/sft/agent/lora-v6/epoch0
echo "start $(date) adapter $AD" | tee -a $R
CUDA_VISIBLE_DEVICES=0 $PY distill/merge_agent_lora.py --adapter $AD --out runs/sft/agent/qc-q4_0.gguf --keep-f16 \
  > logs/merge_qc.log 2>&1 || { echo "merge failed" | tee -a $R; exit 1; }
for q in f16 q4_0; do
  M=$PWD/runs/sft/agent/qc-$q.gguf
  env MODEL=$M TRANSCRIPTS=data/alimeeting/transcripts GOLD=runs/v2/ali-gold SPLIT=data/split_ali20.json PREFIX=ali \
    bash scripts/rt_dev.sh qc-$q $DEPLOY > logs/eval_ali_qc-$q.log 2>&1
  env MODEL=$M PREFIX=h38 SPLIT=data/split_rt_heldout38.json bash scripts/rt_dev.sh qc-$q $DEPLOY > logs/eval_h38_qc-$q.log 2>&1
done
kill $(cat logs/dev_student.pid) 2>/dev/null; sleep 20
# conversions (judge on 8700 is still up from rt_dev.sh, TP2 at 0.72: the students fit beside it)
for q in f16 q4_0; do
  CUDA_VISIBLE_DEVICES=1 ~/llama.cpp/build-cuda/bin/llama-server -m $PWD/runs/sft/agent/qc-$q.gguf -ngl 99 -c 65536 -np 4 --jinja \
    --port 8140 --alias rt > logs/qc_$q.srv.log 2>&1 &
  ST=$!
  until curl -sf localhost:8140/health >/dev/null; do sleep 3; done
  for n in h38 ali; do
    python3 distill/convert_targets.py --journals runs/student/$n-ft-v5 --urls http://127.0.0.1:8140/v1 --model rt --out runs/convert/$n-qc-$q > /dev/null 2>&1
    python3 eval/judge_prose_notes.py --dir runs/convert/$n-qc-$q --judge-url http://127.0.0.1:8700/v1 \
      --out reports/judge_prose_notes_$n-qc-$q.json 2>&1 | tail -1 | tee -a $R
  done
  python3 eval/title_eval.py --run runs/student/h38-ft-v5 --split data/split_rt_heldout38.json --gold runs/v2/w4000 \
    --titler http://127.0.0.1:8140/v1:rt --judge http://127.0.0.1:8700/v1:judge --tag h38/qc-$q --out reports/titles_h38_qc-$q.json 2>&1 | tail -1 | tee -a $R
  python3 eval/title_eval.py --run runs/student/ali-ft-v5 --split data/split_ali20.json --gold runs/v2/ali-gold \
    --transcripts data/alimeeting/transcripts --titler http://127.0.0.1:8140/v1:rt --judge http://127.0.0.1:8700/v1:judge \
    --tag ali/qc-$q --out reports/titles_ali_qc-$q.json 2>&1 | tail -1 | tee -a $R
  kill $ST; sleep 5
done
kill $(cat logs/judge_dev.pid) 2>/dev/null
for f in logs/eval_*_qc-*.log; do echo "== $f"; grep -E "^runs/student|^decision|precision" $f; done | tee -a $R
python3 eval/rt_report.py "h38-qc-*" | tee -a $R
python3 eval/rt_report.py "ali-qc-*" | tee -a $R
echo "QC DONE $(date)" | tee -a $R
