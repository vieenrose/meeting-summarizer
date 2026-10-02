#!/usr/bin/env bash
# v12 candidates without retraining: weighted sums of the v8 adapter with v9 (better titles and
# prose, worse reading coverage) and with v11 (more precise decisions, worse coverage). If a soup
# keeps v8's reading and gains the other's strength, it is the next release. Evaluation: IVOD 38.
set -uo pipefail
cd /home/luigi/meeting-summarizer
until grep -q "Q35 LP DONE" reports/q35_lp.txt 2>/dev/null && ! pgrep -f "^bash scripts/q35_lp.sh" >/dev/null; do sleep 60; done
R=reports/v12_soup.txt
PY=~/.venvs/vllm/bin/python
V8=runs/sft/agent/grpo-v8/step150/policy
DEPLOY="--harness v5 --no-check --overview none --number-section --read-max-tokens 400 --restart-journal-tokens 2500 --ctx 8192"
echo "start $(date)" | tee -a $R
for spec in "v12a;runs/sft/agent/grpo-v9/step100/policy;0.5,0.5" "v12b;runs/sft/agent/dpo-v11/policy;0.5,0.5"; do
  IFS=';' read -r tag other w <<< "$spec"
  CUDA_VISIBLE_DEVICES= $PY distill/soup_adapters.py --adapters $V8,$other --weights $w --out runs/sft/agent/soup-$tag > logs/soup_$tag.log 2>&1 \
    || { echo "$tag soup failed" | tee -a $R; continue; }
  CUDA_VISIBLE_DEVICES= $PY distill/merge_agent_lora.py --adapter runs/sft/agent/soup-$tag/soup --out runs/sft/agent/ft-$tag-q4_0.gguf \
    > logs/merge_$tag.log 2>&1 || { echo "$tag merge failed" | tee -a $R; continue; }
  M=$PWD/runs/sft/agent/ft-$tag-q4_0.gguf
  env MODEL=$M PREFIX=h38 SPLIT=data/split_rt_heldout38.json bash scripts/rt_dev.sh ft-$tag $DEPLOY > logs/eval_h38_$tag.log 2>&1
  kill $(cat logs/dev_student.pid) 2>/dev/null; sleep 10
  CUDA_VISIBLE_DEVICES=1 ~/llama.cpp/build-cuda/bin/llama-server -m $M -ngl 99 -c 65536 -np 4 --jinja --port 8140 --alias rt > logs/${tag}_conv.srv.log 2>&1 &
  ST=$!
  until curl -sf localhost:8140/health >/dev/null; do sleep 3; done
  python3 distill/convert_targets.py --journals runs/student/h38-ft-v5 --urls http://127.0.0.1:8140/v1 --model rt --out runs/convert/h38-$tag > /dev/null 2>&1
  python3 eval/judge_prose_notes.py --dir runs/convert/h38-$tag --judge-url http://127.0.0.1:8700/v1 --out reports/judge_prose_notes_h38-$tag.json 2>&1 | tail -1 | tee -a $R
  python3 eval/title_eval.py --run runs/student/h38-ft-v5 --split data/split_rt_heldout38.json --gold runs/v2/w4000 \
    --titler http://127.0.0.1:8140/v1:rt --judge http://127.0.0.1:8700/v1:judge --tag h38/$tag --out reports/titles_h38_$tag.json 2>&1 | tail -1 | tee -a $R
  kill $ST; sleep 5
  echo "== $tag" | tee -a $R
  grep -E "^runs/student|^decision|precision" logs/eval_h38_$tag.log | tee -a $R
done
kill $(cat logs/judge_dev.pid) 2>/dev/null
python3 eval/rt_report.py "h38-ft-v*" | tee -a $R
echo "V12 DONE $(date)" | tee -a $R
