#!/usr/bin/env bash
# Evaluations of the 1B-class screen (scripts/screen_1b.sh), each as soon as its training is merged,
# once the Gemma v10 evaluation is over (the judge needs both cards). IVOD 38: reading, prose, titles.
set -uo pipefail
cd /home/luigi/meeting-summarizer
R=reports/screen_1b.txt
DEPLOY="--harness v5 --no-check --overview none --number-section --read-max-tokens 400 --restart-journal-tokens 2500 --ctx 8192"
until grep -q "V10 DONE" reports/v10_grpo.txt 2>/dev/null && ! pgrep -f "^bash scripts/v10_grpo.sh" >/dev/null; do sleep 60; done
done_n=0
while :; do
  n=$(wc -l < logs/screen_1b.queue 2>/dev/null || echo 0)
  if [ "$done_n" -ge "$n" ]; then
    grep -q TRAINED reports/screen_1b.txt 2>/dev/null && [ "$done_n" -ge "$(wc -l < logs/screen_1b.queue 2>/dev/null || echo 0)" ] && break
    sleep 60; continue
  fi
  done_n=$((done_n + 1))
  tag=$(sed -n "${done_n}p" logs/screen_1b.queue)
  out=runs/sft/$tag
  echo "== $tag eval" | tee -a $R
  env SERVER_ARGS=--skip-chat-parsing MODEL=$PWD/$out/ft-q4_0.gguf PREFIX=h38 SPLIT=data/split_rt_heldout38.json \
    bash scripts/rt_dev.sh $tag-ft $DEPLOY > logs/eval_h38_$tag-ft.log 2>&1
  kill $(cat logs/dev_student.pid) 2>/dev/null; sleep 10
  CUDA_VISIBLE_DEVICES=1 ~/llama.cpp/build-cuda/bin/llama-server -m $PWD/$out/ft-q4_0.gguf -ngl 99 -c 65536 -np 4 --jinja --skip-chat-parsing \
    --port 8140 --alias rt > logs/${tag}_conv.srv.log 2>&1 &
  ST=$!
  for i in $(seq 120); do curl -sf localhost:8140/health >/dev/null && break; sleep 3; done
  python3 distill/convert_targets.py --journals runs/student/h38-ft-v5 --urls http://127.0.0.1:8140/v1 --model rt --out runs/convert/h38-$tag > /dev/null 2>&1
  python3 eval/judge_prose_notes.py --dir runs/convert/h38-$tag --judge-url http://127.0.0.1:8700/v1 \
    --out reports/judge_prose_notes_h38-$tag.json 2>&1 | tail -1 | tee -a $R
  python3 eval/title_eval.py --run runs/student/h38-ft-v5 --split data/split_rt_heldout38.json --gold runs/v2/w4000 \
    --titler http://127.0.0.1:8140/v1:rt --judge http://127.0.0.1:8700/v1:judge --tag h38/$tag --out reports/titles_h38_$tag.json 2>&1 | tail -1 | tee -a $R
  kill $ST; sleep 5
  grep -E "^runs/student|^decision|precision" logs/eval_h38_$tag-ft.log | tee -a $R
done
kill $(cat logs/judge_dev.pid) 2>/dev/null
python3 eval/rt_report.py "h38-*" | grep -E "^run|m1-ft|g350-ft|q35-0.8b|gemma3-1b|lfm25-1.2b|hy-0.5b|q3-0.6b|ft-v8" | tee -a $R
echo "SCREEN DONE $(date)" | tee -a $R
