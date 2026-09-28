#!/usr/bin/env bash
# The acceptance question: is each deployable method's summary comparable to the teacher's
# single-pass summary of the same meeting? Head-to-head, both orders, on every session where the
# reference exists.
set -uo pipefail
ROOT=/home/luigi/meeting-summarizer; cd $ROOT
STATUS=$ROOT/logs/overnight_status.txt
say() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
gpus_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -lt 2000 ]; }
stop_servers() { pkill -f "[v]llm.entrypoints.openai.api_server" 2>/dev/null; until gpus_free; do sleep 10; done; }

while pgrep -f "[a]ll_methods.sh" >/dev/null; do sleep 60; done
REF=runs/singlepass/w64k
if [ ! -d "$REF" ] || [ -z "$(ls $REF/*.json 2>/dev/null)" ]; then
  say "AA: no reference summaries; cannot judge comparability"; exit 1
fi

say "=== AA: comparability against the teacher's single-pass summary"
stop_servers
nohup /home/luigi/.venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model baidu/ERNIE-4.5-21B-A3B-PT --served-model-name judge --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 --max-model-len 32768 --max-num-seqs 16 --trust-remote-code \
  --enable-prefix-caching --port 8000 > logs/judge_pairwise.log 2>&1 &
sleep 20
waited=0
while ! curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null; do
  pgrep -f "[v]llm.entrypoints.openai.api_server" >/dev/null || { say "AA: judge died"; tail -8 logs/judge_pairwise.log | tee -a "$STATUS"; exit 1; }
  sleep 10; waited=$((waited+10))
  [ $waited -gt 2400 ] && { say "AA: timeout"; exit 1; }
done
say "AA: judge ready"

for cand in runs/final2/mapreduce/w4000 runs/refine/w4000; do
  [ -d "$cand" ] && [ -n "$(ls $cand/*.json 2>/dev/null)" ] || { say "AA: $cand empty"; continue; }
  python3 eval/pairwise.py --candidate "$cand" --reference "$REF" \
    --judge-url http://127.0.0.1:8000/v1 --judge-model judge \
    --out "reports/pairwise_$(basename $(dirname $cand))_$(basename $cand).json" 2>&1 | tee -a "$STATUS"
done
stop_servers
say "=== AA finished"
