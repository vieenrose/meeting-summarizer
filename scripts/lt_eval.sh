#!/usr/bin/env bash
# Reading-agent evaluation of a model served by LiteRT-LM (litert-lm serve, OpenAI API, GPU backend):
# same harness, same judge (Gemma-4-31B, vLLM TP2 at 0.62/card) and same reports as scripts/rt_dev.sh.
# usage: [SKIP_READ=1] lt_eval.sh <tag> <model param, e.g. "v11,gpu,8192">
set -uo pipefail
cd /home/luigi/meeting-summarizer
tag=$1; MP=$2
SPLIT=data/split_rt_heldout38.json; TRANSCRIPTS=data/v2/transcripts; GOLD=runs/v2/w4000; PREFIX=h38
DEPLOY="--harness v5 --no-check --overview none --number-section --read-max-tokens 400 --restart-journal-tokens 2500 --ctx 8192"
out=runs/student/$PREFIX-$tag
[ -n "${SKIP_READ:-}" ] || python3 eval/realtime_agent.py --url http://127.0.0.1:9379/v1 --model "$MP" --parallel 2 --split $SPLIT --transcripts $TRANSCRIPTS \
  --phone-pp 34.6 --phone-tg 7.0 --out $out $DEPLOY 2>&1 | grep -v -i warn | tail -n 4
# the judge needs 62 % of each card: wait for any mobile training / GPTQ to finish
while pgrep -f "sft_mobile_qat|gptq_mobile" >/dev/null; do sleep 60; done
if ! curl -sf localhost:8700/v1/models >/dev/null; then
  ~/.venvs/cu13/bin/python -m vllm.entrypoints.openai.api_server --model bahadirakdemir/gemma-4-31B-it-text-fp8 \
    --served-model-name judge --tensor-parallel-size 2 --max-model-len 24000 --gpu-memory-utilization 0.62 \
    --max-num-seqs 8 --port 8700 > logs/judge_dev.log 2>&1 &
  echo $! > logs/judge_dev.pid
  until curl -sf localhost:8700/v1/models >/dev/null; do sleep 15; done
fi
J="--judge-url http://127.0.0.1:8700/v1 --judge-model judge --split $SPLIT"
python3 eval/judge_prose_tx.py --candidate $out $J --transcripts $TRANSCRIPTS --out reports/judge_prose_tx_$PREFIX-$tag.json 2>&1 | tail -1
python3 eval/judge_prose.py --candidate $out $J --gold $GOLD --out reports/judge_prose_$PREFIX-$tag.json 2>&1 | tail -1
python3 eval/minutes_report.py --run $PREFIX-$tag --gold $GOLD | tail -1
python3 eval/section_precision.py --candidate $out --split $SPLIT --transcripts $TRANSCRIPTS --judge-url http://127.0.0.1:8700/v1 --out reports/section_precision_$PREFIX-$tag.json 2>&1 | tail -2
python3 eval/rt_report.py "$PREFIX-$tag"
