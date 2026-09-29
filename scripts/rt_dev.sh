#!/usr/bin/env bash
# One harness-tuning iteration for Gemma-4-E2B on the dev split by default (8 training sessions, never
# held-out): run the agent with the given options into runs/student/dev-<tag>, then judge notes
# (25 sampled per session) and minutes against the transcript, and coverage against the gold.
# Keeps a judge (Gemma-4-31B, vLLM TP2, port 8700) and the student (llama.cpp, port 8140) up.
# usage: rt_dev.sh <tag> [realtime_agent options...]
set -uo pipefail
cd /home/luigi/meeting-summarizer
tag=$1; shift
SPLIT=${SPLIT:-data/split_rt_dev.json}
PREFIX=${PREFIX:-dev}   # PREFIX=rt SPLIT=data/split_rt_bakeoff.json for the held-out measure
if ! curl -sf localhost:8700/v1/models >/dev/null; then
  ~/.venvs/cu13/bin/python -m vllm.entrypoints.openai.api_server --model bahadirakdemir/gemma-4-31B-it-text-fp8 \
    --served-model-name judge --tensor-parallel-size 2 --max-model-len 24000 --gpu-memory-utilization 0.72 \
    --max-num-seqs 8 --port 8700 > logs/judge_dev.log 2>&1 &
  echo $! > logs/judge_dev.pid
  until curl -sf localhost:8700/v1/models >/dev/null; do sleep 15; done
fi
if ! curl -sf localhost:8140/health >/dev/null; then
  CUDA_VISIBLE_DEVICES=1 ~/llama.cpp/build-cuda/bin/llama-server -m ~/Bonsai-demo/models/small/gemma4e2b/gemma-4-E2B_q4_0-it.gguf \
    -ngl 99 -c 131072 -np 4 --jinja --swa-full --port 8140 --alias rt > logs/dev_student.log 2>&1 &
  echo $! > logs/dev_student.pid
  until curl -sf localhost:8140/health >/dev/null; do sleep 3; done
fi
out=runs/student/$PREFIX-$tag
python3 eval/realtime_agent.py --url http://127.0.0.1:8140/v1 --model rt --parallel 4 --split $SPLIT \
  --phone-pp 34.6 --phone-tg 7.0 --out $out "$@" 2>&1 | grep -v -i warn | tail -n 8
python3 - $PREFIX-$tag <<'P'
import glob, json, os, random, sys
src = f'runs/student/{sys.argv[1]}'; dst = f'runs/student/nj25-{sys.argv[1]}'
os.makedirs(dst, exist_ok=True)
for f in glob.glob(f'{src}/ivod_*.json'):
    s = os.path.basename(f); r = json.load(open(f)); rng = random.Random(s + '25')
    ns = rng.sample(r['notes'], min(25, len(r['notes'])))
    prose = ''.join(n['text'].rstrip('。') + f" [{n['ts']}]。" for n in ns)
    json.dump({'notes': r['notes'], 'prose': prose}, open(f'{dst}/{s}', 'w'), ensure_ascii=False)
P
J="--judge-url http://127.0.0.1:8700/v1 --judge-model judge --split $SPLIT"
for d in $out runs/student/nj25-$PREFIX-$tag; do
  python3 eval/judge_prose_tx.py --candidate $d $J --out reports/judge_prose_tx_$(basename $d).json 2>&1 | tail -1
done
python3 eval/judge_prose.py --candidate $out $J --out reports/judge_prose_$PREFIX-$tag.json 2>&1 | tail -1
python3 eval/minutes_report.py --run $PREFIX-$tag | tail -1
python3 eval/rt_report.py "$PREFIX-$tag"
