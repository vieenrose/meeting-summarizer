#!/usr/bin/env bash
# Overnight v2 SFT evaluation: held-out generation (notes -> points -> prose) for every arm, then
# mechanical scores and a judged prose comparison against the QA'd gold.
#
# Arms: untrained base, v1 SFT (v1 prompts, what it was trained on), v2 IVOD-only (ablation),
# v2 IVOD+AliMeeting (main). The judge is Gemma-4-31B: it taught AliMeeting but never IVOD, and
# every held-out session is IVOD. Waits on PID files, never on process names.
set -uo pipefail
ROOT=/home/luigi/meeting-summarizer; cd $ROOT
PY=~/vllm019/bin/python
REPORT=reports/v2_sft_night.txt; mkdir -p reports
say() { echo "$(date '+%F %T') $*" | tee -a $REPORT; }
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

evaluate() {  # gpu out prompt-version [adapter]
  local gpu=$1 out=$2 pv=$3 adapter=${4:-}
  [ -f "$out/.done" ] && return
  say "eval $out on GPU$gpu"
  CUDA_VISIBLE_DEVICES=$gpu timeout 5400 $PY eval/run_student_vllm.py ${adapter:+--adapter "$adapter"} \
    --out "$out" --prompt-version "$pv" > logs/eval_$(basename $out).log 2>&1 && touch "$out/.done"
  tail -1 logs/eval_$(basename $out).log | tee -a $REPORT
}

# GPU1: ablation, then the two reference arms.
(
  while alive logs/sft_g4_v2_ivod_only.pid; do sleep 60; done
  say "IVOD-only SFT finished: $(grep -o "'eval_loss': '[0-9.]*'" logs/sft_g4_v2_ivod_only.log | tr '\n' ' ')"
  evaluate 1 runs/student/v2-ivod-only v2 runs/sft/g4-v2-ivod-only/final
  evaluate 1 runs/student/v2-base v2
  evaluate 1 runs/student/v2-v1sft v1 runs/sft/g4it-s5/final
) &
G1=$!
# GPU0: the main run.
(
  while alive logs/sft_g4_v2_ivod_ali.pid; do sleep 60; done
  say "IVOD+Ali SFT finished: $(grep -o "'eval_loss': '[0-9.]*'" logs/sft_g4_v2_ivod_ali.log | tr '\n' ' ')"
  evaluate 0 runs/student/v2-ivod-ali v2 runs/sft/g4-v2-ivod-ali/final
) &
G0=$!
wait $G1 $G0
rm -rf runs/merged/*g4-v2* 2>/dev/null

say "=== mechanical scores"
python3 eval/score_v2.py runs/v2/w4000 runs/student/v2-base runs/student/v2-v1sft \
  runs/student/v2-ivod-only runs/student/v2-ivod-ali | tee -a $REPORT

say "=== judge: Gemma-4-31B fp8, TP2"
~/.venvs/cu13/bin/python -m vllm.entrypoints.openai.api_server \
  --model bahadirakdemir/gemma-4-31B-it-text-fp8 --served-model-name judge --tensor-parallel-size 2 \
  --max-model-len 24000 --gpu-memory-utilization 0.85 --max-num-seqs 8 --port 8700 \
  > logs/judge_v2_night.log 2>&1 &
JUDGE=$!
until curl -sf localhost:8700/v1/models >/dev/null; do
  kill -0 $JUDGE 2>/dev/null || { say "judge died, see logs/judge_v2_night.log"; exit 1; }
  sleep 15
done
for d in runs/v2/w4000 runs/student/v2-base runs/student/v2-v1sft runs/student/v2-ivod-only runs/student/v2-ivod-ali; do
  python3 eval/judge_prose.py --candidate $d --judge-url http://127.0.0.1:8700/v1 --judge-model judge \
    --out reports/judge_prose_$(basename $d).json 2>&1 | tail -1 | tee -a $REPORT
done
kill $JUDGE
say "NIGHT DONE"
