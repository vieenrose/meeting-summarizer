#!/usr/bin/env bash
# Pick the teacher by measurement, not by reputation.
#
# Each candidate summarises the same eight sessions (spanning 20 KB to 207 KB of transcript) by the
# same two procedures, and is judged on how much of its output survives validation: citations that
# resolve to real transcript lines, coverage of each third of the meeting, length, no repeated
# points. A teacher whose output fails validation cannot be used for training whatever its
# reputation, and the local Qwen3.6-35B-A3B already showed the procedure matters more than the
# model (69% valid through the notes pipeline, 37% single-pass).
#
# Cost: the whole 39-session corpus runs about $0.20-$0.90 on these models, so an eight-session
# probe per candidate is a few cents.
set -uo pipefail
ROOT=/home/luigi/meeting-summarizer; cd $ROOT
SESSIONS=${SESSIONS:-$(cat data/bakeoff_sessions.txt)}
OUT=reports/teacher_bakeoff.txt
: > $OUT
say() { echo "$*" | tee -a $OUT; }

# Candidates come from eval/zen_probe.py: of 70 models on the gateway, 20 answer at all and in
# Traditional Chinese. The shortlist keeps one model per family generation across the price range.
CANDIDATES=${CANDIDATES:-$(cat data/teacher_candidates.txt)}

for model in $CANDIDATES; do
  say ""
  say "=== $model"
  for mode in singlepass notes; do
    outdir="runs/bakeoff/${model}/${mode}"
    if [ "$mode" = "singlepass" ]; then
      timeout 2400 python3 eval/run_singlepass.py --backend zen --model "$model" \
        --tokenizer Qwen/Qwen3.6-35B-A3B-FP8 --out "$outdir" --parallel 2 \
        --max-output-tokens 24000 --only $SESSIONS >> "logs/bakeoff_${model}_${mode}.log" 2>&1
    else
      timeout 2400 python3 eval/run_teacher.py --backend zen --model "$model" \
        --tokenizer Qwen/Qwen3.6-35B-A3B-FP8 --window-tokens 4000 --out "runs/bakeoff/${model}" \
        --parallel 2 --max-output-tokens 24000 --only $SESSIONS >> "logs/bakeoff_${model}_${mode}.log" 2>&1
      outdir="runs/bakeoff/${model}/w4000"
    fi
    n=$(ls "$outdir"/*.json 2>/dev/null | grep -vc score || true)
    if [ "${n:-0}" -eq 0 ]; then
      say "  $mode: no output (see logs/bakeoff_${model}_${mode}.log)"
      continue
    fi
    say "  $mode: $(python3 eval/validate_teacher.py --run-dir "$outdir" 2>/dev/null | head -1 | sed 's|.*: ||')"
  done
done

say ""
say "Validation pass rate is the selection criterion: teacher output that fails it cannot be used."
