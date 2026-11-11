#!/usr/bin/env bash
# 2026-11-11 questionnaire metric (user idea): build on the 3 dev sets, then validate it: transcript-as-summary ceiling is not possible (length), so
# checks = E4B (s5-*-e4bOPD) vs Spark control vs rdsft3p1 vs 4B zero-shot, plus E4B cut to 1500 chars (equal length).
cd ~/meeting-summarizer; export VIBEVOICE_PLUGIN=$HOME/.cache/ms-tools/stub PYTHONPATH=$HOME/.cache/ms-tools/stub; R=reports/qa_metric.txt; mkdir -p reports/qa data/qa
for p in ${WAITPIDS:-}; do while kill -0 $p 2>/dev/null; do sleep 20; done; done
for S in "msdev|data/split_msdev.json|data/moneyshow/transcripts" "q12|data/split_rt_quick12.json|data/v2/transcripts" "ali20|data/split_ali20.json|data/alimeeting/transcripts"; do IFS='|' read T SP TR <<< "$S"
  python3 eval/qa_build.py $SP $TR data/qa/$T 2>&1 | tail -n 1 >> $R
done; echo "BUILD DONE $(date -u +%H:%M)" >> $R
for T in msdev q12 ali20; do
  for C in s5-$T-e4bOPD rdsft1p1u85s2-$T rdsft1p1u85s3-$T rdsft3p1s2-$T sp4brds2-$T; do [ -d runs/student/$C ] && python3 eval/qa_score.py data/qa/$T runs/student/$C reports/qa/$C.json 2>&1 | tail -n 1 >> $R; done
  python3 eval/qa_score.py data/qa/$T runs/student/s5-$T-e4bOPD reports/qa/s5-$T-e4bOPD-cut1500.json --cut 1500 2>&1 | tail -n 1 >> $R
done; echo "QA DONE $(date -u +%H:%M)" >> $R
