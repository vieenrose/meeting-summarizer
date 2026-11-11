#!/usr/bin/env bash
# 2026-11-11 multiple-choice form of the questionnaire: convert, then score the same candidates as qa_metric.sh/qa_spread.sh (code-graded).
cd ~/meeting-summarizer; R=reports/qa_mcq.txt; mkdir -p reports/qa_mcq
for p in ${WAITPIDS:-}; do while kill -0 $p 2>/dev/null; do sleep 20; done; done
for T in msdev q12 ali20; do python3 eval/qa_mcq.py data/qa/$T data/qa_mcq/$T 2>&1 | tail -n 1 >> $R; done
for T in msdev q12 ali20; do
  for C in s5-$T-e4bOPD sp4brds2-$T rdsft1p1u85s{2,3,4,5}-$T rdsft3p1s{2,3,4,5}-$T; do [ -d runs/student/$C ] && python3 eval/qa_score_mcq.py data/qa_mcq/$T runs/student/$C reports/qa_mcq/$C.json 2>&1 | tail -n 1 >> $R; done
  python3 eval/qa_score_mcq.py data/qa_mcq/$T runs/student/s5-$T-e4bOPD reports/qa_mcq/s5-$T-e4bOPD-cut1500.json --cut 1500 2>&1 | tail -n 1 >> $R
  python3 eval/qa_score_mcq.py data/qa_mcq/$T runs/student/rdsft1p1u85s2-$T reports/qa_mcq/rdsft1p1u85s2-$T.rep.json 2>&1 | tail -n 1 | sed 's/^/repeat /' >> $R
done; echo "MCQ DONE $(date -u +%H:%M)" >> $R
