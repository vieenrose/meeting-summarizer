#!/usr/bin/env bash
# 2026-11-11 step 2 of the questionnaire plan: is the score spread between samples of one model on one session larger than noise?
# Scores the remaining seeds (control s4 s5, rdsft3p1 s3-s5) and rescoring control s2 once more (repeat noise), then eval/qa_spread.py.
cd ~/meeting-summarizer; R=reports/qa_metric.txt
for p in ${WAITPIDS:-}; do while kill -0 $p 2>/dev/null; do sleep 20; done; done
for T in msdev q12 ali20; do
  for C in rdsft1p1u85s4-$T rdsft1p1u85s5-$T rdsft3p1s3-$T rdsft3p1s4-$T rdsft3p1s5-$T; do [ -d runs/student/$C ] && python3 eval/qa_score.py data/qa/$T runs/student/$C reports/qa/$C.json 2>&1 | tail -n 1 >> $R; done
  python3 eval/qa_score.py data/qa/$T runs/student/rdsft1p1u85s2-$T reports/qa/rdsft1p1u85s2-$T.rep.json 2>&1 | tail -n 1 | sed 's/^/repeat /' >> $R
done
python3 eval/qa_spread.py >> $R 2>&1; echo "SPREAD DONE $(date -u +%H:%M)" >> $R
