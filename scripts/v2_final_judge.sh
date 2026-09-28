#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
until grep -q "TA DONE" reports/v2_ta.txt 2>/dev/null && grep -q EVQ_GEN_DONE reports/v2_evq.txt 2>/dev/null; do sleep 120; done
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 15
JUDGE_SCRIPT=judge_prose_tx DIRS="runs/student/nj25-cad runs/student/nj25-cadref runs/student/v2-e4b-evq runs/student/v2-e4b-evq-guard" bash scripts/v2_judge.sh > logs/v2_judge_final.log 2>&1
grep sentences reports/v2_sft_night.txt | tail -4 | tee -a reports/v2_final.txt
echo FINAL DONE | tee -a reports/v2_final.txt
