#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
until grep -q CURVE_GEN_DONE reports/v2_curve.txt 2>/dev/null && grep -q EVIDENCE_GEN_DONE reports/v2_evidence.txt 2>/dev/null; do sleep 60; done
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 15
D="runs/student/v2-minicpm5-frac025 runs/student/v2-minicpm5-frac05 runs/student/v2-minicpm5-evidence"
python3 eval/score_v2.py $D | tee -a reports/v2_round4.txt
JUDGE_SCRIPT=judge_prose_tx DIRS="$D" bash scripts/v2_judge.sh >> logs/v2_judge_r4.log 2>&1
grep sentences reports/v2_sft_night.txt | tail -3 | tee -a reports/v2_round4.txt
echo ROUND4 DONE | tee -a reports/v2_round4.txt
