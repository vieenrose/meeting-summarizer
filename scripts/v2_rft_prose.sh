#!/usr/bin/env bash
# RFT on the reduce step with the transcript-grounded judge.
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python; R=reports/v2_rft_prose.txt; AD=runs/sft/minicpm5-v2-ivod-ali/final
say() { echo "$(date '+%F %T') $*" | tee -a $R; }
free() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 10; }
CUDA_VISIBLE_DEVICES=0 $PY eval/run_student_vllm.py --base openbmb/MiniCPM5-2B --adapter $AD --split data/split_rft_ivod.json \
  --transcripts data/v2/transcripts --out runs/student/rft-train-ivod --max-model-len 16384 > logs/rft_train_ivod.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY eval/run_student_vllm.py --base openbmb/MiniCPM5-2B --adapter $AD --split data/split_rft_ali.json \
  --transcripts data/alimeeting/transcripts --out runs/student/rft-train-ali --max-model-len 16384 > logs/rft_train_ali.log 2>&1 &
wait; free
say "student notes on train: $(ls runs/student/rft-train-ivod | wc -l) ivod, $(ls runs/student/rft-train-ali | wc -l) ali"
CUDA_VISIBLE_DEVICES=0 $PY eval/sample_prose.py --run-dirs runs/student/rft-train-ivod runs/student/rft-train-ali --adapter $AD \
  --out-prefix runs/student/rft-prose-s > logs/rft_sample_prose.log 2>&1; free
tail -1 logs/rft_sample_prose.log | tee -a $R
for i in 0 1 2 3; do
  JUDGE_SCRIPT=judge_prose_tx JUDGE_TAG=ivod_ JUDGE_EXTRA="--split data/split_rft_ivod.json --transcripts data/v2/transcripts" \
    DIRS=runs/student/rft-prose-s$i bash scripts/v2_judge.sh >> logs/rft_judge.log 2>&1
  JUDGE_SCRIPT=judge_prose_tx JUDGE_TAG=ali_ JUDGE_EXTRA="--split data/split_rft_ali.json --transcripts data/alimeeting/transcripts" \
    DIRS=runs/student/rft-prose-s$i bash scripts/v2_judge.sh >> logs/rft_judge.log 2>&1
done
say "RFT_JUDGED"
