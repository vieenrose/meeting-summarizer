#!/usr/bin/env bash
# Learning curve: MiniCPM5 SFT on 25% and 50% of the rows (100% = runs/sft/minicpm5-v2-ivod-ali).
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python; R=reports/v2_curve.txt
for w in 0.25 0.5; do
  tag=frac${w/./}
  source ~/.venvs/vllm/bin/activate
  CUDA_VISIBLE_DEVICES=1 python3 distill/sft_gemma.py --base openbmb/MiniCPM5-2B \
    --rows data/train/v2_gold_rows_all.jsonl:$w data/train/alimeeting_gold_rows.jsonl:$w \
    --synth-repeat 3 --epochs 2 --out runs/sft/minicpm5-$tag > logs/sft_$tag.log 2>&1
  deactivate
  echo "$tag eval_loss $(grep -o "'eval_loss': '[0-9.a-z]*'" logs/sft_$tag.log | tr '\n' ' ')" | tee -a $R
  CUDA_VISIBLE_DEVICES=1 timeout 7200 $PY eval/run_student_vllm.py --base openbmb/MiniCPM5-2B --adapter runs/sft/minicpm5-$tag/final \
    --out runs/student/v2-minicpm5-$tag --max-model-len 16384 > logs/eval_$tag.log 2>&1
  for p in $(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 10
done
echo CURVE_GEN_DONE | tee -a $R
