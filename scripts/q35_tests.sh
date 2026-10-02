#!/usr/bin/env bash
# Qwen3.5-0.8B (GRPO): the cheap levers against its contradictions, after v11.
#   1. abstention on uncertainty (eval/abstain.py): a rerun with token log-probs, then drop the
#      least confident notes (keep 90 % / 80 %) and judge coverage again
#   2. shorter windows (1k tokens instead of 2k), no retraining
#   3. the attribution guard (eval/attribution_guard.py)
#   4. if v11's contrastive DPO helped Gemma (minutes contradicted <= 15 %): the same pairs on 0.8B
set -uo pipefail
cd /home/luigi/meeting-summarizer
until grep -qE "V11 DONE|dpo failed|merge failed" reports/v11_contrast.txt 2>/dev/null && ! pgrep -f "^bash scripts/v11_contrast.sh" >/dev/null; do sleep 60; done
R=reports/q35_tests.txt
PY=~/.venvs/vllm/bin/python
M=$PWD/runs/sft/q35-0.8b/grpo-q4_0.gguf
DEPLOY="--harness v5 --no-check --overview none --number-section --read-max-tokens 400 --restart-journal-tokens 2500 --ctx 8192"
J="--judge-url http://127.0.0.1:8700/v1 --judge-model judge --split data/split_rt_heldout38.json"
echo "start $(date)" | tee -a $R
env SERVER_ARGS=--skip-chat-parsing MODEL=$M PREFIX=h38 SPLIT=data/split_rt_heldout38.json \
  bash scripts/rt_dev.sh q35-grpo-lp $DEPLOY --logprobs > logs/eval_h38_q35-grpo-lp.log 2>&1
grep -E "^runs/student|^decision|precision" logs/eval_h38_q35-grpo-lp.log | tee -a $R
python3 eval/abstain.py --run h38-q35-grpo-lp 2>&1 | tee -a $R
for q in 90 80; do
  python3 eval/judge_prose.py --candidate runs/student/h38-q35-grpo-lp-keep$q $J --gold runs/v2/w4000 \
    --out reports/judge_prose_h38-q35-grpo-lp-keep$q.json 2>&1 | tail -1 | tee -a $R
done
env SERVER_ARGS=--skip-chat-parsing MODEL=$M PREFIX=h38 SPLIT=data/split_rt_heldout38.json \
  bash scripts/rt_dev.sh q35-grpo-w1k $DEPLOY --window-tokens 1000 > logs/eval_h38_q35-grpo-w1k.log 2>&1
echo "== 1k windows" | tee -a $R
grep -E "^runs/student|^decision|precision" logs/eval_h38_q35-grpo-w1k.log | tee -a $R
python3 eval/attribution_guard.py --run runs/student/h38-q35-0.8b-grpo --out runs/student/h38-q35-0.8b-grpo-ag | tee -a $R
python3 eval/judge_prose_tx.py --candidate runs/student/h38-q35-0.8b-grpo-ag $J --transcripts data/v2/transcripts \
  --out reports/judge_prose_tx_h38-q35-0.8b-grpo-ag.json 2>&1 | tail -1 | tee -a $R
python3 eval/judge_prose.py --candidate runs/student/h38-q35-0.8b-grpo-ag $J --gold runs/v2/w4000 \
  --out reports/judge_prose_h38-q35-0.8b-grpo-ag.json 2>&1 | tail -1 | tee -a $R
V11=$(grep -E "^runs/student/h38-ft-v11:.*contradicted" logs/eval_h38_v11.log | head -1 | sed -E 's/.*contradicted ([0-9]+)%.*/\1/')
echo "v11 minutes contradicted: ${V11:-?} %" | tee -a $R
if [ -n "$V11" ] && [ "$V11" -le 15 ]; then
  kill $(cat logs/judge_dev.pid) 2>/dev/null; kill $(cat logs/dev_student.pid) 2>/dev/null
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ]; do sleep 5; done
  CUDA_VISIBLE_DEVICES=0,1 $PY -m torch.distributed.run --nproc_per_node 2 distill/dpo_agent.py --base Qwen/Qwen3.5-0.8B \
    --pairs data/train/contrast_pairs.jsonl --sft-adapter runs/sft/q35-0.8b/grpo/step150/policy --out runs/sft/q35-0.8b/dpo \
    --beta 0.1 --alpha 0.2 --lr 5e-6 --epochs 1 > logs/dpo_q35.log 2>&1 || echo "q35 dpo failed" | tee -a $R
  CUDA_VISIBLE_DEVICES=0 $PY distill/merge_agent_lora.py --base Qwen/Qwen3.5-0.8B --adapter runs/sft/q35-0.8b/dpo \
    --out runs/sft/q35-0.8b/dpo-q4_0.gguf > logs/merge_q35_dpo.log 2>&1 \
  && env SERVER_ARGS=--skip-chat-parsing MODEL=$PWD/runs/sft/q35-0.8b/dpo-q4_0.gguf PREFIX=h38 SPLIT=data/split_rt_heldout38.json \
    bash scripts/rt_dev.sh q35-dpo $DEPLOY > logs/eval_h38_q35-dpo.log 2>&1
  echo "== q35 contrastive DPO" | tee -a $R
  grep -E "^runs/student|^decision|precision" logs/eval_h38_q35-dpo.log | tee -a $R
fi
kill $(cat logs/judge_dev.pid) 2>/dev/null; kill $(cat logs/dev_student.pid) 2>/dev/null
python3 eval/rt_report.py "h38-q35*" | tee -a $R
echo "Q35 TESTS DONE $(date)" | tee -a $R
