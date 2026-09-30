#!/usr/bin/env bash
# On-policy correction + DPO for the agent-SFT Gemma-4-E2B, then evaluation with the deployment
# harness on dev and on the 38 held-out sessions (same judge).
set -uo pipefail
cd /home/luigi/meeting-summarizer
R=reports/dpo_night.txt
PY=~/.venvs/vllm/bin/python
DEPLOY="--harness v3 --no-check --overview none --number-section --read-max-tokens 400 --restart-journal-tokens 2500"
until grep -q "ONPOL JUDGED" logs/onpol_judge.log 2>/dev/null; do sleep 60; done
grep "notes-onpol" reports/v2_sft_night.txt | tail -1 | tee -a $R
ART=~/models/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer
srv=()
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g ~/ninfer/build/apps/ninfer-serve $ART --host 127.0.0.1 --port 812$g --max-context 32768 \
    --kv-capacity auto --max-concurrency 4 --kv-dtype fp8 --spec mtp --draft-tokens 3 --lm-head-draft --model-id q38 \
    > logs/ninfer_correct_$g.log 2>&1 &
  srv+=($!)
done
for g in 0 1; do until curl -sf localhost:812$g/v1/models >/dev/null; do sleep 5; done; done
python3 distill/correct_onpolicy.py 2>&1 | grep -v -i warn | tail -1 | tee -a $R
kill "${srv[@]}"; sleep 30
CUDA_VISIBLE_DEVICES=0 $PY distill/dpo_agent.py > logs/dpo_agent.log 2>&1 || { echo "dpo failed" | tee -a $R; exit 1; }
grep -E "^pairs|step" logs/dpo_agent.log | tail -n 4 | tee -a $R
CUDA_VISIBLE_DEVICES=0 $PY distill/merge_agent_lora.py --adapter runs/sft/agent/dpo/policy --out runs/sft/agent/dpo-q4_0.gguf \
  > logs/merge_dpo.log 2>&1 || { echo "merge failed" | tee -a $R; exit 1; }
M=$PWD/runs/sft/agent/dpo-q4_0.gguf
MODEL=$M bash scripts/rt_dev.sh ft-dpo-deploy $DEPLOY > logs/dev_dpo.log 2>&1
MODEL=$PWD/runs/sft/agent/ft-ep0f16-q4_0.gguf bash scripts/rt_dev.sh ft-ep0-deploy $DEPLOY > logs/dev_ftdeploy.log 2>&1
python3 eval/rt_report.py "dev-ft-*-deploy" | tee -a $R
MODEL=$M PREFIX=h38 SPLIT=data/split_rt_heldout38.json bash scripts/rt_dev.sh gemma4-e2b-dpo-v3xc $DEPLOY > logs/h38_dpo.log 2>&1
python3 eval/rt_report.py "h38-gemma4-e2b-*" | tee -a $R
for r in h38-gemma4-e2b-ft-ep0-v3xc h38-gemma4-e2b-dpo-v3xc; do echo "== $r"; python3 eval/minutes_report.py --run $r; done | tee -a $R
kill $(cat logs/judge_dev.pid) $(cat logs/dev_student.pid) 2>/dev/null
echo "DPO DONE $(date)" | tee -a $R
