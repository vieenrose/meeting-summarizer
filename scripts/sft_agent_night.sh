#!/usr/bin/env bash
# Overnight: distill Gemma-4-E2B on the 27B's realtime-agent traces and evaluate it.
#  1. LoRA SFT, 2 epochs (distill/sft_agent.py), one adapter per epoch
#  2. merge each epoch into the QAT weights -> Q4_0 GGUF (distill/merge_agent_lora.py)
#  3. dev split, tuned protocol v3x: each epoch + the base through the same conversion (control)
#  4. held-out (10 bake-off sessions): the epoch with the fewest contradicted minutes on dev
set -uo pipefail
cd /home/luigi/meeting-summarizer
R=reports/sft_agent_night.txt
PY=~/.venvs/vllm/bin/python
V3X="--harness v3 --no-check --overview none --number-section"
echo "start $(date)" | tee -a $R
CUDA_VISIBLE_DEVICES=0 $PY distill/sft_agent.py --out runs/sft/agent/lora > logs/sft_agent.log 2>&1 || { echo "train failed" | tee -a $R; exit 1; }
grep -E "val loss" logs/sft_agent.log | tee -a $R
for ep in 0 1; do
  CUDA_VISIBLE_DEVICES=0 $PY distill/merge_agent_lora.py --adapter runs/sft/agent/lora/epoch$ep \
    --out runs/sft/agent/ft-ep$ep-q4_0.gguf > logs/merge_ep$ep.log 2>&1 || { echo "merge ep$ep failed" | tee -a $R; exit 1; }
done
for spec in "ft-ep0|runs/sft/agent/ft-ep0-q4_0.gguf" "ft-ep1|runs/sft/agent/ft-ep1-q4_0.gguf" "conv-base|runs/sft/agent/base-q4_0.gguf"; do
  IFS='|' read -r tag m <<< "$spec"
  MODEL=$PWD/$m bash scripts/rt_dev.sh v3x-$tag $V3X > logs/dev_v3x-$tag.log 2>&1
  grep -E "^dev-" logs/dev_v3x-$tag.log | tee -a $R
  grep -E "^decision" logs/dev_v3x-$tag.log | sed "s/^/$tag /" | tee -a $R
done
best=$(python3 - <<'P'
import json
def contr(tag):
    rows = [r for r in json.load(open(f"reports/judge_prose_tx_dev-v3x-{tag}.json"))["rows"]
            if r["verdict"] in ("supported", "contradicted", "unsupported")]
    return sum(r["verdict"] == "contradicted" for r in rows) / max(1, len(rows))
print(min(["ft-ep0", "ft-ep1"], key=contr))
P
)
echo "best on dev: $best" | tee -a $R
MODEL=$PWD/runs/sft/agent/$best-q4_0.gguf PREFIX=rt SPLIT=data/split_rt_bakeoff.json \
  bash scripts/rt_dev.sh gemma4-e2b-$best-v3x $V3X > logs/ho_$best.log 2>&1
python3 eval/rt_report.py "rt-gemma4-e2b*" | tee -a $R
python3 eval/minutes_report.py --run rt-gemma4-e2b-$best-v3x | tee -a $R
kill $(cat logs/judge_dev.pid) $(cat logs/dev_student.pid) 2>/dev/null
echo "NIGHT DONE $(date)" | tee -a $R
