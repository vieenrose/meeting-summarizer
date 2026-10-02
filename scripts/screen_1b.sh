#!/usr/bin/env bash
# 1B-class screen (user: "Qwen3 1.7B is too big"): candidates of at most 1.2B, fine-tuned exactly like MiniCPM5-1B
# (agent_sft_rows_v6, LoRA r32, 2 epochs), then evaluated on IVOD 38 (reading, prose, titles).
# GPU split (user, 2026-10-01): Gemma work keeps its lane; the 1B trainings run on GPU 0 alone, in the
# memory left beside the Gemma evaluation (judge at 0.62 per card), from the end of v10's GRPO steps.
set -uo pipefail
cd /home/luigi/meeting-summarizer
until [ -s logs/merge_v10.log ] || grep -q "grpo failed" reports/v10_grpo.txt 2>/dev/null; do sleep 60; done
R=reports/screen_1b.txt
PY=~/.venvs/vllm/bin/python
gpu_free() { kill $(cat logs/judge_dev.pid) 2>/dev/null; kill $(cat logs/dev_student.pid) 2>/dev/null
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ]; do sleep 5; done; }
echo "start $(date)" | tee -a $R
ATT='.*model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)'
SPECS=(
  "q35-0.8b;Qwen/Qwen3.5-0.8B;.*model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj);chatml-nothink"
  "gemma3-1b;google/gemma-3-1b-it;$ATT;auto"
  "lfm25-1.2b;LiquidAI/LFM2.5-1.2B-Instruct;.*model\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|conv\.(in|out)_proj|feed_forward\.w[123]);auto"
  "hy-0.5b;tencent/Hunyuan-0.5B-Instruct;$ATT;hunyuan-nothink"
  "q3-0.6b;Qwen/Qwen3-0.6B;$ATT;chatml-nothink"
)
DEPLOY="--harness v5 --no-check --overview none --number-section --read-max-tokens 400 --restart-journal-tokens 2500 --ctx 8192"
for spec in "${SPECS[@]}"; do
  IFS=';' read -r tag base targets tmpl <<< "$spec"
  out=runs/sft/$tag
  mkdir -p $out
  # start only once the Gemma evaluation's judge holds its memory (vLLM needs its share free at startup)
  until { curl -sf localhost:8700/v1/models >/dev/null || grep -q "V10 DONE" reports/v10_grpo.txt 2>/dev/null; } \
    && [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 23000 ]; do sleep 30; done
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 $PY distill/sft_agent.py --base $base \
    --targets "$targets" --template $tmpl --rank 32 --lr 2e-4 --epochs 2 --accum 8 \
    --rows data/train/agent_sft_rows_v6.jsonl --out $out/lora > logs/sft_$tag.log 2>&1 || { echo "$tag sft failed" | tee -a $R; continue; }
  echo "== $tag" | tee -a $R
  grep -E "val loss|trainable" logs/sft_$tag.log | tee -a $R
  CUDA_VISIBLE_DEVICES=0 $PY distill/merge_agent_lora.py --base $base --adapter $out/lora/epoch1 --out $out/ft-q4_0.gguf \
    > logs/merge_$tag.log 2>&1 && echo "$tag" >> logs/screen_1b.queue || echo "$tag merge failed" | tee -a $R
done
echo "TRAINED $(date)" | tee -a $R
