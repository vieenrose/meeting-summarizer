#!/usr/bin/env bash
# Two lighter students on the same data and protocol as Gemma v6 (reading v5 + notes->prose + notes->title):
# MiniCPM5-1B and Granite-4.0-350M. Zero-shot baseline, then LoRA SFT (2 GPUs, data-parallel), then
# the same evaluation on the 38 held-out IVOD sessions (reading, prose fidelity, titles).
set -uo pipefail
cd /home/luigi/meeting-summarizer
# until grep -q "V9 DONE" reports/v9_grpo.txt 2>/dev/null && ! pgrep -f "^bash scripts/v9_eval.sh" >/dev/null
R=reports/small_students.txt
PY=~/.venvs/vllm/bin/python
gpu_free() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ]; do sleep 5; done; }
echo "start $(date)" | tee -a $R
SPECS=(
  "m1;openbmb/MiniCPM5-1B;.*model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj);chatml-nothink;8192"
  "g350;ibm-granite/granite-4.0-350m;.*model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|shared_mlp\.(input|output)_linear);auto;16384"
)
for spec in "${SPECS[@]}"; do
  IFS=';' read -r tag base targets tmpl ctx <<< "$spec"
  out=runs/sft/$tag
  mkdir -p $out
  # base, no fine-tuning
  snap=$($PY -c "from huggingface_hub import snapshot_download as s; print(s('$base'))")
  $PY ~/llama.cpp/convert_hf_to_gguf.py $snap --outtype f16 --outfile /dev/shm/$tag-base-f16.gguf > logs/${tag}_base_convert.log 2>&1 \
    && ~/llama.cpp/build-cuda/bin/llama-quantize /dev/shm/$tag-base-f16.gguf $out/base-q4_0.gguf Q4_0 > /dev/null 2>&1
  rm -f /dev/shm/$tag-base-f16.gguf
  # SFT
  gpu_free
  CUDA_VISIBLE_DEVICES=0,1 $PY -m torch.distributed.run --nproc_per_node 2 distill/sft_agent.py --base $base \
    --targets "$targets" --template $tmpl --rank 32 --lr 2e-4 --epochs 2 \
    --rows data/train/agent_sft_rows_v6.jsonl --out $out/lora > logs/sft_$tag.log 2>&1 || { echo "$tag sft failed" | tee -a $R; continue; }
  grep -E "val loss|trainable" logs/sft_$tag.log | tee -a $R
  CUDA_VISIBLE_DEVICES=0 $PY distill/merge_agent_lora.py --base $base --adapter $out/lora/epoch1 --out $out/ft-q4_0.gguf \
    > logs/merge_$tag.log 2>&1 || { echo "$tag merge failed" | tee -a $R; continue; }
  ls -la $out/*.gguf | tee -a $R
  DEPLOY="--harness v5 --no-check --overview none --number-section --read-max-tokens 400 --restart-journal-tokens 2500 --ctx $ctx"
  for v in base ft; do
    env MODEL=$PWD/$out/$v-q4_0.gguf PREFIX=h38 SPLIT=data/split_rt_heldout38.json \
      bash scripts/rt_dev.sh $tag-$v $DEPLOY > logs/eval_h38_$tag-$v.log 2>&1
  done
  kill $(cat logs/dev_student.pid) 2>/dev/null; sleep 10
  CUDA_VISIBLE_DEVICES=1 ~/llama.cpp/build-cuda/bin/llama-server -m $PWD/$out/ft-q4_0.gguf -ngl 99 -c 65536 -np 4 --jinja \
    --port 8140 --alias rt > logs/${tag}_conv.srv.log 2>&1 &
  ST=$!
  until curl -sf localhost:8140/health >/dev/null; do sleep 3; done
  python3 distill/convert_targets.py --journals runs/student/h38-ft-v5 --urls http://127.0.0.1:8140/v1 --model rt --out runs/convert/h38-$tag > /dev/null 2>&1
  python3 eval/judge_prose_notes.py --dir runs/convert/h38-$tag --judge-url http://127.0.0.1:8700/v1 \
    --out reports/judge_prose_notes_h38-$tag.json 2>&1 | tail -1 | tee -a $R
  python3 eval/title_eval.py --run runs/student/h38-ft-v5 --split data/split_rt_heldout38.json --gold runs/v2/w4000 \
    --titler http://127.0.0.1:8140/v1:rt --judge http://127.0.0.1:8700/v1:judge --tag h38/$tag --out reports/titles_h38_$tag.json 2>&1 | tail -1 | tee -a $R
  kill $ST; sleep 5
  for v in base ft; do echo "== $tag-$v"; grep -E "^runs/student|^decision|precision" logs/eval_h38_$tag-$v.log; done | tee -a $R
done
kill $(cat logs/judge_dev.pid) 2>/dev/null
python3 eval/rt_report.py "h38-*" | grep -E "^run|m1|g350|ft-v8|ft-v5" | tee -a $R
echo "SMALL DONE $(date)" | tee -a $R
