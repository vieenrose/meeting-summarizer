#!/usr/bin/env bash
# Mobile KD: float LoRA on Google's mobile weights trained on the v6 rows with v11's top-32
# distributions as soft targets (carries v8's GRPO and v11's contrastive DPO), then GPTQ on
# Google's grid, injection into Google's .litertlm (fp32 GPU activations), IVOD 38 eval.
set -uo pipefail
cd /home/luigi/meeting-summarizer
PY=~/.venvs/vllm/bin/python
CUDA_VISIBLE_DEVICES=0,1 $PY -m torch.distributed.run --nproc_per_node 2 --master_port 29534 distill/sft_mobile_qat.py --base /dev/shm/mobile-tf \
  --rows data/train/agent_sft_rows_v6.jsonl --epochs 1 --weight-fq off --kd runs/kd/v11_top32.pt --kd-weight 0.7 \
  --out runs/sft/mobile/lora-kd11 > logs/sft_mobile_kd11.log 2>&1 || { echo "kd training failed"; exit 1; }
grep "val loss" logs/sft_mobile_kd11.log
CUDA_VISIBLE_DEVICES=0 $PY distill/gptq_mobile.py --lora runs/sft/mobile/lora-kd11/epoch0/qat_lora.safetensors --damp 0.001 --out /dev/shm/mobile-gptq-kd11 > logs/gptq_mobile_kd11.log 2>&1
grep "val loss" logs/gptq_mobile_kd11.log
/dev/shm/ltvenv/bin/python distill/inject_litertlm.py --mobile /dev/shm/mobile-gptq-kd11 --activation-type fp32 --out /dev/shm/mgptqkd11f32.litertlm 2>&1 | tail -1
mkdir -p /dev/shm/litert-models-mgptqkd11f32; ln -sf /dev/shm/mgptqkd11f32.litertlm /dev/shm/litert-models-mgptqkd11f32/model.litertlm
ln -sfn /dev/shm/litert-models-mgptqkd11f32 ~/.litert-lm/models/mgptqkd11f32
bash scripts/lt_eval.sh mgptqkd11-litert "mgptqkd11f32,gpu,8192" > logs/lt_eval_mgptqkd11.log 2>&1
echo "MOBILE KD DONE $(date)"
