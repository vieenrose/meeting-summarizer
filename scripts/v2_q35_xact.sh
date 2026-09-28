#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python; R=reports/v2_q35_xact.txt
source ~/.venvs/vllm/bin/activate
CUDA_VISIBLE_DEVICES=1 python3 distill/sft_gemma.py --base Qwen/Qwen3.5-4B --rows data/train/extract_act_map_rows.jsonl \
  --epochs 2 --out runs/sft/q35-xact-map > logs/sft_q35_xact_map.log 2>&1
deactivate
AD=$(ls -d runs/sft/q35-xact-map/final 2>/dev/null || ls -d runs/sft/q35-xact-map/checkpoint-* | tail -1)
echo "q35 xact map $AD $(grep -o "'eval_loss': '[0-9.a-z]*'" logs/sft_q35_xact_map.log | tr '\n' ' ')" | tee -a $R
CUDA_VISIBLE_DEVICES=1 timeout 9000 $PY eval/run_student_vllm.py --base Qwen/Qwen3.5-4B --adapter $AD --extract --acts \
  --out runs/student/v2-q35-xact --max-model-len 16384 > logs/eval_q35_xact.log 2>&1
for p in $(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 10; rm -rf runs/merged/runs__sft__q35-xact-map*
python3 - <<'P'
import json,glob,os,random,re
src,dst='runs/student/v2-q35-xact','runs/student/nj25-q35-xact'
os.makedirs(dst,exist_ok=True)
for f in glob.glob(f'{src}/ivod_*.json'):
    s=os.path.basename(f);r=json.load(open(f));rng=random.Random(s+'25')
    ns=rng.sample(r['notes'],min(25,len(r['notes'])))
    prose=''.join(re.sub(r'^[^:：]{1,15}[:：]\s*','',n['text']).rstrip('。')+f" [{n['ts']}]。" for n in ns)
    json.dump({'notes':r['notes'],'prose':prose},open(f'{dst}/{s}','w'),ensure_ascii=False)
P
python3 eval/score_v2.py runs/student/v2-q35-xact runs/student/v2-minicpm5-xact | tee -a $R
until grep -q "E4BX DONE" reports/v2_e4b_xact.txt 2>/dev/null; do sleep 60; done
JUDGE_SCRIPT=judge_prose_tx DIRS="runs/student/nj25-q35-xact" bash scripts/v2_judge.sh >> logs/v2_judge_q35x.log 2>&1
grep sentences reports/v2_sft_night.txt | tail -1 | tee -a $R
echo Q35X DONE | tee -a $R
