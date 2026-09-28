#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python; R=reports/v2_e4b_xact.txt
source ~/.venvs/vllm/bin/activate
CUDA_VISIBLE_DEVICES=0 python3 distill/sft_gemma.py --base google/gemma-4-E4B-it --rows data/train/extract_act_map_rows.jsonl \
  --epochs 2 --out runs/sft/e4b-xact-map > logs/sft_e4b_xact_map.log 2>&1
deactivate
AD=$(ls -d runs/sft/e4b-xact-map/final 2>/dev/null || ls -d runs/sft/e4b-xact-map/checkpoint-* | tail -1)
echo "e4b xact map $AD $(grep -o "'eval_loss': '[0-9.a-z]*'" logs/sft_e4b_xact_map.log | tr '\n' ' ')" | tee -a $R
CUDA_VISIBLE_DEVICES=0 timeout 9000 $PY eval/run_student_vllm.py --base google/gemma-4-E4B-it --adapter $AD --extract --acts \
  --out runs/student/v2-e4b-xact --max-model-len 16384 > logs/eval_e4b_xact.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 10; rm -rf runs/merged/runs__sft__e4b-xact-map*
python3 - <<'P'
import json,glob,os,random,re
src,dst='runs/student/v2-e4b-xact','runs/student/nj25-e4b-xact'
os.makedirs(dst,exist_ok=True)
for f in glob.glob(f'{src}/ivod_*.json'):
    s=os.path.basename(f);r=json.load(open(f));rng=random.Random(s+'25')
    ns=rng.sample(r['notes'],min(25,len(r['notes'])))
    prose=''.join(re.sub(r'^[^:：]{1,15}[:：]\s*','',n['text']).rstrip('。')+f" [{n['ts']}]。" for n in ns)
    json.dump({'notes':r['notes'],'prose':prose},open(f'{dst}/{s}','w'),ensure_ascii=False)
P
python3 eval/score_v2.py runs/student/v2-e4b-xact runs/student/v2-minicpm5-xact | tee -a $R
JUDGE_SCRIPT=judge_prose_tx DIRS="runs/student/nj25-e4b-xact" bash scripts/v2_judge.sh >> logs/v2_judge_e4bx.log 2>&1
grep sentences reports/v2_sft_night.txt | tail -1 | tee -a $R
echo E4BX DONE | tee -a $R
