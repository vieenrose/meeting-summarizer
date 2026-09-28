#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python; R=reports/v2_ta.txt
until grep -q "INTERP DONE" reports/v2_interp.txt 2>/dev/null; do sleep 60; done
source ~/.venvs/vllm/bin/activate
CUDA_VISIBLE_DEVICES=0 python3 distill/sft_gemma.py --base google/gemma-4-E4B-it --rows data/train/map_only_rows.jsonl \
  --epochs 2 --out runs/sft/e4b-map > logs/sft_e4b_map.log 2>&1
deactivate
AD=$(ls -d runs/sft/e4b-map/final 2>/dev/null || ls -d runs/sft/e4b-map/checkpoint-* | tail -1)
echo "e4b map $AD $(grep -o "'eval_loss': '[0-9.a-z]*'" logs/sft_e4b_map.log | tr '\n' ' ')" | tee -a $R
CUDA_VISIBLE_DEVICES=0 $PY eval/run_student_vllm.py --base google/gemma-4-E4B-it --adapter $AD --split data/split_rft_ivod.json \
  --transcripts data/v2/transcripts --out runs/student/ta-train-ivod --max-model-len 16384 > logs/ta_train_ivod.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY eval/run_student_vllm.py --base google/gemma-4-E4B-it --adapter $AD --split data/split_rft_ali.json \
  --transcripts data/alimeeting/transcripts --out runs/student/ta-train-ali --max-model-len 16384 > logs/ta_train_ali.log 2>&1 &
wait; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 10
CUDA_VISIBLE_DEVICES=0 $PY eval/run_student_vllm.py --base google/gemma-4-E4B-it --adapter $AD \
  --out runs/student/v2-e4b-map --max-model-len 16384 > logs/eval_e4b_map.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 10; rm -rf runs/merged/runs__sft__e4b-map*
python3 distill/ta_rows.py runs/student/ta-train-ivod runs/student/ta-train-ali data/train/ta_map_rows.jsonl | tee -a $R
python3 - <<'P'
import json
with open('data/train/ta_mix_rows.jsonl','w') as f:
    for l in open('data/train/ta_map_rows.jsonl'): f.write(l)
    for p in ['data/train/v2_gold_rows_all.jsonl','data/train/alimeeting_gold_rows.jsonl']:
        for l in open(p):
            if json.loads(l)['kind']!='notes': f.write(l)
P
source ~/.venvs/vllm/bin/activate
CUDA_VISIBLE_DEVICES=0 python3 distill/sft_gemma.py --base openbmb/MiniCPM5-2B --rows data/train/ta_mix_rows.jsonl \
  --synth-repeat 3 --epochs 2 --out runs/sft/minicpm5-ta > logs/sft_minicpm5_ta.log 2>&1
deactivate
AD2=$(ls -d runs/sft/minicpm5-ta/final 2>/dev/null || ls -d runs/sft/minicpm5-ta/checkpoint-* | tail -1)
CUDA_VISIBLE_DEVICES=0 $PY eval/run_student_vllm.py --base openbmb/MiniCPM5-2B --adapter $AD2 \
  --out runs/student/v2-minicpm5-ta --max-model-len 16384 > logs/eval_minicpm5_ta.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 10; rm -rf runs/merged/runs__sft__minicpm5-ta*
python3 - <<'P'
import json,glob,os,random,re
for src,dst in [('runs/student/v2-e4b-map','runs/student/nj25-e4b-map'),('runs/student/v2-minicpm5-ta','runs/student/nj25-m-ta')]:
    os.makedirs(dst,exist_ok=True)
    for f in glob.glob(f'{src}/ivod_*.json'):
        s=os.path.basename(f);r=json.load(open(f));rng=random.Random(s+'25')
        ns=rng.sample(r['notes'],min(25,len(r['notes'])))
        prose=''.join(re.sub(r'^[^:：]{1,15}[:：]\s*','',n['text']).rstrip('。')+f" [{n['ts']}]。" for n in ns)
        json.dump({'notes':r['notes'],'prose':prose},open(f'{dst}/{s}','w'),ensure_ascii=False)
P
python3 eval/score_v2.py runs/student/v2-e4b-map runs/student/v2-minicpm5-ta | tee -a $R
JUDGE_SCRIPT=judge_prose_tx DIRS="runs/student/nj25-e4b-map runs/student/nj25-m-ta" bash scripts/v2_judge.sh >> logs/v2_judge_ta.log 2>&1
grep sentences reports/v2_sft_night.txt | tail -2 | tee -a $R
echo TA DONE | tee -a $R
