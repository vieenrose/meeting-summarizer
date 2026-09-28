#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python; R=reports/v2_interp.txt
until grep -q "EXTRACT DONE" reports/v2_extract.txt 2>/dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 timeout 9000 $PY eval/run_student_vllm.py --base runs/merged/interp-ev-a0.5 --evidence \
  --out runs/student/v2-interp-a05 --max-model-len 16384 > logs/eval_interp05.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 timeout 9000 $PY eval/run_student_vllm.py --base runs/merged/interp-ev-a0.75 --evidence \
  --out runs/student/v2-interp-a075 --max-model-len 16384 > logs/eval_interp075.log 2>&1 &
wait
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 10
python3 - <<'P'
import json,glob,os,random,re
for a in ('05','075'):
    src,dst=f'runs/student/v2-interp-a{a}',f'runs/student/nj25-interp-a{a}'
    os.makedirs(dst,exist_ok=True)
    for f in glob.glob(f'{src}/ivod_*.json'):
        s=os.path.basename(f);r=json.load(open(f));rng=random.Random(s+'25')
        ns=rng.sample(r['notes'],min(25,len(r['notes'])))
        prose=''.join(re.sub(r'^[^:：]{1,15}[:：]\s*','',n['text']).rstrip('。')+f" [{n['ts']}]。" for n in ns)
        json.dump({'notes':r['notes'],'prose':prose},open(f'{dst}/{s}','w'),ensure_ascii=False)
P
python3 eval/score_v2.py runs/student/v2-interp-a05 runs/student/v2-interp-a075 runs/student/v2-minicpm5-evidence runs/v2/w4000 | tee -a $R
JUDGE_SCRIPT=judge_prose_tx DIRS="runs/student/nj25-interp-a05 runs/student/nj25-interp-a075" bash scripts/v2_judge.sh >> logs/v2_judge_interp.log 2>&1
grep sentences reports/v2_sft_night.txt | tail -2 | tee -a $R
echo INTERP DONE | tee -a $R
