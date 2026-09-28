#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
until grep -q "INTERP DONE" reports/v2_interp.txt 2>/dev/null; do sleep 60; done
sleep 120
CUDA_VISIBLE_DEVICES=1 ~/.venvs/vllm/bin/python eval/cad_map.py --merged runs/merged/runs__sft__minicpm5-v2-ivod-ali__final \
  --run-dir runs/student/v2-minicpm5 --sessions 15 --beta 0.5 --out runs/student/v2-minicpm5-cad > logs/cad_map.log 2>&1
python3 - <<'P'
import json,glob,os,random,re
cad=sorted(os.path.basename(f) for f in glob.glob('runs/student/v2-minicpm5-cad/ivod_*.json'))
for src,dst in [('runs/student/v2-minicpm5-cad','runs/student/nj25-cad'),('runs/student/v2-minicpm5','runs/student/nj25-cadref')]:
    os.makedirs(dst,exist_ok=True)
    for s in cad:
        r=json.load(open(f'{src}/{s}'));rng=random.Random(s+'25')
        ns=rng.sample(r['notes'],min(25,len(r['notes'])))
        prose=''.join(re.sub(r'^[^:：]{1,15}[:：]\s*','',n['text']).rstrip('。')+f" [{n['ts']}]。" for n in ns)
        json.dump({'notes':r['notes'],'prose':prose},open(f'{dst}/{s}','w'),ensure_ascii=False)
P
echo CAD_GEN_DONE >> reports/v2_cad.txt
