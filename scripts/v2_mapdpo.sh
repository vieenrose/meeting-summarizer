#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
until grep -q "ROUND5 DONE" reports/v2_round5.txt 2>/dev/null; do sleep 60; done
JUDGE_SCRIPT=judge_notes_tx DIRS=data/train/rft_samples.jsonl bash scripts/v2_judge.sh > logs/v2_judge_mapnotes.log 2>&1
echo MAPNOTES_JUDGED >> reports/v2_mapdpo.txt
python3 -m distill.map_dpo_pairs | tee -a reports/v2_mapdpo.txt
PY=~/vllm019/bin/python
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True $PY distill/dpo_notes.py --pairs data/train/map_dpo_pairs.jsonl \
  --epochs 2 --out runs/dpo/minicpm5-map-judged > logs/dpo_map_judged.log 2>&1
grep "preference pairs" logs/dpo_map_judged.log | tee -a reports/v2_mapdpo.txt
grep -o "'rewards/accuracies': '[0-9.]*'" logs/dpo_map_judged.log | tail -1 | tee -a reports/v2_mapdpo.txt
CUDA_VISIBLE_DEVICES=0 timeout 7200 $PY eval/run_student_vllm.py --base runs/merged/runs__sft__minicpm5-v2-ivod-ali__final \
  --adapter runs/dpo/minicpm5-map-judged/final --out runs/student/v2-minicpm5-mapdpo --max-model-len 16384 > logs/eval_mapdpo.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 10
python3 - <<'P'
import json,glob,os,random,re
src,dst='runs/student/v2-minicpm5-mapdpo','runs/student/nj25-m-mapdpo'
os.makedirs(dst,exist_ok=True)
for f in glob.glob(f'{src}/ivod_*.json'):
    s=os.path.basename(f);r=json.load(open(f));rng=random.Random(s+'25')
    ns=rng.sample(r['notes'],min(25,len(r['notes'])))
    prose=''.join(re.sub(r'^[^:：]{1,15}[:：]\s*','',n['text']).rstrip('。')+f" [{n['ts']}]。" for n in ns)
    json.dump({'notes':r['notes'],'prose':prose},open(f'{dst}/{s}','w'),ensure_ascii=False)
P
python3 eval/score_v2.py runs/student/v2-minicpm5-mapdpo | tee -a reports/v2_mapdpo.txt
JUDGE_SCRIPT=judge_prose_tx DIRS="runs/student/nj25-m-mapdpo runs/student/v2-minicpm5-mapdpo" bash scripts/v2_judge.sh >> logs/v2_judge_mapdpo.log 2>&1
grep sentences reports/v2_sft_night.txt | tail -2 | tee -a reports/v2_mapdpo.txt
echo MAPDPO DONE | tee -a reports/v2_mapdpo.txt
