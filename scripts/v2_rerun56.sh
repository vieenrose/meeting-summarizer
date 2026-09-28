#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python
CUDA_VISIBLE_DEVICES=0 timeout 9000 $PY eval/run_student_vllm.py --adapter runs/sft/gemma-evidence/checkpoint-720 --evidence \
  --out runs/student/v2-gemma-evidence --max-model-len 16384 > logs/eval_gemma_evidence.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 timeout 9000 $PY eval/run_student_vllm.py --base runs/merged/runs__sft__minicpm5-v2-ivod-ali__final \
  --adapter runs/dpo/minicpm5-map-judged/final --out runs/student/v2-minicpm5-mapdpo --max-model-len 16384 > logs/eval_mapdpo.log 2>&1 &
wait
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 10
python3 - <<'P'
import json,glob,os,random,re
for src,dst in [('runs/student/v2-gemma-evidence','runs/student/nj25-g-ev'),('runs/student/v2-minicpm5-mapdpo','runs/student/nj25-m-mapdpo')]:
    os.makedirs(dst,exist_ok=True)
    for f in glob.glob(f'{src}/ivod_*.json'):
        s=os.path.basename(f);r=json.load(open(f));rng=random.Random(s+'25')
        ns=rng.sample(r['notes'],min(25,len(r['notes'])))
        prose=''.join(re.sub(r'^[^:：]{1,15}[:：]\s*','',n['text']).rstrip('。')+f" [{n['ts']}]。" for n in ns)
        json.dump({'notes':r['notes'],'prose':prose},open(f'{dst}/{s}','w'),ensure_ascii=False)
P
python3 eval/score_v2.py runs/student/v2-gemma-evidence runs/student/v2-minicpm5-mapdpo | tee -a reports/v2_rerun56.txt
JUDGE_SCRIPT=judge_prose_tx DIRS="runs/student/nj25-g-ev runs/student/nj25-m-mapdpo runs/student/v2-minicpm5-mapdpo" bash scripts/v2_judge.sh >> logs/v2_judge_rr.log 2>&1
grep sentences reports/v2_sft_night.txt | tail -3 | tee -a reports/v2_rerun56.txt
rm -rf runs/merged/runs__sft__gemma-evidence* runs/merged/runs__dpo__*
echo RERUN DONE | tee -a reports/v2_rerun56.txt
