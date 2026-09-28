#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python; R=reports/v2_round5.txt
while kill -0 $(cat logs/v2_judge_nj25.pid) 2>/dev/null; do sleep 60; done
until grep -q "gemma-4-E4B-it" <(ls ~/.cache/huggingface/hub) && ! pgrep -f "hf download google/gemma-4-E4B" >/dev/null; do sleep 60; done
source ~/.venvs/vllm/bin/activate
( CUDA_VISIBLE_DEVICES=0 python3 distill/sft_gemma.py --base google/gemma-4-E2B-it --rows data/train/evidence_rows.jsonl \
    --synth-repeat 3 --epochs 2 --out runs/sft/gemma-evidence > logs/sft_gemma_evidence.log 2>&1
  CUDA_VISIBLE_DEVICES=0 timeout 9000 $PY eval/run_student_vllm.py --adapter runs/sft/gemma-evidence/final --evidence \
    --out runs/student/v2-gemma-evidence --max-model-len 16384 > logs/eval_gemma_evidence.log 2>&1
  for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done
  echo GEMMA_EV_DONE | tee -a $R ) &
( CUDA_VISIBLE_DEVICES=1 python3 distill/sft_gemma.py --base google/gemma-4-E4B-it --rows data/train/reduce_only_rows.jsonl \
    --synth-repeat 1 --epochs 2 --out runs/sft/e4b-reduce > logs/sft_e4b_reduce.log 2>&1
  CUDA_VISIBLE_DEVICES=1 $PY eval/prose_from_gold.py --base google/gemma-4-E4B-it --adapter runs/sft/e4b-reduce/final \
    --notes-dir runs/student/v2-minicpm5-evidence --out runs/student/v2-e4b-reduce-on-mev > logs/pfg_e4b.log 2>&1
  for p in $(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done
  echo E4B_DONE | tee -a $R ) &
wait
deactivate
echo "e4b eval_loss $(grep -o "'eval_loss': '[0-9.a-z]*'" logs/sft_e4b_reduce.log | tr '\n' ' ')" | tee -a $R
python3 - <<'P'
import json,glob,os,random,re
src,dst='runs/student/v2-gemma-evidence','runs/student/nj25-g-ev'
os.makedirs(dst,exist_ok=True)
for f in glob.glob(f'{src}/ivod_*.json'):
    s=os.path.basename(f);r=json.load(open(f));rng=random.Random(s+'25')
    ns=rng.sample(r['notes'],min(25,len(r['notes'])))
    prose=''.join(re.sub(r'^[^:：]{1,15}[:：]\s*','',n['text']).rstrip('。')+f" [{n['ts']}]。" for n in ns)
    json.dump({'notes':r['notes'],'prose':prose},open(f'{dst}/{s}','w'),ensure_ascii=False)
P
python3 eval/score_v2.py runs/student/v2-gemma-evidence runs/student/v2-minicpm5-evidence | tee -a $R
JUDGE_SCRIPT=judge_prose_tx DIRS="runs/student/nj25-g-ev runs/student/v2-e4b-reduce-on-mev runs/student/v2-gemma-evidence" bash scripts/v2_judge.sh >> logs/v2_judge_r5.log 2>&1
grep sentences reports/v2_sft_night.txt | tail -3 | tee -a $R
echo ROUND5 DONE | tee -a $R
