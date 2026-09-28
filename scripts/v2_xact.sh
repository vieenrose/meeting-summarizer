#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
PY=~/vllm019/bin/python; R=reports/v2_xact.txt
source ~/.venvs/vllm/bin/activate
( CUDA_VISIBLE_DEVICES=0 python3 distill/sft_gemma.py --base openbmb/MiniCPM5-2B --rows data/train/extract_act_map_rows.jsonl \
    --epochs 2 --out runs/sft/minicpm5-xact > logs/sft_xact.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=1 python3 distill/sft_gemma.py --base google/gemma-4-E4B-it --rows data/train/extract_act_reduce_rows.jsonl \
    --epochs 3 --out runs/sft/e4b-xact-reduce > logs/sft_e4b_xact.log 2>&1 ) &
wait; deactivate
for f in sft_xact sft_e4b_xact; do echo "$f $(grep -o "'eval_loss': '[0-9.a-z]*'" logs/$f.log | tr '\n' ' ')" | tee -a $R; done
MAP=$(ls -d runs/sft/minicpm5-xact/final 2>/dev/null || ls -d runs/sft/minicpm5-xact/checkpoint-* | tail -1)
RED=$(ls -d runs/sft/e4b-xact-reduce/final 2>/dev/null || ls -d runs/sft/e4b-xact-reduce/checkpoint-* | tail -1)
CUDA_VISIBLE_DEVICES=0 timeout 9000 $PY eval/run_student_vllm.py --base openbmb/MiniCPM5-2B --adapter $MAP --extract --acts \
  --out runs/student/v2-minicpm5-xact --max-model-len 16384 > logs/eval_xact.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 10; rm -rf runs/merged/runs__sft__minicpm5-xact*
CUDA_VISIBLE_DEVICES=0 $PY eval/prose_from_gold.py --base google/gemma-4-E4B-it --adapter $RED \
  --notes-dir runs/student/v2-minicpm5-xact --out runs/student/v2-xact-e4b > logs/pfg_xact.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 10; rm -rf runs/merged/runs__sft__e4b*
python3 eval/prose_guard.py --run-dir runs/student/v2-xact-e4b --out runs/student/v2-xact-e4b-guard | tee -a $R
python3 eval/score_v2.py runs/student/v2-xact-e4b | tee -a $R
python3 - <<'P'
import json,glob,os,random,re
src,dst='runs/student/v2-minicpm5-xact','runs/student/nj25-m-xact'
os.makedirs(dst,exist_ok=True)
for f in glob.glob(f'{src}/ivod_*.json'):
    s=os.path.basename(f);r=json.load(open(f));rng=random.Random(s+'25')
    ns=rng.sample(r['notes'],min(25,len(r['notes'])))
    prose=''.join(re.sub(r'^[^:：]{1,15}[:：]\s*','',n['text']).rstrip('。')+f" [{n['ts']}]。" for n in ns)
    json.dump({'notes':r['notes'],'prose':prose},open(f'{dst}/{s}','w'),ensure_ascii=False)
P
JUDGE_SCRIPT=judge_prose_tx DIRS="runs/student/nj25-m-xact runs/student/v2-xact-e4b runs/student/v2-xact-e4b-guard" bash scripts/v2_judge.sh >> logs/v2_judge_xact.log 2>&1
JUDGE_SCRIPT=judge_prose DIRS="runs/student/v2-xact-e4b" bash scripts/v2_judge.sh >> logs/v2_judge_xact.log 2>&1
grep -E "sentences|coverage" reports/v2_sft_night.txt | tail -5 | tee -a $R
echo XACT DONE | tee -a $R
