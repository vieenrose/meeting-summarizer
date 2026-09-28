#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
until grep -q "BONSAI_CR DONE" reports/v2_bonsai_cr.txt 2>/dev/null; do sleep 120; done
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 15
B=~/Bonsai-demo/bin/cuda
# 32k per slot, as on the phone
for g in 0 1; do CUDA_VISIBLE_DEVICES=$g LD_LIBRARY_PATH=$B nohup $B/llama-server -m /home/luigi/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf -ngl 99 -c 65536 -np 2 --jinja   --port 809$g --alias bonsai > logs/bonsai_ja$g.log 2>&1 & echo $! > logs/bonsai_ja$g.pid; done
for g in 0 1; do until curl -sf localhost:809$g/health >/dev/null; do sleep 10; done; done
python3 eval/journal_agent.py --out runs/student/v2-bonsai-journal > logs/journal_agent.log 2>&1
tail -5 logs/journal_agent.log | tee -a reports/v2_journal.txt
for g in 0 1; do kill $(cat logs/bonsai_ja$g.pid); done; sleep 20
python3 - <<'P'
import json,glob,os,random,re
src,dst='runs/student/v2-bonsai-journal','runs/student/nj25-bonsai-journal'
os.makedirs(dst,exist_ok=True)
for f in glob.glob(f'{src}/ivod_*.json'):
    s=os.path.basename(f);r=json.load(open(f));rng=random.Random(s+'25')
    ns=rng.sample(r['notes'],min(25,len(r['notes'])))
    prose=''.join(re.sub(r'^[^:：]{1,15}[:：]\s*','',n['text']).rstrip('。')+f" [{n['ts']}]。" for n in ns)
    json.dump({'notes':r['notes'],'prose':prose},open(f'{dst}/{s}','w'),ensure_ascii=False)
P
D="runs/student/nj25-bonsai-journal runs/student/v2-bonsai-journal runs/student/v2-bonsai-journal-draft"
JUDGE_SCRIPT=judge_prose_tx DIRS="$D" bash scripts/v2_judge.sh >> logs/v2_judge_ja.log 2>&1
grep sentences reports/v2_sft_night.txt | tail -3 | tee -a reports/v2_journal.txt
echo JOURNAL DONE | tee -a reports/v2_journal.txt
