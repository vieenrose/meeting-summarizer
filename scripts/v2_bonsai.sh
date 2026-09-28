#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
until grep -q "SP DONE" reports/v2_singlepass.txt 2>/dev/null; do sleep 120; done
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 15
B=~/Bonsai-demo/bin/cuda
for g in 0 1; do CUDA_VISIBLE_DEVICES=$g LD_LIBRARY_PATH=$B nohup $B/llama-server -m /home/luigi/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf -ngl 99 -c 65536 -np 4 --jinja   --port 809$g --alias bonsai > logs/bonsai_srv$g.log 2>&1 & echo $! > logs/bonsai_srv$g.pid; done
for g in 0 1; do until curl -sf localhost:809$g/health >/dev/null; do sleep 10; done; done
python3 eval/run_api_pipeline.py --urls http://127.0.0.1:8090/v1,http://127.0.0.1:8091/v1 --out runs/student/v2-bonsai-map > logs/bonsai_map.log 2>&1
tail -3 logs/bonsai_map.log | tee -a reports/v2_bonsai.txt
# reduce role: Bonsai prose from the 2B student's evidence-first notes
python3 - <<'P'
import json,glob,os,sys,requests
sys.path.insert(0,'.')
from distill.write_prose import write_one
from summarizer.ingest import parse_line
from concurrent.futures import ThreadPoolExecutor
os.makedirs('runs/student/v2-bonsai-reduce-on-mev',exist_ok=True)
U=['http://127.0.0.1:8090/v1','http://127.0.0.1:8091/v1']
def one(i_f):
    i,f=i_f
    from summarizer.pipeline import ChatClient
    chat=ChatClient(U[i%2],'bonsai',max_tokens=1500)
    r=json.load(open(f));s=os.path.basename(f)[:-5]
    lines=[parse_line(l) for l in open(f'data/v2/transcripts/{s}.txt').read().splitlines() if l.strip()]
    p,pr,_=write_one(chat,r,lines,turns=1)
    json.dump({'notes':r['notes'],'prose':p},open(f'runs/student/v2-bonsai-reduce-on-mev/{s}.json','w'),ensure_ascii=False)
list(ThreadPoolExecutor(8).map(one,enumerate(sorted(glob.glob('runs/student/v2-minicpm5-evidence/ivod_*.json')))))
P
for g in 0 1; do kill $(cat logs/bonsai_srv$g.pid); done; sleep 15
# single pass: one server with 131k context
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$B nohup $B/llama-server -m /home/luigi/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf -ngl 99 -c 131072 -np 1 --jinja --port 8090 --alias bonsai > logs/bonsai_srv_sp.log 2>&1 & echo $! > logs/bonsai_srv_sp.pid
until curl -sf localhost:8090/health >/dev/null; do sleep 10; done
python3 eval/singlepass_prose.py --urls http://127.0.0.1:8090/v1 --model bonsai --out runs/student/v2-singlepass-bonsai > logs/singlepass_bonsai.log 2>&1
kill $(cat logs/bonsai_srv_sp.pid); sleep 15
python3 - <<'P'
import json,glob,os,random,re
src,dst='runs/student/v2-bonsai-map','runs/student/nj25-bonsai-map'
os.makedirs(dst,exist_ok=True)
for f in glob.glob(f'{src}/ivod_*.json'):
    s=os.path.basename(f);r=json.load(open(f));rng=random.Random(s+'25')
    ns=rng.sample(r['notes'],min(25,len(r['notes'])))
    prose=''.join(re.sub(r'^[^:：]{1,15}[:：]\s*','',n['text']).rstrip('。')+f" [{n['ts']}]。" for n in ns)
    json.dump({'notes':r['notes'],'prose':prose},open(f'{dst}/{s}','w'),ensure_ascii=False)
P
python3 eval/score_v2.py runs/student/v2-bonsai-map | tee -a reports/v2_bonsai.txt
D="runs/student/nj25-bonsai-map runs/student/v2-bonsai-map runs/student/v2-bonsai-reduce-on-mev runs/student/v2-singlepass-bonsai"
JUDGE_SCRIPT=judge_prose_tx DIRS="$D" bash scripts/v2_judge.sh >> logs/v2_judge_bonsai.log 2>&1
JUDGE_SCRIPT=judge_prose DIRS="runs/student/v2-bonsai-map runs/student/v2-bonsai-reduce-on-mev runs/student/v2-singlepass-bonsai" bash scripts/v2_judge.sh >> logs/v2_judge_bonsai.log 2>&1
grep -E "sentences|coverage" reports/v2_sft_night.txt | tail -7 | tee -a reports/v2_bonsai.txt
echo BONSAI DONE | tee -a reports/v2_bonsai.txt
