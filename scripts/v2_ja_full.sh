#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
python3 eval/journal_agent.py --urls http://127.0.0.1:8090/v1 --parallel 2 --out runs/student/ja-bonsai2-off > logs/ja_off.log 2>&1 &
python3 eval/journal_agent.py --urls http://127.0.0.1:8091/v1 --parallel 2 --think 768 --out runs/student/ja-bonsai2-think > logs/ja_think.log 2>&1 &
wait
for f in bonsai_off bonsai_think; do kill $(cat logs/$f.pid); done; sleep 20
python3 - <<'P'
import json,glob,os,random,re
for m in ('off','think'):
    src,dst=f'runs/student/ja-bonsai2-{m}',f'runs/student/nj25-ja-{m}'
    os.makedirs(dst,exist_ok=True)
    for f in glob.glob(f'{src}/ivod_*.json'):
        s=os.path.basename(f);r=json.load(open(f));rng=random.Random(s+'25')
        ns=rng.sample(r['notes'],min(25,len(r['notes'])))
        prose=''.join(re.sub(r'^[^:：]{1,15}[:：]\s*','',n['text']).rstrip('。')+f" [{n['ts']}]。" for n in ns)
        json.dump({'notes':r['notes'],'prose':prose},open(f'{dst}/{s}','w'),ensure_ascii=False)
P
D="runs/student/nj25-ja-off runs/student/nj25-ja-think runs/student/ja-bonsai2-off runs/student/ja-bonsai2-off-draft runs/student/ja-bonsai2-think runs/student/ja-bonsai2-think-draft"
JUDGE_SCRIPT=judge_prose_tx DIRS="$D" bash scripts/v2_judge.sh >> logs/v2_judge_ja.log 2>&1
JUDGE_SCRIPT=judge_prose DIRS="runs/student/ja-bonsai2-off runs/student/ja-bonsai2-think" bash scripts/v2_judge.sh >> logs/v2_judge_ja.log 2>&1
grep -E "sentences|coverage" reports/v2_sft_night.txt | tail -8 | tee -a reports/v2_ja.txt
echo JA DONE | tee -a reports/v2_ja.txt
