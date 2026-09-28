#!/usr/bin/env bash
cd /home/luigi/meeting-summarizer
until grep -q "Q35X DONE" reports/v2_q35_xact.txt 2>/dev/null; do sleep 120; done
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done; sleep 15
cd ~/FreeToken
for g in 0 1; do P=$([ $g = 0 ] && echo 2919 || echo 1919); PATH=/usr/local/cuda-13.4/bin:$PATH nohup .venv/bin/ft serve --model ~/models/Qwen3.8-Flash-Next-NVFP4 --gpu $g --port $P --served-model-name fn-local --text-model-only --kv-reserve-tokens 250000 > /home/luigi/meeting-summarizer/logs/ft_gpu$g.log 2>&1 & echo $! > /home/luigi/meeting-summarizer/logs/ft_gpu$g.pid; done
cd /home/luigi/meeting-summarizer
for P in 1919 2919; do until curl -s -m 30 localhost:$P/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"fn-local","messages":[{"role":"user","content":"hi"}],"max_tokens":3,"chat_template_kwargs":{"enable_thinking":false}}' | grep -q choices; do sleep 15; done; done
python3 eval/singlepass_prose.py --out runs/student/v2-singlepass-fn > logs/singlepass_fn.log 2>&1
for f in ft_gpu0 ft_gpu1; do kill $(cat logs/$f.pid); done; sleep 30
python3 eval/score_v2.py runs/student/v2-singlepass-fn | tee -a reports/v2_singlepass.txt
JUDGE_SCRIPT=judge_prose_tx DIRS=runs/student/v2-singlepass-fn bash scripts/v2_judge.sh >> logs/v2_judge_sp.log 2>&1
JUDGE_SCRIPT=judge_prose DIRS=runs/student/v2-singlepass-fn bash scripts/v2_judge.sh >> logs/v2_judge_sp.log 2>&1
grep -E "sentences|coverage" reports/v2_sft_night.txt | tail -2 | tee -a reports/v2_singlepass.txt
echo SP DONE | tee -a reports/v2_singlepass.txt
