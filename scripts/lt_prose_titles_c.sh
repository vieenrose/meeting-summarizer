#!/usr/bin/env bash
# notes->prose and titles of a LiteRT-LM model (litert-lm serve :9379), same inputs and judge as mobile_v1b.sh
set -uo pipefail
cd /home/luigi/meeting-summarizer
tag=$1; MP=$2
python3 distill/convert_targets.py --journals runs/student/h38-ft-v5 --urls http://127.0.0.1:9379/v1 --model "$MP" --parallel 2 --max-journal-chars ${MAXJ:-0} --out runs/convert/h38-$tag > logs/conv_$tag.log 2>&1
python3 eval/judge_prose_notes.py --dir runs/convert/h38-$tag --judge-url http://127.0.0.1:8700/v1 --out reports/judge_prose_notes_h38-$tag.json 2>&1 | tail -1
python3 eval/title_eval.py --run runs/student/h38-ft-v5 --split data/split_rt_heldout38.json --gold runs/v2/w4000 \
  --titler "http://127.0.0.1:9379/v1:$MP" --judge http://127.0.0.1:8700/v1:judge --max-journal-chars ${MAXJ:-0} --tag h38/$tag --out reports/titles_h38_$tag.json 2>&1 | tail -1
