#!/usr/bin/env bash
# Stage 0 teacher runs, started automatically once every session is transcribed:
# wait for the transcription watcher to finish, bring up the FP8+MTP teacher, then run the
# reading agent and the map-reduce baseline at 2k / 4k / 8k windows over all transcripts.
set -uo pipefail
ROOT=/home/luigi/meeting-summarizer
cd $ROOT
while pgrep -f "[t]ranscribe_watch.sh" >/dev/null || pgrep -f "[d]ata/fetch_audio.py" >/dev/null; do sleep 60; done
echo "$(date) transcripts: $(ls data/transcripts/*.txt 2>/dev/null | wc -l) of $(ls data/audio/*.m4a | wc -l) sessions"
# Do not start the teacher on a partial set: a missing transcript silently shrinks the eval.
if [ "$(ls data/transcripts/*.txt 2>/dev/null | wc -l)" -lt "$(ls data/audio/*.m4a | wc -l)" ]; then
  echo "transcription incomplete, not starting teacher"; exit 1
fi
# The GPUs must be free, or vLLM fails to allocate its KV cache.
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -lt 2000 ]; do sleep 30; done

nohup scripts/serve_teacher.sh > logs/teacher_server.log 2>&1 &
until curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null; do
  pgrep -f "[v]llm.entrypoints.openai.api_server" >/dev/null || { echo "teacher server exited"; tail -20 logs/teacher_server.log; exit 1; }
  sleep 10
done
echo "$(date) teacher ready"

for mode in agent mapreduce; do
  python3 eval/run_teacher.py --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.6-35B-A3B \
    --tokenizer Qwen/Qwen3.6-35B-A3B-FP8 --transcripts data/transcripts \
    --window-tokens 4000 2000 8000 --mode $mode --out runs/teacher --parallel 8
done
echo "$(date) stage0 teacher runs done"
