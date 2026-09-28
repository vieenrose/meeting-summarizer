#!/usr/bin/env bash
# Corpus v2 transcription: one VibeVoice worker per GPU, taking sessions as their audio lands.
#
# Same as data/transcribe_watch.sh with v2's directories, plus a GPU wait: GPU 0 may still be
# training, so its worker starts only when no GRPO/SFT process is using it. Waiting is on the
# python entry point's own command line, never a shell's (a shell can contain the name as text).
set -uo pipefail
ROOT=/home/luigi/meeting-summarizer
AUDIO=${AUDIO:-$ROOT/data/v2/audio}
OUT=${OUT:-$ROOT/data/v2/transcripts}
FETCH_PATTERN=${FETCH_PATTERN:-data/v2/pilot}   # matches the manifest and its download shards

gpu_busy_with_training() {   # true while a training process runs (they use GPU 0)
  for p in /proc/[0-9]*; do
    cmd=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
    case "$cmd" in
      /home/luigi/vllm019/bin/python\ distill/grpo_vllm.py*|/home/luigi/.venvs/vllm/bin/python\ distill/sft_gemma.py*) return 0 ;;
    esac
  done
  return 1
}

fetch_running() {
  for p in /proc/[0-9]*; do
    cmd=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
    case "$cmd" in python3\ data/fetch_audio.py*"$FETCH_PATTERN"*) return 0 ;; esac
  done
  return 1
}

worker() {
  local gpu=$1
  if [ "$gpu" = 0 ]; then
    while gpu_busy_with_training; do sleep 60; done
    echo "[gpu0] training finished, starting"
  fi
  cd /home/luigi/vibevoice
  while true; do
    local did=0
    for f in $AUDIO/ivod_*.m4a; do
      [ -e "$f" ] || continue
      case "$f" in *.part.m4a) continue ;; esac
      local stem=$(basename "$f" .m4a)
      [ -e "$OUT/$stem.chunks.json" ] && continue
      mkdir "$OUT/.lock_$stem" 2>/dev/null || continue
      CUDA_VISIBLE_DEVICES=$gpu PYTORCH_ALLOC_CONF=expandable_segments:True \
        /home/luigi/.venvs/vibevoice/bin/python $ROOT/data/transcribe_vibevoice.py \
        --files "$f" --out-dir "$OUT" --device cuda:0 2>&1 \
        | grep -vE "Warning|warn|tokenizer class|torch_dtype|Loading checkpoint|optimum" | sed "s/^/[gpu$gpu] /"
      rmdir "$OUT/.lock_$stem"
      did=1
    done
    if [ $did -eq 0 ]; then
      fetch_running || break
      sleep 30
    fi
  done
  echo "[gpu$gpu] worker done"
}
mkdir -p "$OUT"
# Clearing locks is right only when no other watcher is running; a second watcher started alongside
# a live one would otherwise remove the lock on a session still being transcribed.
[ "${CLEAR_LOCKS:-1}" = 1 ] && rmdir "$OUT"/.lock_* 2>/dev/null
# GPUS picks which cards transcribe, e.g. GPUS=1 leaves GPU 0 free for other work.
for g in ${GPUS:-0 1}; do worker "$g" & done
wait
echo "watcher done"
