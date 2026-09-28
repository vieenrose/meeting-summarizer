#!/usr/bin/env bash
# CPU-only memory and speed probe for the candidate student GGUFs.
#
# This box is x86, so its tok/s says nothing about a Dimensity 900 and is recorded only as a
# relative figure. Peak RSS does transfer: weights plus KV plus compute buffers are the same bytes
# on either architecture, so a model that blows the 2.5 GB ceiling here will blow it on the phone.
# The phone still has to be measured for speed, thermals and the real ceiling.
set -uo pipefail
ROOT=/home/luigi/meeting-summarizer
HUB=/home/luigi/.cache/huggingface/hub
BENCH=/home/luigi/llama.cpp/build-cuda/bin/llama-bench
CLI=/home/luigi/llama.cpp/build/bin/llama-cli
cd $ROOT
OUT=reports/cpu_probe.txt
: > $OUT
say() { echo "$*" | tee -a $OUT; }

find_gguf() { find -L "$HUB" -name "$1" -type f 2>/dev/null | head -1; }

for spec in "MiniCPM5-2B-Q4_K_M.gguf MiniCPM5-2B-Q4_K_M" \
            "gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf Gemma-4-E2B-QAT-Q2_K_XL"; do
  set -- $spec
  file=$(find_gguf "$1"); name=$2
  if [ -z "$file" ]; then say "$name: MISSING"; continue; fi
  say "=== $name  ($(du -hL "$file" | cut -f1) on disk)"

  # Peak RSS at the context the phone will actually use, with a prompt that fills a window.
  python3 - "$file" "$name" <<'PY' | tee -a $OUT
import json, resource, subprocess, sys, time, os
gguf, name = sys.argv[1], sys.argv[2]
prompt = ("[0:00] S1: 今天審查能源管理法部分條文修正草案。" * 120)
cmd = ["/home/luigi/llama.cpp/build/bin/llama-cli", "-m", gguf, "-c", "8192", "-n", "64",
       "-t", "4", "-ngl", "0", "--no-warmup", "-st", "-p", prompt]
t0 = time.time()
proc = subprocess.run(cmd, capture_output=True, text=True)
elapsed = time.time() - t0
peak_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
print(f"  peak RSS {peak_kb/1024/1024:.2f} GB at 8k context, {elapsed:.0f}s for 64 tokens "
      f"(rc={proc.returncode})")
if proc.returncode != 0:
    print("  stderr tail:", proc.stderr.strip().splitlines()[-2:])
PY

  # Prompt-processing speed by depth: the shape that decides whether 20k tokens fits the budget.
  say "  llama-bench (CPU, 4 threads):"
  say "    (llama-bench unavailable in this build; RSS above is the measurement that matters)"
done
say ""
say "Reminder: tok/s here is x86 and does not transfer to the Reno7; peak RSS does."
