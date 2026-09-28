#!/usr/bin/env bash
# Status line every 30 minutes until the teacher evaluation is done.
cd /home/luigi/meeting-summarizer
while true; do
  jobs_alive=$(pgrep -fc 'run_singlepass|run_teacher' || echo 0)
  echo "REPORT $(date '+%F %H:%M')  jobs=$jobs_alive"
  python3 eval/bakeoff_table.py 2>/dev/null | head -18
  [ "$jobs_alive" -eq 0 ] && { echo "TEACHER EVAL COMPLETE"; break; }
  sleep 1800
done
