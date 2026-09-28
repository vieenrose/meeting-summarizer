"""Live training monitor, served on the tailnet.

Read-only: every request re-reads the trainer logs, the held-out checkpoint reports and
nvidia-smi, so it needs no hook into training and cannot slow or break a run. Standard library only.

  GET /            the dashboard (monitor/index.html)
  GET /api/status  everything the dashboard draws, as JSON

Run:  python3 monitor/server.py --port 8322
Expose on the tailnet:  tailscale serve --bg --http=8322 http://127.0.0.1:8322
"""
import argparse
import ast
import glob
import json
import os
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Training runs, newest first. Each is a trainer log; a run's checkpoint evaluations live in a report.
RUNS = [
    {"id": "notes-run2", "title": "Notes GRPO · run 2", "stage": "notes",
     "log": "logs/grpo_notes_vllm_run2.log", "report": "reports/grpo_checkpoints_run2.txt",
     "report_filter": "notes-"},
    {"id": "summary-run2", "title": "Summary GRPO · run 2 (length-gated reward)", "stage": "summary",
     "log": "logs/grpo_g4it_vllm_run2.log", "report": "reports/grpo_checkpoints_run2.txt",
     "report_filter": "checkpoint-"},
    {"id": "summary-run1", "title": "Summary GRPO · run 1 (reward hacked after ~420)", "stage": "summary",
     "log": "logs/grpo_g4it_vllm.log", "report": "reports/grpo_checkpoints_run1.txt",
     "report_filter": "checkpoint-"},
]

DICT_RE = re.compile(r"\{'loss'.*?\}")
TQDM_RE = re.compile(r"(\d+)/(\d+) \[(\d[\d:]*)<(\d[\d:]*),\s*([\d.]+)s/it\]")
REPORT_RE = re.compile(r"^(?:(\d\d:\d\d) )?(.+?)\s+pass (\d+)/(\d+)\s+reward ([\d.]+)\s+zeros (\d+)\s+"
                       r"notes-recall (\d+)%\s+empty (\d+)")
KEEP = ("reward", "reward_std", "loss", "completions/mean_length", "completions/clipped_ratio",
        "frac_reward_zero_std", "entropy", "epoch")


def _secs(hms):
    parts = [int(p) for p in hms.split(":")]
    total = 0
    for p in parts:
        total = total * 60 + p
    return total


def read_log(path):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    with open(full, "rb") as fh:
        raw = fh.read().decode("utf-8", "replace")
    steps = []
    for m in DICT_RE.finditer(raw):
        try:
            d = ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            continue
        if "reward" not in d:
            continue
        row = {}
        for k in KEEP:
            try:
                row[k] = float(d[k])
            except (KeyError, TypeError, ValueError):
                pass
        steps.append(row)
    progress = None
    for m in TQDM_RE.finditer(raw.replace("\r", "\n")[-4000:]):
        progress = {"step": int(m.group(1)), "total": int(m.group(2)), "elapsed_s": _secs(m.group(3)),
                    "remaining_s": _secs(m.group(4)), "s_per_it": float(m.group(5))}
    finished = "train_runtime" in raw[-3000:]
    failed = "Traceback" in raw[-6000:]
    return {"steps": steps, "progress": progress, "finished": finished, "failed": failed,
            "updated": os.path.getmtime(full)}


def read_report(path, label_filter):
    full = os.path.join(ROOT, path)
    rows, start, picked, events = [], None, None, []
    if not os.path.exists(full):
        return rows, start, picked, events
    for line in open(full, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("picked "):
            picked = line
        m = REPORT_RE.search(line)
        if not m:
            if line.strip():
                events.append(line)
            continue
        n = int(m.group(4))
        row = {"time": m.group(1), "label": m.group(2), "pass": int(m.group(3)), "n": n,
               "reward": float(m.group(5)), "zeros": int(m.group(6)), "recall": int(m.group(7)),
               "empty": int(m.group(8))}
        if n == 0:
            continue                        # a failed evaluation, not a score
        if "(start)" in row["label"]:
            start = row
        elif row["label"].startswith(label_filter) or (label_filter == "checkpoint-" and row["label"] == "final"):
            rows.append(row)
    return rows, start, picked, events


def gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,"
                              "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    result = []
    for line in out.strip().splitlines():
        idx, name, used, total, util, temp, power = [x.strip() for x in line.split(",")]
        result.append({"index": int(idx), "name": name, "mem_used": int(used), "mem_total": int(total),
                       "util": int(util), "temp": int(temp), "power": float(power) if power else None})
    return result


def active_jobs():
    jobs = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        # Match the python entry points only, never shells: a shell's command line can contain these
        # names as text (the lesson that stalled the notes stage for three hours).
        if not cmd.startswith(("/home/luigi/vllm019/bin/python", "/home/luigi/.venvs/vllm/bin/python")):
            continue
        for script, kind in (("distill/grpo_vllm.py", "train"), ("eval/run_student_vllm.py", "eval"),
                             ("distill/sft_gemma.py", "sft")):
            if script in cmd and "accelerate" not in cmd:
                task = "notes" if "--task notes" in cmd else ("summary" if kind == "train" else "")
                m = re.search(r"--adapter (\S+)", cmd)
                jobs.append({"pid": int(pid), "kind": kind, "task": task, "adapter": m.group(1) if m else None})
    return jobs


def status():
    runs = []
    for spec in RUNS:
        log = read_log(spec["log"])
        if log is None:
            continue
        rows, start, picked, events = read_report(spec["report"], spec["report_filter"])
        runs.append({**{k: spec[k] for k in ("id", "title", "stage")}, **log,
                     "checkpoints": rows, "start": start, "picked": picked})
    # A notes run starts from the summary checkpoint picked for it, not from SFT; compare against that
    # or a regression below the true starting point reads as a gain over the SFT baseline.
    for run in runs:
        if run["stage"] == "notes" and run.get("picked"):
            m = re.search(r"picked (\S+)", run["picked"])
            source = next((r for r in runs if r["stage"] == "summary" and r["id"].endswith(run["id"].split("-")[-1])), None)
            row = next((c for c in (source or {}).get("checkpoints", []) if m and c["label"] == m.group(1)), None)
            if row:
                run["start"] = row | {"label": f"{row['label']} (start)"}
    _, _, _, events = read_report(RUNS[0]["report"], "")
    return {"now": time.time(), "runs": runs, "gpus": gpus(), "jobs": active_jobs(),
            "events": events[-8:]}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/status"):
            body = json.dumps(status()).encode()
            ctype = "application/json"
        elif self.path in ("/", "/index.html"):
            body = open(os.path.join(os.path.dirname(__file__), "index.html"), "rb").read()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8322)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
