# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Run a list of Isaac Lab jobs with a fixed number of slots per GPU (no Isaac Sim import here).

Jobs are JSON lines: ``{"name": ..., "args": ["script.py", "--flag", ...], "out": "file that marks success"}``.
A job whose ``out`` file already exists is skipped, so an interrupted sweep can simply be restarted. A new job only
starts while enough RAM is available, because every Isaac Sim process needs about 4 GB.

    python assignments/hw1_ant/scripts/run_queue.py jobs.jsonl --gpus 0,1 --per_gpu 2
"""

import argparse
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def available_ram_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1024**2
    return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", help="JSON-lines file of jobs.")
    parser.add_argument("--gpus", default="0,1", help="Comma-separated GPU indices.")
    parser.add_argument("--per_gpu", type=int, default=2, help="Concurrent jobs per GPU.")
    parser.add_argument("--min_ram_gb", type=float, default=6.0, help="Free RAM required to start a job.")
    parser.add_argument("--log_dir", default=os.path.join(REPO_ROOT, "logs", "hw1_runlogs"))
    args = parser.parse_args()

    with open(args.jobs) as f:
        jobs = [json.loads(line) for line in f if line.strip()]
    pending = [j for j in jobs if not (j.get("out") and os.path.exists(os.path.join(REPO_ROOT, j["out"])))]
    print(f"[QUEUE] {len(jobs)} jobs, {len(jobs) - len(pending)} already done, {len(pending)} to run")
    slots = {int(g): [] for g in args.gpus.split(",")}
    failed = []
    while pending or any(slots.values()):
        for gpu, running in slots.items():
            for job, proc in list(running):
                if proc.poll() is not None:
                    running.remove((job, proc))
                    ok = proc.returncode == 0 and (not job.get("out") or os.path.exists(os.path.join(REPO_ROOT, job["out"])))
                    if not ok:
                        failed.append(job["name"])
                    print(f"[QUEUE] {'done' if ok else 'FAILED'} {job['name']} (gpu {gpu}), {len(pending)} pending")
            while pending and len(running) < args.per_gpu and available_ram_gb() >= args.min_ram_gb:
                job = pending.pop(0)
                log = os.path.join(args.log_dir, f"{job['name']}.log")
                cmd = ["bash", os.path.join(HERE, "isaac_run.sh"), log, *job["args"]]
                proc = subprocess.Popen(cmd, env={**os.environ, "GPU": str(gpu)}, cwd=REPO_ROOT)
                running.append((job, proc))
                print(f"[QUEUE] start {job['name']} on gpu {gpu}")
                time.sleep(20)  # let the process allocate its memory before checking RAM again
        time.sleep(5)
    print(f"[QUEUE] finished, {len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
