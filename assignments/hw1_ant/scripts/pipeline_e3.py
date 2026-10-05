# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Run every training of the third addendum (E3) and then the evaluation sweep (no Isaac Sim import here).

Trainings (see lab notes ``00d_plan_addendum_E3.md``), all resume ``E2_seed<S>/model_2998.pt`` for 1500 iterations:

* ``E3_seed{42,43,44}``: stage 3 on the DR-Harder terrain (submission candidate).
* ``Sobs_seed42``: only the stage-2 blocks (obstacle specialist; diagnosis, never submitted).
* ``E3c_seed{42,43,44}``: the stage-2 terrain once more (control for the amount of training; also a candidate).

At most four trainings run at once (RAM), each only when enough memory is free. Once every training has started and
the E3 and specialist runs are done, the evaluation queue starts in the background; after the last training it runs
once more for the remaining jobs. The file ``logs/hw1_runlogs/pipeline_e3.done`` marks the end.

    nohup python assignments/hw1_ant/scripts/pipeline_e3.py > logs/hw1_runlogs/pipeline_e3.log 2>&1 &
"""

import json
import os
import subprocess
import sys
import time

import hw1_common as common
from pipeline_e2 import LOGS, MIN_RAM_GB, Job, available_ram_gb, log

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 43, 44)
STAGE2_LAST = 2998
FINAL = f"model_{STAGE2_LAST + 1500 - 1}.pt"
MAX_TRAININGS = 4
TASK = {
    "E3": "Isaac-Ant-WideScan-DRHarder-v0",
    "Sobs": "Isaac-Ant-WideScan-ObstHard-v0",
    "E3c": "Isaac-Ant-WideScan-DRHard-v0",
}


def resume_args(cond: str, seed: int) -> list[str]:
    """Training that continues the last stage-2 checkpoint of ``E2_seed<seed>``."""
    parent_dir = os.path.basename(common.find_run(f"E2_seed{seed}"))
    return ["--task", TASK[cond], "--headless", "--seed", str(seed), "--run_name", f"{cond}_seed{seed}",
            "--resume", "--load_run", parent_dir, "--checkpoint", f"model_{STAGE2_LAST}.pt"]


def eval_queue(tag: str, wait: bool) -> subprocess.Popen | None:
    jobs_file = os.path.join(LOGS, f"eval_jobs_e3_{tag}.jsonl")
    py = sys.executable
    subprocess.run([py, os.path.join(HERE, "sweep_eval.py"), "jobs", "--out", jobs_file], cwd=common.REPO_ROOT, check=True)
    log(f"evaluation queue {tag}")
    cmd = [py, os.path.join(HERE, "run_queue.py"), jobs_file, "--gpus", "0,1", "--per_gpu", "2", "--min_ram_gb", "6"]
    env = {**os.environ, "NO_SLOT": "1"}
    if wait:
        subprocess.run(cmd, cwd=common.REPO_ROOT, env=env, check=False)
        return None
    out = open(os.path.join(LOGS, f"run_queue_e3_{tag}.log"), "w")
    return subprocess.Popen(cmd, cwd=common.REPO_ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)


def main():
    os.makedirs(LOGS, exist_ok=True)
    order = [("E3", s) for s in SEEDS] + [("Sobs", 42)] + [("E3c", s) for s in SEEDS]
    jobs = [Job(f"{cond}_seed{seed}", 0, resume_args(cond, seed), FINAL) for cond, seed in order]

    early_queue = None
    while True:
        for job in jobs:
            job.poll()
        running = [j for j in jobs if j.state == "running"]
        for job in jobs:
            if job.state != "waiting" or len(running) >= MAX_TRAININGS:
                continue
            if available_ram_gb() < MIN_RAM_GB:
                break
            load = {g: sum(1 for j in running if j.gpu == g) for g in (0, 1)}
            job.gpu = min(load, key=load.get)
            job.start()
            running.append(job)
            time.sleep(40)  # let the process allocate its memory before the next RAM check
        all_started = all(j.state != "waiting" for j in jobs)
        first = [j for j in jobs if not j.name.startswith("E3c")]
        if early_queue is None and all_started and all(j.state in ("done", "failed") for j in first):
            early_queue = eval_queue("main", wait=False)
        if all(j.state in ("done", "failed") for j in jobs):
            break
        time.sleep(20)

    if early_queue is not None:
        early_queue.wait()
    eval_queue("final", wait=True)
    summary = {j.name: j.state for j in jobs}
    with open(os.path.join(LOGS, "pipeline_e3.done"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"pipeline finished: {summary}")


if __name__ == "__main__":
    main()
