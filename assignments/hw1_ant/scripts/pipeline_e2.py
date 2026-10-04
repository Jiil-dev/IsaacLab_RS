# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Run every training of the second addendum (E2) and then the evaluation sweep (no Isaac Sim import here).

Trainings (see lab notes ``00c_plan_addendum_E2.md``):

* ``E2s1_seed{42,43,44}``: E2 stage 1, 1500 iterations on the DR terrain.
* ``E2_seed{42,43,44}``: E2 stage 2, resumes stage 1 for 1500 iterations on the DR-Hard terrain.
* ``Eent_seed42``: E with the entropy bonus, 1000 iterations (diagnosis).
* ``Oracle_seed42``: resumes ``E2s1_seed42`` on the shapes of T1/T4 (ceiling reference, never submitted).
* ``E2c_seed42``: resumes ``E2s1_seed42`` on the stage-1 terrain (diagnosis: effect of the stage-2 terrain).

At most four trainings run at once (RAM), each only when enough memory is free. A stage-2 run starts as soon as its
stage 1 has finished. Once the E2 and oracle runs are done, the evaluation queue runs; after the last diagnosis run
it runs once more for the remaining jobs. The file ``logs/hw1_runlogs/pipeline_e2.done`` marks the end.

    nohup python assignments/hw1_ant/scripts/pipeline_e2.py > logs/hw1_runlogs/pipeline_e2.log 2>&1 &
"""

import json
import os
import subprocess
import sys
import time

import hw1_common as common

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(common.REPO_ROOT, "logs", "hw1_runlogs")
TRAIN = "scripts/reinforcement_learning/rsl_rl/train.py"
SEEDS = (42, 43, 44)
STAGE1_LAST = 1499
MAX_TRAININGS = 4
MIN_RAM_GB = 7.0


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def available_ram_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1024**2
    return 0.0


class Job:
    def __init__(self, name: str, gpu: int, args: list[str], final: str, needs: str | None = None):
        self.name, self.gpu, self.args, self.final, self.needs = name, gpu, args, final, needs
        self.proc: subprocess.Popen | None = None
        self.state = "waiting"  # waiting -> running -> done | failed

    def start(self):
        log_file = os.path.join(LOGS, f"train_{self.name}.log")
        env = {**os.environ, "GPU": str(self.gpu), "NO_SLOT": "1"}
        cmd = ["bash", os.path.join(HERE, "isaac_run.sh"), log_file, TRAIN, *self.args]
        self.proc = subprocess.Popen(cmd, env=env, cwd=common.REPO_ROOT)
        self.state = "running"
        log(f"start {self.name} on gpu {self.gpu}: {' '.join(self.args)}")

    def poll(self):
        if self.state == "running" and self.proc.poll() is not None:
            try:
                ok = self.proc.returncode == 0 and os.path.exists(os.path.join(common.find_run(self.name), self.final))
            except FileNotFoundError:
                ok = False
            self.state = "done" if ok else "failed"
            log(f"{self.state} {self.name} (exit {self.proc.returncode})")


def resume_args(task: str, seed: int, name: str, parent: str) -> list[str]:
    """Training that continues the last stage-1 checkpoint of ``parent`` (a run name)."""
    parent_dir = os.path.basename(common.find_run(parent))
    return ["--task", task, "--headless", "--seed", str(seed), "--run_name", name,
            "--resume", "--load_run", parent_dir, "--checkpoint", f"model_{STAGE1_LAST}.pt"]


def run_eval_queue(tag: str):
    jobs_file = os.path.join(LOGS, f"eval_jobs_e2_{tag}.jsonl")
    py = sys.executable
    subprocess.run([py, os.path.join(HERE, "sweep_eval.py"), "jobs", "--out", jobs_file], cwd=common.REPO_ROOT, check=True)
    log(f"evaluation queue {tag}")
    subprocess.run(
        [py, os.path.join(HERE, "run_queue.py"), jobs_file, "--gpus", "0,1", "--per_gpu", "2", "--min_ram_gb", "6"],
        cwd=common.REPO_ROOT, env={**os.environ, "NO_SLOT": "1"}, check=False,
    )


def main():
    os.makedirs(LOGS, exist_ok=True)
    gpu_of = {42: 0, 43: 0, 44: 1}
    jobs = []
    for seed in SEEDS:
        jobs.append(Job(f"E2s1_seed{seed}", gpu_of[seed],
                        ["--task", "Isaac-Ant-WideScan-DR-v0", "--headless", "--seed", str(seed),
                         "--run_name", f"E2s1_seed{seed}"], f"model_{STAGE1_LAST}.pt"))
    jobs.append(Job("Eent_seed42", 1,
                    ["--task", "Isaac-Ant-Scan-DR-v0", "--headless", "--seed", "42", "--run_name", "Eent_seed42",
                     "--max_iterations", "1000", "agent.algorithm.entropy_coef=0.005"], "model_999.pt"))
    for seed in SEEDS:
        jobs.append(Job(f"E2_seed{seed}", gpu_of[seed], [], "model_2998.pt", needs=f"E2s1_seed{seed}"))
    jobs.append(Job("Oracle_seed42", 1, [], "model_2998.pt", needs="E2s1_seed42"))
    jobs.append(Job("E2c_seed42", 0, [], "model_2998.pt", needs="E2s1_seed42"))
    by_name = {j.name: j for j in jobs}
    resume_task = {"E2": "Isaac-Ant-WideScan-DRHard-v0", "Oracle": "Isaac-Ant-WideScan-Oracle-v0",
                   "E2c": "Isaac-Ant-WideScan-DR-v0"}

    first_eval_done = False
    while True:
        for job in jobs:
            job.poll()
        running = [j for j in jobs if j.state == "running"]
        for job in jobs:
            if job.state != "waiting" or len(running) >= MAX_TRAININGS:
                continue
            if job.needs:
                parent = by_name[job.needs]
                if parent.state == "failed":
                    job.state = "failed"
                    log(f"skip {job.name}: {job.needs} failed")
                    continue
                if parent.state != "done":
                    continue
                if not job.args:
                    seed = int(job.name.rsplit("seed", 1)[1])
                    job.args = resume_args(resume_task[job.name.split("_")[0]], seed, job.name, job.needs)
            if available_ram_gb() < MIN_RAM_GB:
                break
            # the GPU with fewer running trainings, unless the job is the stage 2 of a seed (keep its GPU)
            if job.name.startswith(("Oracle", "E2c")):
                load = {g: sum(1 for j in running if j.gpu == g) for g in (0, 1)}
                job.gpu = min(load, key=load.get)
            job.start()
            running.append(job)
            time.sleep(40)  # let the process allocate its memory before the next RAM check
        main_runs = [by_name[f"E2_seed{s}"] for s in SEEDS] + [by_name["Oracle_seed42"], by_name["Eent_seed42"]]
        if not first_eval_done and all(j.state in ("done", "failed") for j in main_runs):
            run_eval_queue("main")
            first_eval_done = True
            continue
        if first_eval_done and all(j.state in ("done", "failed") for j in jobs):
            break
        time.sleep(20)

    run_eval_queue("final")
    summary = {j.name: j.state for j in jobs}
    with open(os.path.join(LOGS, "pipeline_e2.done"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"pipeline finished: {summary}")


if __name__ == "__main__":
    main()
