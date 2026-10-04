# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Build the evaluation job list and aggregate the results (no Isaac Sim import here).

    python assignments/hw1_ant/scripts/sweep_eval.py jobs --out logs/hw1_runlogs/eval_jobs.jsonl
    python assignments/hw1_ant/scripts/run_queue.py logs/hw1_runlogs/eval_jobs.jsonl --gpus 0,1 --per_gpu 2
    python assignments/hw1_ant/scripts/sweep_eval.py aggregate

Evaluated policies: A_ref, A, B, C, D (3 seeds each) and B@2000 (the frozen base of D, to isolate the correction).
Environments: Flat, T1, T2, T3 (100 envs), Grid at five frictions (200 envs) and the friction switch (100 envs).
"""

import argparse
import csv
import json
import os

import hw1_common as common

A_REF_CHECKPOINT = "logs/rsl_rl/ant/2026-09-17_13-19-56_ant_baseline/model_999.pt"
MAIN_ENVS = ("Flat", "T1", "T2", "T3")
UNSEEN_ENVS = ("T1", "T2", "T3")
GRID_FRICTIONS = (0.2, 0.3, 0.5, 0.8, 1.0)
SWITCH = "300:0.25"
OBS_TASK_PREFIX = {"base": "Isaac-Ant", "hist": "Isaac-Ant-Hist", "residual": "Isaac-Ant-Residual"}
FINAL_ITERATION = {"A": 2999, "B": 2999, "C": 2999, "D": 999}
"""Last checkpoint of a finished run (A, B, C: 3000 iterations; D: 1000 iterations on top of B@2000)."""


def policies() -> list[dict]:
    """Every evaluated policy with its checkpoint, observation type and extra environment overrides."""
    rel = lambda p: os.path.relpath(p, common.REPO_ROOT)  # noqa: E731
    out = [{"name": "A_ref", "condition": "A_ref", "seed": 42, "obs": "base", "checkpoint": A_REF_CHECKPOINT}]
    for seed in common.SEEDS:
        # runs that do not exist yet or are still training are skipped
        for cond, obs in (("A", "base"), ("B", "base"), ("C", "hist"), ("D", "residual")):
            try:
                run = common.find_run(f"{cond}_seed{seed}")
            except FileNotFoundError:
                continue
            final = os.path.join(run, f"model_{FINAL_ITERATION[cond]}.pt")
            if not os.path.exists(final):
                continue
            pol = {"name": f"{cond}_seed{seed}", "condition": cond, "seed": seed, "obs": obs,
                   "checkpoint": rel(final)}
            if cond == "D":
                pol["overrides"] = [f"env.actions.joint_effort.base_policy_file=base_policy_seed{seed}.pt",
                                    f"env.actions.joint_effort.dynamics_file=dynamics_seed{seed}.pt"]
            out.append(pol)
            if cond == "B" and os.path.exists(os.path.join(run, "model_2000.pt")):
                out.append({"name": f"B2000_seed{seed}", "condition": "B2000", "seed": seed, "obs": "base",
                            "checkpoint": rel(os.path.join(run, "model_2000.pt"))})
    return out


def task_id(obs: str, env: str) -> str:
    prefix = OBS_TASK_PREFIX[obs]
    return f"{prefix}-v0" if env == "Flat" else f"{prefix}-{env}-v0"


def make_jobs(args):
    jobs = []
    for pol in policies():
        if args.only and pol["condition"] not in args.only.split(","):
            continue
        base = ["assignments/hw1_ant/scripts/eval.py", "--headless", "--checkpoint", pol["checkpoint"],
                "--label", pol["name"]]
        extra = pol.get("overrides", [])
        for env in MAIN_ENVS:
            out = f"assignments/hw1_ant/results/raw/main/{pol['name']}_{env}.json"
            jobs.append({"name": f"eval_{pol['name']}_{env}", "out": out,
                         "args": [*base, "--task", task_id(pol["obs"], env), "--out", out, *extra]})
        for mu in GRID_FRICTIONS:
            out = f"assignments/hw1_ant/results/raw/grid/{pol['name']}_mu{mu}.json"
            jobs.append({"name": f"grid_{pol['name']}_mu{mu}", "out": out,
                         "args": [*base, "--task", task_id(pol["obs"], "Grid"), "--num_envs", "200", "--out", out,
                                  *extra, f"env.scene.terrain.physics_material.static_friction={mu}",
                                  f"env.scene.terrain.physics_material.dynamic_friction={mu}"]})
        out = f"assignments/hw1_ant/results/raw/switch/{pol['name']}.json"
        jobs.append({"name": f"switch_{pol['name']}", "out": out,
                     "args": [*base, "--task", task_id(pol["obs"], "Switch"), "--friction_switch", SWITCH,
                              "--out", out, *extra]})
    with open(args.out, "w") as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")
    print(f"[INFO] wrote {len(jobs)} jobs to {args.out}")


def _load(path: str) -> dict | None:
    full = os.path.join(common.REPO_ROOT, path)
    if not os.path.exists(full):
        return None
    with open(full) as f:
        return json.load(f)


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mean, std


def compare(x: dict[int, float], y: dict[int, float]) -> str:
    """Plan section 9: 'better' needs mean difference > max std AND all three seed pairs in the same direction."""
    seeds = sorted(set(x) & set(y))
    mx, sx = _mean_std([x[s] for s in seeds])
    my, sy = _mean_std([y[s] for s in seeds])
    bigger = abs(mx - my) > max(sx, sy)
    if all(x[s] > y[s] for s in seeds):
        same = ">"
    elif all(x[s] < y[s] for s in seeds):
        same = "<"
    else:
        same = None
    if bigger and same:
        return same
    if bigger or same:
        return f"{'>' if mx > my else '<'} (trend)"
    return "~"


def aggregate(args):
    rows = []
    pols = policies()
    for pol in pols:
        for env in MAIN_ENVS:
            res = _load(f"assignments/hw1_ant/results/raw/main/{pol['name']}_{env}.json")
            if res is None:
                continue
            s = res["summary"]
            rows.append({
                "policy": pol["name"], "condition": pol["condition"], "seed": pol["seed"], "env": env,
                "reward_mean": s["reward"]["mean"], "reward_std": s["reward"]["std"],
                "nonfinite_envs": s["reward"]["num_nonfinite"], "fall_rate": s["fall_rate"],
                "distance_m": s["distance_m"]["mean"], "speed_mps": s["speed_mps"]["mean"],
                "steps": s["steps"]["mean"], "alpha_mean": s.get("alpha_mean", {}).get("mean"),
            })
    os.makedirs(common.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(common.RESULTS_DIR, "main_runs.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # condition level: mean +- std over the seed-level means
    by = {}
    for r in rows:
        by.setdefault((r["condition"], r["env"]), {})[r["seed"]] = r
    conditions = ["A_ref", "A", "B", "C", "B2000", "D"]
    summary = {}
    for cond in conditions:
        for env in MAIN_ENVS:
            runs = by.get((cond, env))
            if not runs:
                continue
            entry = {}
            for key in ("reward_mean", "fall_rate", "distance_m", "speed_mps", "alpha_mean"):
                vals = [r[key] for r in runs.values() if r[key] is not None]
                if vals:
                    entry[key] = _mean_std(vals)
            entry["per_seed_reward"] = {s: r["reward_mean"] for s, r in runs.items()}
            summary[(cond, env)] = entry
        unseen = [summary.get((cond, e)) for e in UNSEEN_ENVS]
        if all(unseen):
            seeds = set.intersection(*[set(u["per_seed_reward"]) for u in unseen])
            per_seed = {s: sum(u["per_seed_reward"][s] for u in unseen) / 3 for s in seeds}
            summary[(cond, "Unseen")] = {"reward_mean": _mean_std(list(per_seed.values())), "per_seed_reward": per_seed}

    lines = ["# HW1 evaluation summary", "", "Reward: mean +- std over seed-level means (each = 100-env mean).", ""]
    lines.append("| condition | " + " | ".join([*MAIN_ENVS, "Unseen"]) + " |")
    lines.append("|---" * (len(MAIN_ENVS) + 2) + "|")
    for cond in conditions:
        cells = []
        for env in [*MAIN_ENVS, "Unseen"]:
            e = summary.get((cond, env))
            cells.append(f"{e['reward_mean'][0]:.1f} +- {e['reward_mean'][1]:.1f}" if e else "-")
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines += ["", "| condition | env | fall rate | distance (m) | speed (m/s) | alpha |", "|---|---|---|---|---|---|"]
    for cond in conditions:
        for env in MAIN_ENVS:
            e = summary.get((cond, env))
            if not e:
                continue
            alpha = f"{e['alpha_mean'][0]:.3f}" if "alpha_mean" in e else "-"
            lines.append(f"| {cond} | {env} | {e['fall_rate'][0]:.2f} | {e['distance_m'][0]:.1f} | "
                         f"{e['speed_mps'][0]:.2f} | {alpha} |")
    lines += ["", "## Pre-registered comparisons (plan section 9)", "", "| comparison | " + " | ".join([*MAIN_ENVS, "Unseen"]) + " |",
              "|---" * (len(MAIN_ENVS) + 2) + "|"]
    for x, y in (("B", "A"), ("C", "B"), ("D", "B"), ("D", "C"), ("D", "B2000"), ("A", "A_ref")):
        cells = []
        for env in [*MAIN_ENVS, "Unseen"]:
            ex, ey = summary.get((x, env)), summary.get((y, env))
            if ex and ey and len(ex["per_seed_reward"]) > 1 and len(ey["per_seed_reward"]) > 1:
                cells.append(compare(ex["per_seed_reward"], ey["per_seed_reward"]))
            else:
                cells.append("-")
        lines.append(f"| {x} vs {y} | " + " | ".join(cells) + " |")

    # heatmap table
    grid_rows = []
    for pol in pols:
        for mu in GRID_FRICTIONS:
            res = _load(f"assignments/hw1_ant/results/raw/grid/{pol['name']}_mu{mu}.json")
            if res is None:
                continue
            rewards = res["per_env"]["reward"]
            fell = res["per_env"]["fell"]
            per_col = len(rewards) // 4
            for col, height in enumerate((0.0, 0.05, 0.10, 0.15)):
                sl = slice(col * per_col, (col + 1) * per_col)
                vals = [v for v in rewards[sl]]
                grid_rows.append({"policy": pol["name"], "condition": pol["condition"], "seed": pol["seed"],
                                  "friction": mu, "max_height": height, "reward_mean": sum(vals) / len(vals),
                                  "fall_rate": sum(fell[sl]) / len(vals)})
    if grid_rows:
        with open(os.path.join(common.RESULTS_DIR, "grid_runs.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
            writer.writeheader()
            writer.writerows(grid_rows)
    with open(os.path.join(common.RESULTS_DIR, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(common.RESULTS_DIR, "summary.json"), "w") as f:
        json.dump({f"{c}|{e}": v for (c, e), v in summary.items()}, f, indent=1)
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_jobs = sub.add_parser("jobs")
    p_jobs.add_argument("--out", required=True)
    p_jobs.add_argument("--only", default=None, help="Comma-separated conditions to include.")
    sub.add_parser("aggregate")
    args = parser.parse_args()
    make_jobs(args) if args.cmd == "jobs" else aggregate(args)


if __name__ == "__main__":
    main()
