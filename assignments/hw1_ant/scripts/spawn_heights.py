# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnosis: spawn height (terrain origin z) and terrain column of every environment of a test task.

Used to relate the walking distance of each environment (``eval.py`` per-env results, same seed and env count) to
where it started. With seed 24 and 100 envs, the env order matches the evaluation runs.

    ./isaaclab.sh -p assignments/hw1_ant/scripts/spawn_heights.py --headless --task Isaac-Ant-T1-v0 \\
        --out assignments/hw1_ant/results/diagnosis/spawn_T1.json
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record the spawn height of every environment.")
parser.add_argument("--task", default="Isaac-Ant-T1-v0")
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--seed", type=int, default=24)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    terrain = env.unwrapped.scene.terrain
    result = {
        "task": args_cli.task,
        "origin_z": terrain.env_origins[:, 2].tolist(),
        "column": terrain.terrain_types.tolist() if terrain.terrain_types is not None else None,
    }
    with open(args_cli.out, "w") as f:
        json.dump(result, f)
    print(f"[INFO] saved {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
