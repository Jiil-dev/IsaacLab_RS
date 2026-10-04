# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Roll out a frozen base policy and record (input, target) pairs of the dynamics model.

The inputs are built with the same :class:`TransitionHistory` the residual action term uses at run time:
``x_t`` = last 3 (state, applied action) pairs, ``y_t`` = ``s_{t+1} - s_t`` on the predicted entries. A pair is
valid only if no reset happened in between.

Three datasets per seed (plan, section 5.4):

* ``flat_noisy``: Isaac-Ant-v0 (flat, friction 1.0), actions + N(0, 0.3^2) noise -> dynamics training data
* ``flat_det``: same, deterministic actions -> lower calibration point e_lo
* ``dr_det``: Isaac-Ant-DR-v0 (training distribution), deterministic -> upper calibration point e_hi

Example:

    ./isaaclab.sh -p assignments/hw1_ant/scripts/collect_rollouts.py --headless --seed 42 --dataset flat_noisy
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect dynamics-model data with a frozen base policy.")
parser.add_argument("--seed", type=int, required=True, help="Seed of the base policy (B run).")
parser.add_argument("--dataset", choices=["flat_noisy", "flat_det", "dr_det"], required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--action_noise", type=float, default=0.3, help="Noise std for the 'noisy' dataset.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.classic.ant_robust.mdp import models, proprio_state, robot_friction
from isaaclab_tasks.utils import parse_env_cfg

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    task = "Isaac-Ant-DR-v0" if args_cli.dataset == "dr_det" else "Isaac-Ant-v0"
    noise = args_cli.action_noise if args_cli.dataset == "flat_noisy" else 0.0
    # different data streams for different datasets of the same base policy
    env_seed = 1000 + args_cli.seed + {"flat_noisy": 0, "flat_det": 1, "dr_det": 2}[args_cli.dataset]
    torch.manual_seed(env_seed)

    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = env_seed
    env = gym.make(task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device
    policy = models.load_frozen_policy(f"base_policy_seed{args_cli.seed}.pt", device)
    history = models.TransitionHistory(unwrapped.num_envs, history=3, device=device)

    xs, ys, valids, resets = [], [], [], []
    obs, _ = env.reset()
    features = None
    with torch.inference_mode():
        for t in range(args_cli.steps + 1):
            state = proprio_state(unwrapped)
            if features is not None:
                # target of the prediction made from the previous step's history
                ys.append((state[:, models.TARGET_SLICE] - history.last_state()[:, models.TARGET_SLICE]).cpu())
                xs.append(features.cpu())
                valids.append((~history.fresh).cpu())
            if t == args_cli.steps:
                break
            actions = policy(obs["policy"])
            if noise > 0.0:
                actions = actions + noise * torch.randn_like(actions)
            history.push(state, actions)
            features = history.features().clone()
            obs, _, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated
            history.reset(done.nonzero().flatten())
            resets.append(done.cpu())

    data = {
        "x": torch.stack(xs),  # (T, N, 90) input of the prediction made at step t
        "y": torch.stack(ys),  # (T, N, 11) realized change s_{t+1} - s_t
        "valid": torch.stack(valids),  # (T, N) False if the env was reset in between
        "reset": torch.stack(resets),  # (T, N) episode ended at step t (window must be reset)
        "friction": robot_friction(unwrapped).cpu(),  # (N,) effective friction (DR only differs)
        "task": task,
        "seed": args_cli.seed,
        "action_noise": noise,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, f"{args_cli.dataset}_seed{args_cli.seed}.pt")
    torch.save(data, out)
    print(f"[INFO] saved {out}: x {tuple(data['x'].shape)}, valid fraction {data['valid'].float().mean():.3f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
