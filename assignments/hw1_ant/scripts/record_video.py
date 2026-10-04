# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Record one episode of a chosen environment (the official play script always follows env 0).

The camera follows the robot of ``--env_index``. With 100 environments on a test terrain, env ``i`` starts on
terrain column ``i // 10`` (see ``results/diagnosis/spawn_T1.json`` for the spawn heights).

    ./isaaclab.sh -p assignments/hw1_ant/scripts/record_video.py --headless --task Isaac-Ant-T1-v0 \\
        --checkpoint logs/rsl_rl/ant/2026-09-17_13-19-56_ant_baseline/model_999.pt --env_index 10 \\
        --out assignments/hw1_ant/videos/T1_A_ref.mp4
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "reinforcement_learning", "rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Record an episode of one environment.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--seed", type=int, default=24)
parser.add_argument("--env_index", type=int, default=0, help="Environment followed by the camera.")
parser.add_argument("--video_length", type=int, default=960)
parser.add_argument("--out", type=str, required=True, help="Output .mp4 file.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import glob
import gymnasium as gym
import shutil
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = args_cli.env_index
    env_cfg.viewer.eye = (-4.0, 4.0, 2.5)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)

    video_dir = os.path.splitext(os.path.abspath(args_cli.out))[0] + "_frames"
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env, video_folder=video_dir, step_trigger=lambda step: step == 0, video_length=args_cli.video_length,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args_cli.video_length + 1):
            obs, _, _, _ = env.step(policy(obs))
    env.close()
    recorded = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    shutil.move(recorded[-1], args_cli.out)
    shutil.rmtree(video_dir, ignore_errors=True)
    print(f"[INFO] saved {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
