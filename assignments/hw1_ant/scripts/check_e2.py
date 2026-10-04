# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Gate G7 of the second addendum (E2): configuration checks before any E2 training.

No trained policy is rolled out (the robots get zero actions), so no test score is seen before the plan is frozen.
For one task it prints the policy observation size, the height-scan footprint, the terrain height range, the number of
different spawn tiles and the robot friction range.

    ./isaaclab.sh -p assignments/hw1_ant/scripts/check_e2.py --headless --task Isaac-Ant-WideScan-T5-v0 --num_envs 100
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="HW1 gate G7 (E2 configuration).")
parser.add_argument("--task", type=str, required=True, help="Task to inspect.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.terrains import TerrainGenerator

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.classic.ant_robust import mdp
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    print(f"[G7] task {args_cli.task}, {unwrapped.num_envs} envs", flush=True)
    print(f"[G7] policy observation dims: {unwrapped.observation_manager.group_obs_dim['policy']}")

    if "height_scan" in unwrapped.scene.sensors:
        starts = unwrapped.scene.sensors["height_scan"].ray_starts[0]
        print(
            f"[G7] height scan: {starts.shape[0]} rays, x {starts[:, 0].min():+.2f}..{starts[:, 0].max():+.2f} m,"
            f" y {starts[:, 1].min():+.2f}..{starts[:, 1].max():+.2f} m (torso frame, yaw aligned)"
        )

    generator_cfg = env_cfg.scene.terrain.terrain_generator
    if env_cfg.scene.terrain.terrain_type == "generator" and generator_cfg is not None:
        heights = TerrainGenerator(cfg=generator_cfg, device="cpu").terrain_mesh.vertices[:, 2]
        print(
            f"[G7] terrain: seed {generator_cfg.seed}, {generator_cfg.num_rows} x {generator_cfg.num_cols} tiles,"
            f" difficulty {generator_cfg.difficulty_range}, sub-terrains {list(generator_cfg.sub_terrains)}"
        )
        print(f"[G7] terrain height: min {heights.min():+.3f} m, max {heights.max():+.3f} m")
        terrain = unwrapped.scene.terrain
        origins = unwrapped.scene.env_origins[:, :2]
        distinct = torch.unique(torch.round(origins * 100), dim=0).shape[0]
        print(
            f"[G7] spawn tiles: {distinct} distinct of {unwrapped.num_envs} envs,"
            f" rows {int(terrain.terrain_levels.min())}..{int(terrain.terrain_levels.max())},"
            f" columns {int(terrain.terrain_types.min())}..{int(terrain.terrain_types.max())}"
        )
    else:
        print(f"[G7] terrain: {env_cfg.scene.terrain.terrain_type}")

    friction = mdp.robot_friction(unwrapped)
    material = env_cfg.scene.terrain.physics_material
    print(
        f"[G7] robot friction {friction.min():.2f}..{friction.max():.2f}; ground friction {material.static_friction}"
        f" ({material.friction_combine_mode})"
    )

    obs, _ = env.reset()
    for _ in range(10):
        obs, _, _, _, _ = env.step(torch.zeros(unwrapped.num_envs, 8, device=unwrapped.device))
    policy_obs = obs["policy"]
    print(f"[G7] observations finite after 10 zero-action steps: {bool(torch.isfinite(policy_obs).all())}")
    if "height_scan" in unwrapped.scene.sensors:
        scan = policy_obs[:, 60:]
        print(f"[G7] height-scan values: min {scan.min():+.3f}, median {scan.median():+.3f}, max {scan.max():+.3f} m")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
