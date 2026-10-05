# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Gate G8 of the third addendum (E3): configuration checks before any E3 training.

Like gate G7, no trained policy is rolled out (the robots get zero actions). For one task it prints the policy
observation size, the walkable terrain heights (rays cast straight down every 5 cm over all tiles), the spawn heights,
the number of different spawn tiles, the robot friction range and how many robots end their episode within 10
zero-action steps.

    ./isaaclab.sh -p assignments/hw1_ant/scripts/check_e3.py --headless --task Isaac-Ant-WideScan-T6-v0 --num_envs 100
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="HW1 gate G8 (E3 configuration).")
parser.add_argument("--task", type=str, required=True, help="Task to inspect.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

from isaaclab.terrains import TerrainGenerator
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.classic.ant_robust import mdp
from isaaclab_tasks.utils import parse_env_cfg


def walkable_heights(generator_cfg, device: str, spacing: float = 0.05) -> torch.Tensor:
    """Height of the first surface below each point of a grid over all tiles (the 20 m outer border excluded)."""
    generator = TerrainGenerator(cfg=generator_cfg, device="cpu")
    mesh = convert_to_warp_mesh(
        generator.terrain_mesh.vertices.astype(np.float32), generator.terrain_mesh.faces.astype(np.int32), device
    )
    half_x = 0.5 * generator_cfg.size[0] * generator_cfg.num_rows
    half_y = 0.5 * generator_cfg.size[1] * generator_cfg.num_cols
    xs = torch.arange(-half_x + 0.5 * spacing, half_x, spacing, device=device)
    ys = torch.arange(-half_y + 0.5 * spacing, half_y, spacing, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    starts = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.full_like(grid_x.flatten(), 5.0)], dim=-1)
    directions = torch.zeros_like(starts)
    directions[:, 2] = -1.0
    hits = raycast_mesh(starts, directions, mesh)[0]
    return hits[:, 2]


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    print(f"[G8] task {args_cli.task}, {unwrapped.num_envs} envs", flush=True)
    print(f"[G8] policy observation dims: {unwrapped.observation_manager.group_obs_dim['policy']}")

    generator_cfg = env_cfg.scene.terrain.terrain_generator
    if env_cfg.scene.terrain.terrain_type == "generator" and generator_cfg is not None:
        print(
            f"[G8] terrain: seed {generator_cfg.seed}, {generator_cfg.num_rows} x {generator_cfg.num_cols} tiles,"
            f" difficulty {generator_cfg.difficulty_range}, sub-terrains"
            f" {[(k, round(v.proportion, 2)) for k, v in generator_cfg.sub_terrains.items()]}"
        )
        z = walkable_heights(generator_cfg, unwrapped.device)
        finite = z[torch.isfinite(z)]
        q = torch.quantile(finite[torch.randperm(finite.numel(), device=finite.device)[:1_000_000]],
                           torch.tensor([0.01, 0.5, 0.99], device=finite.device))
        print(
            f"[G8] walkable height over {z.numel()} rays ({z.numel() - finite.numel()} misses): min {finite.min():+.3f},"
            f" p1 {q[0]:+.3f}, median {q[1]:+.3f}, p99 {q[2]:+.3f}, max {finite.max():+.3f} m;"
            f" below 0: {100 * (finite < -1e-4).float().mean():.1f} %, below -0.10 m: {100 * (finite < -0.10).float().mean():.1f} %"
        )
        terrain = unwrapped.scene.terrain
        origins = unwrapped.scene.env_origins
        distinct = torch.unique(torch.round(origins[:, :2] * 100), dim=0).shape[0]
        print(
            f"[G8] spawn tiles: {distinct} distinct of {unwrapped.num_envs} envs,"
            f" rows {int(terrain.terrain_levels.min())}..{int(terrain.terrain_levels.max())},"
            f" columns {int(terrain.terrain_types.min())}..{int(terrain.terrain_types.max())};"
            f" spawn ground z {origins[:, 2].min():+.3f}..{origins[:, 2].max():+.3f} m"
        )
    else:
        print(f"[G8] terrain: {env_cfg.scene.terrain.terrain_type}")

    friction = mdp.robot_friction(unwrapped)
    material = env_cfg.scene.terrain.physics_material
    print(
        f"[G8] robot friction {friction.min():.2f}..{friction.max():.2f}; ground friction {material.static_friction}"
        f" ({material.friction_combine_mode})"
    )

    obs, _ = env.reset()
    ended = torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device)
    for _ in range(10):
        obs, _, terminated, truncated, _ = env.step(torch.zeros(unwrapped.num_envs, 8, device=unwrapped.device))
        ended |= terminated | truncated
    policy_obs = obs["policy"]
    print(f"[G8] observations finite after 10 zero-action steps: {bool(torch.isfinite(policy_obs).all())};"
          f" episodes ended: {int(ended.sum())} of {unwrapped.num_envs}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
