# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Friction events for the HW1 Ant tasks.

The terrain is a single mesh with a single material, so friction cannot differ between environments on the ground
side. Instead, every environment gets its own robot friction. With the ground material combine mode set to
``multiply`` and ground friction 1.0, the robot value is the effective contact friction.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_robot_friction(
    env: ManagerBasedEnv,
    friction: float | torch.Tensor,
    env_ids: torch.Tensor | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set static and dynamic friction of all robot shapes.

    Args:
        env: The environment.
        friction: One value for all selected environments, or one value per selected environment.
        env_ids: Environments to modify. Defaults to all.
        asset_cfg: The robot.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # the PhysX tensor API for materials works on CPU tensors
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    friction = torch.as_tensor(friction, dtype=torch.float32).cpu().reshape(-1, 1)
    materials = asset.root_physx_view.get_material_properties()  # (num_envs, num_shapes, [static, dynamic, restitution])
    materials[env_ids, :, 0] = friction
    materials[env_ids, :, 1] = friction
    asset.root_physx_view.set_material_properties(materials, env_ids)


def randomize_robot_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    friction_range: tuple[float, float],
    num_buckets: int = 64,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Give every environment one friction value (static = dynamic) drawn from ``num_buckets`` evenly spaced values.

    Used as a ``startup`` event, so each environment keeps its friction for the whole training run. Buckets keep the
    number of distinct PhysX materials small.
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    buckets = torch.linspace(friction_range[0], friction_range[1], num_buckets)
    friction = buckets[torch.randint(0, num_buckets, (len(env_ids),))]
    set_robot_friction(env, friction, env_ids, asset_cfg)


def robot_friction(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Current static friction of the first robot shape in every environment (for analysis). Shape (num_envs,)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view.get_material_properties()[:, 0, 0].clone()
