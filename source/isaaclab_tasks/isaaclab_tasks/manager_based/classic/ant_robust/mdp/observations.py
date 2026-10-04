# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the HW1 Ant tasks.

All functions are stateless: they only read the current simulation state. The observation manager calls them once
at start-up to infer their dimensions, so keeping state here would corrupt it.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.classic.humanoid.mdp as ant_mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

TARGET_POS = (1000.0, 0.0, 0.0)
"""Walking target of the original Ant task (far along +x)."""

FEET_BODY_NAMES = ["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
"""Bodies whose incoming joint wrench is part of the original observation."""

BASE_OBS_DIM = 60
"""Size of the original Isaac-Ant-v0 policy observation."""

PROPRIO_DIM = 22
"""Size of :func:`proprio_state`."""


def proprio_state(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """How the body is responding (22): joint positions (8), joint velocities (8), base angular velocity (3),
    gravity direction in the base frame (3).

    Scales follow the original Ant observation (joint positions normalized to the limits, joint velocities x 0.2),
    so every entry is of order one. Layout: ``[0:8]`` joint positions, ``[8:16]`` joint velocities,
    ``[16:19]`` base angular velocity, ``[19:22]`` projected gravity.
    """
    return torch.cat(
        (
            ant_mdp.joint_pos_limit_normalized(env, asset_cfg),
            0.2 * ant_mdp.joint_vel_rel(env, asset_cfg),
            ant_mdp.base_ang_vel(env, asset_cfg),
            ant_mdp.projected_gravity(env, asset_cfg),
        ),
        dim=-1,
    )


def base_height_above_origin(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Torso height above the environment origin, i.e. above the terrain where the robot spawned (diagnostic).

    The original ``base_pos_z`` is the absolute world height, which is far outside the training range when an episode
    starts on a raised tile (e.g. the top of a pyramid of stairs).
    """
    asset = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)


def ground_relative_heights(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_height: float = 2.0,
) -> torch.Tensor:
    """Torso height above the ground hit by every ray of a ray-caster (conditions E0 and E).

    Unlike ``base_pos_z`` (absolute world z), the value does not change when the whole robot stands higher, e.g. on top
    of a raised tile. Rays that hit nothing count as ``max_height``; values are clipped to ``[-max_height, max_height]``.
    Shape (num_envs, num_rays).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    hits_z = sensor.data.ray_hits_w[..., 2]
    heights = asset.data.root_pos_w[:, 2:3] - hits_z
    heights = torch.where(torch.isfinite(hits_z), heights, torch.full_like(heights, max_height))
    return heights.clamp(-max_height, max_height)


def resolved_feet_cfg(env: ManagerBasedEnv) -> SceneEntityCfg:
    """Entity config for the four feet with body indices resolved (needed by ``body_incoming_wrench``)."""
    feet_cfg = SceneEntityCfg("robot", body_names=FEET_BODY_NAMES)
    feet_cfg.resolve(env.scene)
    return feet_cfg


def base_obs(env: ManagerBasedEnv, last_action: torch.Tensor, feet_cfg: SceneEntityCfg) -> torch.Tensor:
    """The original 60-dim Isaac-Ant-v0 policy observation, computed outside the observation manager.

    The order and scales match ``ObservationsCfg.PolicyCfg`` in ``ant/ant_env_cfg.py``. ``last_action`` is passed
    explicitly: inside an action term the action buffer already holds the new (residual) action, while the original
    observation must contain the action that was applied in the previous step.

    Args:
        env: The environment.
        last_action: The action applied in the previous step. Shape (num_envs, 8).
        feet_cfg: Resolved entity config of the feet (see :func:`resolved_feet_cfg`).
    """
    robot_cfg = SceneEntityCfg("robot")
    return torch.cat(
        (
            ant_mdp.base_pos_z(env, robot_cfg),
            ant_mdp.base_lin_vel(env, robot_cfg),
            ant_mdp.base_ang_vel(env, robot_cfg),
            ant_mdp.base_yaw_roll(env, robot_cfg),
            ant_mdp.base_angle_to_target(env, TARGET_POS, robot_cfg),
            ant_mdp.base_up_proj(env, robot_cfg),
            ant_mdp.base_heading_proj(env, TARGET_POS, robot_cfg),
            ant_mdp.joint_pos_limit_normalized(env, robot_cfg),
            0.2 * ant_mdp.joint_vel_rel(env, robot_cfg),
            0.1 * ant_mdp.body_incoming_wrench(env, feet_cfg),
            last_action,
        ),
        dim=-1,
    )


def base_action(env: ManagerBasedEnv, action_name: str = "joint_effort") -> torch.Tensor:
    """Action that the frozen base policy proposes for the current state (observation of condition D).

    At observation time the action buffer holds the action applied in this step, which is exactly the
    ``last_action`` the base policy will see at the start of the next step.
    """
    term = env.action_manager.get_term(action_name)
    return term.compute_base_action(env.action_manager.action)
