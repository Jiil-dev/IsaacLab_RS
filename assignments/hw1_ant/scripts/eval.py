# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a checkpoint like ``play_one_episode.py`` and record walking metrics (HW1).

The accumulation rule is the official one: every environment accumulates reward up to and including the step where
its first episode ends; later (auto-reset) episodes are ignored. On top of the reward this script records, per
environment, whether it fell, how far it walked, the reward of every term, and (for condition D) the correction
strength alpha. NaN rewards are reported instead of silently poisoning the mean.

Example:

    ./isaaclab.sh -p assignments/hw1_ant/scripts/eval.py --headless --task Isaac-Ant-v0 \\
        --checkpoint logs/rsl_rl/ant/2026-09-17_13-19-56_ant_baseline/model_999.pt \\
        --out assignments/hw1_ant/results/raw/A_ref_flat.json

Hydra overrides of the environment can be appended, e.g.
``env.scene.terrain.physics_material.static_friction=0.3``.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "reinforcement_learning", "rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint with walking metrics.")
parser.add_argument("--task", type=str, required=True, help="Task to evaluate on.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments.")
parser.add_argument("--seed", type=int, default=24, help="Seed of the environment (official command uses 24).")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
parser.add_argument("--out", type=str, required=True, help="Output JSON file.")
parser.add_argument("--label", type=str, default=None, help="Free-form label stored in the output.")
parser.add_argument(
    "--friction_switch", type=str, default=None, help="'STEP:MU' sets the robot friction of all envs at STEP."
)
parser.add_argument("--trace_envs", type=int, default=8, help="Number of envs whose time series are saved.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import numpy as np
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.classic.ant_robust.mdp import set_robot_friction
from isaaclab_tasks.manager_based.classic.ant_robust.mdp.residual_action import ResidualJointEffortAction
from isaaclab_tasks.utils.hydra import hydra_task_config


def _stats(values: torch.Tensor) -> dict:
    """Mean and population std (as in play_one_episode.py), also ignoring NaN entries."""
    values = values.double()
    finite = values[torch.isfinite(values)]
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "mean_finite": finite.mean().item() if len(finite) else float("nan"),
        "std_finite": finite.std(unbiased=False).item() if len(finite) > 1 else float("nan"),
        "num_nonfinite": int((~torch.isfinite(values)).sum().item()),
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    checkpoint = retrieve_file_path(args_cli.checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=unwrapped.device)

    robot = unwrapped.scene["robot"]
    reward_manager = unwrapped.reward_manager
    term_names = list(reward_manager.active_terms)
    num_envs, device, dt = env.num_envs, unwrapped.device, unwrapped.step_dt
    residual = next(
        (t for t in unwrapped.action_manager._terms.values() if isinstance(t, ResidualJointEffortAction)), None
    )
    switch = None
    if args_cli.friction_switch:
        step_str, mu_str = args_cli.friction_switch.split(":")
        switch = (int(step_str), float(mu_str))

    reward_sum = torch.zeros(num_envs, dtype=torch.float64, device=device)
    term_sums = torch.zeros(num_envs, len(term_names), dtype=torch.float64, device=device)
    steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
    fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
    alpha_sum = torch.zeros(num_envs, dtype=torch.float64, device=device)
    # size of the base action, of the PPO correction and of the applied correction alpha * delta (D only)
    norm_sums = {key: torch.zeros(num_envs, dtype=torch.float64, device=device) for key in ("base", "delta", "applied")}
    k = min(args_cli.trace_envs, num_envs)
    traces = {"alpha": [], "pred_error": [], "vel_x": [], "torso_z": [], "active": []}
    nan_events = []

    obs = env.get_observations()
    start = time.time()
    for t in range(int(env.max_episode_length)):
        if switch is not None and t == switch[0]:
            set_robot_friction(unwrapped, switch[1])
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
        active = ~finished
        step_terms = reward_manager._step_reward.double() * dt  # reward contribution of every term in this step
        bad = active & ~torch.isfinite(rewards)
        if bad.any() and len(nan_events) < 20:
            for env_id in bad.nonzero().flatten().tolist()[:5]:
                nan_events.append({
                    "step": t,
                    "env": env_id,
                    "terms": dict(zip(term_names, step_terms[env_id].tolist())),
                    "root_pos": robot.data.root_pos_w[env_id].tolist(),
                    "max_abs_joint_vel": robot.data.joint_vel[env_id].abs().max().item(),
                })
        reward_sum[active] += rewards[active].double()
        term_sums[active] += step_terms[active]
        steps[active] += 1
        if residual is not None:
            alpha_sum[active] += residual.alpha[active].double()
            applied = residual.alpha.unsqueeze(-1) * residual.raw_actions
            norm_sums["base"][active] += residual.base_actions[active].norm(dim=-1).double()
            norm_sums["delta"][active] += residual.raw_actions[active].norm(dim=-1).double()
            norm_sums["applied"][active] += applied[active].norm(dim=-1).double()
        ended = dones.bool() & active
        fell |= ended & ~extras["time_outs"].bool()
        finished |= dones.bool()
        traces["active"].append(active[:k].cpu())
        traces["vel_x"].append(robot.data.root_lin_vel_w[:k, 0].cpu())
        traces["torso_z"].append(robot.data.root_pos_w[:k, 2].cpu())
        if residual is not None:
            traces["alpha"].append(residual.alpha[:k].cpu())
            traces["pred_error"].append(residual.pred_error[:k].cpu())
        if finished.all():
            break
    wall_time = time.time() - start

    progress = term_sums[:, term_names.index("progress")]
    survival_time = steps.double() * dt
    summary = {
        "reward": _stats(reward_sum),
        "fall_rate": fell.double().mean().item(),
        "distance_m": _stats(progress),
        "speed_mps": _stats(progress / survival_time.clamp(min=dt)),
        "steps": _stats(steps.double()),
        "terms": {name: _stats(term_sums[:, i]) for i, name in enumerate(term_names)},
        "completed_envs": int(finished.sum().item()),
    }
    if residual is not None:
        summary["alpha_mean"] = _stats(alpha_sum / steps.double().clamp(min=1))
        # mean L2 norm per step: base action, PPO correction delta, and the applied correction alpha * delta
        summary["action_norms"] = {
            key: _stats(value / steps.double().clamp(min=1)) for key, value in norm_sums.items()
        }
    result = {
        "label": args_cli.label,
        "task": args_cli.task,
        "checkpoint": os.path.relpath(checkpoint, REPO_ROOT),
        "seed": agent_cfg.seed,
        "num_envs": num_envs,
        "hydra_overrides": hydra_args,
        "friction_switch": args_cli.friction_switch,
        "wall_time_s": wall_time,
        "summary": summary,
        "per_env": {
            "reward": reward_sum.tolist(),
            "steps": steps.tolist(),
            "fell": fell.tolist(),
            "distance_m": progress.tolist(),
            "alpha_mean": (alpha_sum / steps.double().clamp(min=1)).tolist() if residual is not None else None,
        },
        "nan_events": nan_events,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(
        os.path.splitext(args_cli.out)[0] + "_traces.npz",
        **{key: torch.stack(val).numpy() for key, val in traces.items() if len(val)},
        dt=dt,
    )
    r = summary["reward"]
    print(f"[RESULT] Episode reward total: mean={r['mean']:.6f}, std={r['std']:.6f}")
    print(f"[RESULT] finite-only mean={r['mean_finite']:.6f}, std={r['std_finite']:.6f}, non-finite envs={r['num_nonfinite']}")
    print(f"[RESULT] fall rate={summary['fall_rate']:.3f}, distance={summary['distance_m']['mean_finite']:.2f} m")
    if nan_events:
        print(f"[NAN] first events: {json.dumps(nan_events[:3])}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
