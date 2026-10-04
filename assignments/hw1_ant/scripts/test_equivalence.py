# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Gate G4: check that the residual action term of D is wired exactly as intended.

Every step, inside the D environment, with a random correction ``delta``:

(a) the first 60 observation entries (observation manager) equal ``base_obs()`` recomputed by hand,
(b) the ``base_action`` observation equals B's actor applied to those 60 entries,
(c) the base action used by the term equals B's actor, rebuilt independently from the raw B checkpoint,
(d) the applied action is ``base + alpha * delta`` (``alpha = 0`` with ``--alpha_mode zero``),
(e) the action manager buffer (read by the rewards and the ``last_action`` observation) holds the applied action,
(f) the joint torques are ``7.5 x`` the applied action.

With ``alpha = 0`` this proves that D reproduces B exactly, whatever the PPO policy outputs.

    ./isaaclab.sh -p assignments/hw1_ant/scripts/test_equivalence.py --headless --seed 42 --alpha_mode zero
"""

import argparse
import glob
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G4 equivalence test of the residual action term.")
parser.add_argument("--seed", type=int, default=42, help="Seed of the B run whose policy is the base.")
parser.add_argument("--alpha_mode", choices=["zero", "error", "one"], default="zero")
parser.add_argument("--task", type=str, default="Isaac-Ant-Residual-DR-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--tol", type=float, default=1e-5)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.classic.ant_robust.mdp import base_obs, models, resolved_feet_cfg
from isaaclab_tasks.utils import parse_env_cfg

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 7
    action_cfg = env_cfg.actions.joint_effort
    action_cfg.alpha_mode = args_cli.alpha_mode
    action_cfg.base_policy_file = f"base_policy_seed{args_cli.seed}.pt"
    action_cfg.dynamics_file = f"dynamics_seed{args_cli.seed}.pt"
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    term = unwrapped.action_manager.get_term("joint_effort")
    feet_cfg = resolved_feet_cfg(unwrapped)

    run = sorted(glob.glob(os.path.join(REPO_ROOT, "logs", "rsl_rl", "ant_hw1", f"*_B_seed{args_cli.seed}")))[-1]
    reference, _ = models.actor_from_rsl_rl_checkpoint(os.path.join(run, "model_2000.pt"))
    reference = reference.to(unwrapped.device).eval()

    worst = {k: 0.0 for k in ("a_obs", "b_base_action_obs", "c_base_action", "d_final", "e_buffer", "f_torque")}
    alphas = []
    obs, _ = env.reset()
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            policy_obs = obs["policy"]
            base_part = policy_obs[:, :60]
            recomputed = base_obs(unwrapped, unwrapped.action_manager.action, feet_cfg)
            expected_base = reference(base_part)
            worst["a_obs"] = max(worst["a_obs"], (base_part - recomputed).abs().max().item())
            worst["b_base_action_obs"] = max(
                worst["b_base_action_obs"], (policy_obs[:, -8:] - expected_base).abs().max().item()
            )

            delta = 0.5 * torch.randn(unwrapped.num_envs, 8, device=unwrapped.device)
            obs, _, _, _, _ = env.step(delta)

            # environments that terminated in this step were reset at its end (buffers zeroed): skip them
            alive = unwrapped.episode_length_buf > 0
            alpha = term.alpha.unsqueeze(-1)
            alphas.append(term.alpha[alive].clone())
            worst["c_base_action"] = max(
                worst["c_base_action"], (term.base_actions - expected_base)[alive].abs().max().item()
            )
            worst["d_final"] = max(
                worst["d_final"], (term.final_actions - (expected_base + alpha * delta))[alive].abs().max().item()
            )
            worst["e_buffer"] = max(
                worst["e_buffer"], (unwrapped.action_manager.action - term.final_actions)[alive].abs().max().item()
            )
            worst["f_torque"] = max(
                worst["f_torque"], (term.processed_actions - 7.5 * term.final_actions)[alive].abs().max().item()
            )

    alphas = torch.cat(alphas)
    if args_cli.alpha_mode == "zero":
        worst["alpha_max"] = alphas.abs().max().item()
    passed = all(v <= args_cli.tol for v in worst.values())
    report = {
        "seed": args_cli.seed,
        "alpha_mode": args_cli.alpha_mode,
        "steps": args_cli.steps,
        "num_envs": unwrapped.num_envs,
        "max_abs_differences": worst,
        "alpha_mean": alphas.mean().item(),
        "alpha_fraction_positive": (alphas > 0).float().mean().item(),
        "passed": passed,
    }
    out_dir = os.path.join(REPO_ROOT, "assignments", "hw1_ant", "results", "checks")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"equivalence_seed{args_cli.seed}_{args_cli.alpha_mode}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"[RESULT] G4 {'PASS' if passed else 'FAIL'}: {json.dumps(worst)}")
    print(f"[RESULT] alpha mean {report['alpha_mean']:.3f}, fraction > 0: {report['alpha_fraction_positive']:.3f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
