# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Create the starting checkpoint of condition D (runs without Isaac Sim).

* actor (518 -> 400 -> 200 -> 100 -> 8): random hidden layers, **last layer zero**, so the correction is zero and D
  starts exactly as the frozen base policy B@2000;
* critic (518 -> ... -> 1): copied from B@2000's critic; the first layer keeps B's weights on the 60 original
  observation inputs and zeros on the 458 new ones, so V_D(s) = V_B(s_base) at the start;
* exploration noise std 0.2 and a fresh Adam optimizer (lr 1e-4).

The file is written to ``logs/rsl_rl/ant_hw1/D_init_seed<S>/model_0.pt`` and training resumes from it:

    python assignments/hw1_ant/scripts/make_residual_init.py --seed 42
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-Residual-DR-v0 --headless \\
        --seed 42 --run_name D_seed42 --resume --load_run D_init_seed42 --checkpoint model_0.pt \\
        env.actions.joint_effort.base_policy_file=base_policy_seed42.pt \\
        env.actions.joint_effort.dynamics_file=dynamics_seed42.pt
"""

import argparse
import os
import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic

import hw1_common as common

BASE_OBS_DIM = 60
RESIDUAL_OBS_DIM = 518


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iteration", type=int, default=2000, help="Iteration of the B checkpoint.")
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    obs = TensorDict({"policy": torch.zeros(1, RESIDUAL_OBS_DIM)}, batch_size=[1])
    policy = ActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["policy"]},
        num_actions=8,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="elu",
        init_noise_std=args.noise_std,
        noise_std_type="scalar",
    )

    b_checkpoint = os.path.join(common.find_run(f"B_seed{args.seed}"), f"model_{args.iteration}.pt")
    b_state = torch.load(b_checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    state = policy.state_dict()
    # actor: zero correction at the start
    state["actor.6.weight"].zero_()
    state["actor.6.bias"].zero_()
    # critic: B's critic on the original 60 inputs, zero weights on the new inputs
    state["critic.0.weight"].zero_()
    state["critic.0.weight"][:, :BASE_OBS_DIM] = b_state["critic.0.weight"]
    for key in ("critic.0.bias", "critic.2.weight", "critic.2.bias", "critic.4.weight", "critic.4.bias",
                "critic.6.weight", "critic.6.bias"):
        state[key] = b_state[key].clone()
    policy.load_state_dict(state)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    out_dir = os.path.join(common.LOG_ROOT, f"D_init_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "model_0.pt")
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": 0,
            "infos": {"critic_from": os.path.relpath(b_checkpoint, common.REPO_ROOT)},
        },
        out,
    )
    print(f"[INFO] saved {os.path.relpath(out, common.REPO_ROOT)} (critic from {os.path.relpath(b_checkpoint, common.REPO_ROOT)})")


if __name__ == "__main__":
    main()
