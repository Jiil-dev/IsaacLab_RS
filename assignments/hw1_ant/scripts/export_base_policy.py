# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Extract the actor of condition B at iteration 2000 as the frozen base policy of condition D.

Runs without Isaac Sim:

    python assignments/hw1_ant/scripts/export_base_policy.py --seed 42
"""

import argparse
import os

import hw1_common as common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="Seed of the B run.")
    parser.add_argument("--iteration", type=int, default=2000, help="Checkpoint iteration of B.")
    args = parser.parse_args()

    models = common.load_models_module()
    run_dir = common.find_run(f"B_seed{args.seed}")
    checkpoint = os.path.join(run_dir, f"model_{args.iteration}.pt")
    actor, info = models.actor_from_rsl_rl_checkpoint(checkpoint)
    if info["obs_dim"] != 60 or info["act_dim"] != 8:
        raise ValueError(f"Unexpected actor shape: {info}")
    info["source"] = os.path.relpath(info["source"], common.REPO_ROOT)

    os.makedirs(common.WEIGHTS_DIR, exist_ok=True)
    out = os.path.join(common.WEIGHTS_DIR, f"base_policy_seed{args.seed}.pt")
    models.save_frozen_policy(actor, info, out)
    print(f"[INFO] saved {os.path.relpath(out, common.REPO_ROOT)} from {info['source']} (iteration {info['iteration']})")


if __name__ == "__main__":
    main()
