# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Pick the submitted model with the pre-registered rule and copy every final checkpoint (no Isaac Sim needed).

Rules (fixed before the corresponding results were seen):

* plan section 11: among B, C and D, the highest mean over seeds of the unseen average (T1, T2, T3);
* addendum section 8: among B, C, D, E0 and E, the highest mean of the T1-T4 average;
* second addendum section 8 (used for the submission): E2 replaces the addendum choice only if E2 > E on the T1-T4
  average by the comparison rule of plan section 9 and E2 is not worse than E on T5 by that rule.

The seed is the one with the highest training reward on its own training environment (mean of ``Train/mean_reward``
over the last 50 iterations; for E2 the last 50 iterations of stage 2); the test terrains are not used to pick it.

Every final checkpoint is copied to ``assignments/hw1_ant/checkpoints/<cond>_seed<S>/model.pt`` together with its
``params/`` (also the E2 ceiling reference ``Oracle_seed42``, which is never a candidate). The decision is written to
``results/submission.json``.

    python assignments/hw1_ant/scripts/select_submission.py
"""

import csv
import json
import os
import shutil

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import hw1_common as common
from sweep_eval import compare

RULES = {
    "plan": (("B", "C", "D"), ("T1", "T2", "T3")),
    "addendum": (("B", "C", "D", "E0", "E"), ("T1", "T2", "T3", "T4")),
}
EVAL_TASK = {
    "A": "Isaac-Ant-v0",
    "B": "Isaac-Ant-v0",
    "C": "Isaac-Ant-Hist-v0",
    "D": "Isaac-Ant-Residual-v0",
    "E0": "Isaac-Ant-RelHeight-v0",
    "E": "Isaac-Ant-Scan-v0",
    "E2": "Isaac-Ant-WideScan-v0",
}
FINAL_ITERATION = {"A": 2999, "B": 2999, "C": 2999, "D": 999, "E0": 2999, "E": 2999, "E2": 2998, "Oracle": 2998}


def training_reward(run_dir: str, last: int = 50) -> float:
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    values = [e.value for e in ea.Scalars("Train/mean_reward")[-last:]]
    return sum(values) / len(values)


def main():
    with open(os.path.join(common.RESULTS_DIR, "main_runs.csv")) as f:
        rows = list(csv.DictReader(f))

    train_rewards = {}
    runs = [(cond, seed) for cond in ("A", "B", "C", "D", "E0", "E", "E2") for seed in common.SEEDS]
    for cond, seed in [*runs, ("Oracle", 42)]:
        run = common.find_run(f"{cond}_seed{seed}")
        train_rewards[f"{cond}_seed{seed}"] = training_reward(run)
        # copy the final checkpoint and its configuration
        out_dir = os.path.join(common.HW1_DIR, "checkpoints", f"{cond}_seed{seed}")
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(os.path.join(run, f"model_{FINAL_ITERATION[cond]}.pt"), os.path.join(out_dir, "model.pt"))
        shutil.copytree(os.path.join(run, "params"), os.path.join(out_dir, "params"), dirs_exist_ok=True)

    decisions = {}
    for rule, (candidates, envs) in RULES.items():
        unseen = {}
        for cond in candidates:
            for seed in common.SEEDS:
                vals = [float(r["reward_mean"]) for r in rows
                        if r["condition"] == cond and int(r["seed"]) == seed and r["env"] in envs]
                if len(vals) == len(envs):
                    unseen.setdefault(cond, {})[seed] = sum(vals) / len(vals)
        if any(len(unseen.get(c, {})) != len(common.SEEDS) for c in candidates):
            raise RuntimeError(f"Missing evaluations for rule '{rule}': {unseen}")
        means = {c: sum(v.values()) / len(v) for c, v in unseen.items()}
        condition = max(means, key=means.get)
        seed = max(common.SEEDS, key=lambda s: train_rewards[f"{condition}_seed{s}"])
        decisions[rule] = {
            "candidates": candidates,
            "unseen_envs": envs,
            "unseen_mean_by_condition": means,
            "unseen_by_seed": unseen,
            "condition": condition,
            "seed": seed,
            "task": EVAL_TASK[condition],
            "checkpoint": f"assignments/hw1_ant/checkpoints/{condition}_seed{seed}/model.pt",
        }
    # second addendum: E2 replaces the addendum choice only if it is better on T1-T4 and not worse on T5
    def per_seed(cond: str, envs: tuple[str, ...]) -> dict[int, float]:
        out = {}
        for seed in common.SEEDS:
            vals = [float(r["reward_mean"]) for r in rows
                    if r["condition"] == cond and int(r["seed"]) == seed and r["env"] in envs]
            if len(vals) == len(envs):
                out[seed] = sum(vals) / len(vals)
        if len(out) != len(common.SEEDS):
            raise RuntimeError(f"Missing evaluations of {cond} on {envs}: {out}")
        return out

    unseen4 = ("T1", "T2", "T3", "T4")
    verdict_unseen4 = compare(per_seed("E2", unseen4), per_seed("E", unseen4))
    verdict_t5 = compare(per_seed("E2", ("T5",)), per_seed("E", ("T5",)))
    replace = verdict_unseen4 == ">" and verdict_t5 != "<"
    e2_seed = max(common.SEEDS, key=lambda s: train_rewards[f"E2_seed{s}"])
    decisions["addendum2"] = {
        "rule": "E2 if E2 > E on the T1-T4 average and not E2 < E on T5 (plan section 9 rule), else the addendum choice",
        "E2_vs_E_unseen4": verdict_unseen4,
        "E2_vs_E_T5": verdict_t5,
        "replace": replace,
        **(
            {"condition": "E2", "seed": e2_seed, "task": EVAL_TASK["E2"],
             "checkpoint": f"assignments/hw1_ant/checkpoints/E2_seed{e2_seed}/model.pt"}
            if replace
            else {k: decisions["addendum"][k] for k in ("condition", "seed", "task", "checkpoint")}
        ),
    }
    result = {"training_reward_last50": train_rewards, "rules": decisions, "submission": decisions["addendum2"]}
    with open(os.path.join(common.RESULTS_DIR, "submission.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(decisions, indent=1))


if __name__ == "__main__":
    main()
