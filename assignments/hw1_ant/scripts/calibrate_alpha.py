# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Fix the alpha calibration of condition D from training-distribution data only (runs without Isaac Sim).

The window-mean prediction error is replayed with the same :class:`ErrorWindow` used at run time on

* ``flat_det``: nominal ground, deterministic base policy -> ``e_lo`` = 95th percentile,
* ``dr_det``: the DR training environment, deterministic base policy -> ``e_hi`` = 90th percentile.

Gate G3 (plan, section 11): validation R^2 >= 0.3, AUROC(flat vs DR window errors) >= 0.70 and ``e_hi > e_lo``.
The test terrains T1-T3 are not used here.

    python assignments/hw1_ant/scripts/calibrate_alpha.py --seed 42
"""

import argparse
import json
import os
import torch

import hw1_common as common

WINDOW = 60
"""Steps averaged before mapping to alpha (60 steps = 1 s). It must match ``error_window`` of the action term.

The plan used 15 steps. Gate G3 failed with 15 (seed 42 AUROC 0.677); the pre-declared retries were 30 then 60
steps, adopting the shortest window that passes for every seed. With 30, seed 44 failed (0.674); with 60 all seeds
passed (0.837 / 0.842 / 0.732)."""


def window_means(models, model, data, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Window-mean errors at every step where the window is full, and the env index of each value."""
    x, y, valid, reset = data["x"], data["y"], data["valid"], data["reset"]
    num_steps, num_envs = valid.shape
    window = models.ErrorWindow(num_envs, WINDOW, device)
    means, env_ids = [], []
    with torch.no_grad():
        for t in range(num_steps):
            error = model.prediction_error(model(x[t].to(device).float()), y[t].to(device).float())
            window.push(error, valid[t].to(device))
            full = window.full()
            means.append(window.mean()[full].cpu())
            env_ids.append(full.nonzero().flatten().cpu())
            # the episode ended in this step: the next prediction belongs to a new episode
            window.reset(reset[t].to(device).nonzero().flatten())
    return torch.cat(means), torch.cat(env_ids)


def auroc(negatives: torch.Tensor, positives: torch.Tensor) -> float:
    """Probability that a random positive scores higher than a random negative (rank formulation)."""
    scores = torch.cat((negatives, positives)).double()
    labels = torch.cat((torch.zeros(len(negatives)), torch.ones(len(positives))))
    ranks = torch.empty_like(scores)
    ranks[scores.argsort()] = torch.arange(1, len(scores) + 1, dtype=torch.double)
    n_pos, n_neg = len(positives), len(negatives)
    return ((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--q_lo", type=float, default=0.95)
    parser.add_argument("--q_hi", type=float, default=0.90)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    models = common.load_models_module()

    path = os.path.join(common.WEIGHTS_DIR, f"dynamics_seed{args.seed}.pt")
    model = models.load_dynamics(path, args.device)
    raw = torch.load(path, weights_only=False)
    flat = torch.load(os.path.join(common.DATA_DIR, f"flat_det_seed{args.seed}.pt"), weights_only=False)
    dr = torch.load(os.path.join(common.DATA_DIR, f"dr_det_seed{args.seed}.pt"), weights_only=False)

    flat_means, _ = window_means(models, model, flat, args.device)
    dr_means, dr_envs = window_means(models, model, dr, args.device)
    e_lo = torch.quantile(flat_means.float(), args.q_lo).item()
    e_hi = torch.quantile(dr_means.float(), args.q_hi).item()
    area = auroc(flat_means, dr_means)

    # descriptive: error and alpha by friction bucket of the DR environments
    friction = dr["friction"][dr_envs]
    edges = torch.tensor([0.3, 0.5, 0.7, 0.9, 1.21])
    by_friction = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (friction >= lo) & (friction < hi)
        if sel.any():
            alpha = ((dr_means[sel] - e_lo) / (e_hi - e_lo)).clamp(0, 1)
            by_friction.append({
                "friction": [round(lo.item(), 2), round(hi.item(), 2)],
                "median_error": dr_means[sel].median().item(),
                "mean_alpha": alpha.mean().item(),
            })

    r2 = raw["info"]["train_report"]["val_r2_mean"]
    gate = {"val_r2_ok": r2 >= 0.3, "auroc_ok": area >= 0.70, "order_ok": e_hi > e_lo}
    report = {
        "seed": args.seed,
        "window": WINDOW,
        "e_lo": e_lo,
        "e_hi": e_hi,
        "auroc_flat_vs_dr": area,
        "val_r2_mean": r2,
        "flat_quantiles": {q: torch.quantile(flat_means.float(), q).item() for q in (0.5, 0.9, 0.95, 0.99)},
        "dr_quantiles": {q: torch.quantile(dr_means.float(), q).item() for q in (0.5, 0.75, 0.9, 0.95)},
        "flat_alpha_mean": ((flat_means - e_lo) / (e_hi - e_lo)).clamp(0, 1).mean().item(),
        "dr_alpha_mean": ((dr_means - e_lo) / (e_hi - e_lo)).clamp(0, 1).mean().item(),
        "dr_by_friction": by_friction,
        "gate_G3": gate,
        "gate_G3_passed": all(gate.values()),
    }

    # store the calibration inside the model file used at run time
    raw["state_dict"]["e_lo"] = torch.tensor(e_lo)
    raw["state_dict"]["e_hi"] = torch.tensor(e_hi)
    raw["info"]["calibration"] = {k: report[k] for k in ("e_lo", "e_hi", "auroc_flat_vs_dr", "window")}
    torch.save(raw, path)

    out_dir = os.path.join(common.RESULTS_DIR, "dynamics")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"calibration_seed{args.seed}.json"), "w") as f:
        json.dump(report, f, indent=1)
    torch.save({"flat": flat_means, "dr": dr_means, "dr_friction": friction}, os.path.join(out_dir, f"window_errors_seed{args.seed}.pt"))
    print(f"[RESULT] seed {args.seed}: e_lo={e_lo:.4f} e_hi={e_hi:.4f} AUROC={area:.3f} R2={r2:.3f} G3={'PASS' if report['gate_G3_passed'] else 'FAIL'}")
    for row in by_friction:
        print(f"  friction {row['friction']}: median error {row['median_error']:.4f}, mean alpha {row['mean_alpha']:.3f}")


if __name__ == "__main__":
    main()
