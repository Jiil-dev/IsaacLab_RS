# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Supervised training of the nominal dynamics model of condition D (runs without Isaac Sim).

Data: ``data/flat_noisy_seed<S>.pt`` from ``collect_rollouts.py`` (flat ground, friction 1.0, noisy actions).
Split: by environment (80/20), so validation trajectories are never seen during training.
Output: ``ant_robust/weights/dynamics_seed<S>.pt`` (alpha calibration is added later by ``calibrate_alpha.py``)
and ``results/dynamics/train_seed<S>.json``.

    python assignments/hw1_ant/scripts/train_dynamics.py --seed 42
"""

import argparse
import json
import os
import torch

import hw1_common as common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience in epochs.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    models = common.load_models_module()

    data = torch.load(os.path.join(common.DATA_DIR, f"flat_noisy_seed{args.seed}.pt"), weights_only=False)
    x, y, valid = data["x"], data["y"], data["valid"]  # (T, N, ...)
    num_envs = x.shape[1]
    perm = torch.randperm(num_envs)
    train_envs, val_envs = perm[: int(0.8 * num_envs)], perm[int(0.8 * num_envs) :]

    def flatten(env_ids):
        mask = valid[:, env_ids]
        return x[:, env_ids][mask].float(), y[:, env_ids][mask].float()

    x_train, y_train = flatten(train_envs)
    x_val, y_val = flatten(val_envs)
    print(f"[INFO] train {len(x_train)} / val {len(x_val)} transitions")

    model = models.DynamicsModel(history=3)
    model.in_mean.copy_(x_train.mean(0))
    model.in_std.copy_(x_train.std(0).clamp(min=1e-6))
    model.out_mean.copy_(y_train.mean(0))
    model.out_std.copy_(y_train.std(0).clamp(min=1e-6))
    model.to(args.device)
    x_train, y_train = x_train.to(args.device), y_train.to(args.device)
    x_val, y_val = x_val.to(args.device), y_val.to(args.device)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=args.lr)

    def evaluate():
        model.eval()
        with torch.no_grad():
            errors = torch.cat([
                model.prediction_error(model(x_val[i : i + 65536]), y_val[i : i + 65536])
                for i in range(0, len(x_val), 65536)
            ])
        model.train()
        return errors.mean().item()

    best, best_state, best_epoch, history = float("inf"), None, -1, []
    for epoch in range(args.epochs):
        order = torch.randperm(len(x_train), device=args.device)
        total = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i : i + args.batch_size]
            loss = model.prediction_error(model(x_train[idx]), y_train[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        val = evaluate()
        history.append({"epoch": epoch, "train_mse": total / len(order), "val_mse": val})
        print(f"[EPOCH {epoch:02d}] train {total / len(order):.4f}  val {val:.4f}")
        if val < best - 1e-4:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)

    # per-dimension R^2 on validation data (normalized MSE of a mean predictor is 1 per dimension)
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(x_val[i : i + 65536]) for i in range(0, len(x_val), 65536)])
        target = (y_val - model.out_mean) / model.out_std
        mse = (pred - target).square().mean(0)
        var = target.var(0)
        r2 = (1.0 - mse / var).tolist()
    names = [f"joint_vel_{i}" for i in range(8)] + ["ang_vel_x", "ang_vel_y", "ang_vel_z"]
    report = {
        "seed": args.seed,
        "train_transitions": len(x_train),
        "val_transitions": len(x_val),
        "best_epoch": best_epoch,
        "val_mse_normalized": best,
        "val_r2": dict(zip(names, r2)),
        "val_r2_mean": sum(r2) / len(r2),
        "history": history,
    }
    os.makedirs(common.WEIGHTS_DIR, exist_ok=True)
    out = os.path.join(common.WEIGHTS_DIR, f"dynamics_seed{args.seed}.pt")
    models.save_dynamics(model.cpu(), out, info={"train_report": {k: v for k, v in report.items() if k != "history"}})
    report_dir = os.path.join(common.RESULTS_DIR, "dynamics")
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, f"train_seed{args.seed}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"[RESULT] val normalized MSE {best:.4f}, mean R2 {report['val_r2_mean']:.3f} -> {out}")


if __name__ == "__main__":
    main()
