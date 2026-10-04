# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Figures of the HW1 report (runs without Isaac Sim). Figures whose data is missing are skipped.

    python assignments/hw1_ant/scripts/sweep_eval.py aggregate
    python assignments/hw1_ant/scripts/plots.py

Colors: one fixed categorical slot per condition (validated palette, adjacent pairs), sequential blue for magnitude,
blue-gray-red for differences. Every multi-series figure has a legend; the numbers are in results/summary.md.
"""

import csv
import glob
import json
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import hw1_common as common  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CONDITIONS = ("A", "B", "C", "D", "E0", "E")
COLORS = {
    "A": "#2a78d6",
    "B": "#eb6834",
    "C": "#1baf7a",
    "D": "#eda100",
    "B2000": "#e87ba4",
    "E0": "#008300",
    "E": "#4a3aa7",
}
LABELS = {
    "A_ref": "A_ref: given baseline (1000 it)",
    "A": "A: flat only",
    "B": "B: + terrain & friction DR",
    "C": "C: B + history input",
    "D": "D: residual on B@2000 (ours)",
    "B2000": "B@2000: frozen base of D",
    "E0": "E0: B + height above ground (exploratory)",
    "E": "E: E0 + height scan (exploratory)",
}
SHORT_LABELS = {
    "A": "A: flat only",
    "B": "B: + DR",
    "C": "C: + history",
    "D": "D: residual",
    "B2000": "B@2000",
    "E0": "E0: + rel. height",
    "E": "E: + height scan",
}
ENVS = ("Flat", "T1", "T2", "T3", "T4")
ENV_LABELS = {
    "Flat": "Flat\n(seen)",
    "T1": "T1\nshapes",
    "T2": "T2\nμ 0.2",
    "T3": "T3\nshapes, μ 0.4",
    "T4": "T4\nnew shapes",
}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#c13434", "#ef9a99", "#f0efec", "#86b6ef", "#1c5cab"])

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "legend.labelcolor": INK2,
})


def _style(ax, grid_axis: str = "y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _save(fig, name: str):
    os.makedirs(common.FIGURES_DIR, exist_ok=True)
    path = os.path.join(common.FIGURES_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] {os.path.relpath(path, common.REPO_ROOT)}")


def _read_csv(name: str) -> list[dict]:
    path = os.path.join(common.RESULTS_DIR, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _seed_stats(rows: list[dict], cond: str, env: str, key: str) -> tuple[float, float] | None:
    vals = [float(r[key]) for r in rows if r["condition"] == cond and r["env"] == env and r[key] not in ("", "None")]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def fig_main(rows: list[dict]):
    """Reward and fall rate per environment (two panels, one scale each)."""
    conds = [c for c in CONDITIONS if any(r["condition"] == c for r in rows)]
    if not conds:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    highlight = "E" if "E" in conds else "D"
    width = 0.8 / len(conds)
    x = np.arange(len(ENVS))
    for ax, key, title in ((axes[0], "reward_mean", "Episode reward (official metric)"),
                           (axes[1], "fall_rate", "Fall rate")):
        for i, cond in enumerate(conds):
            stats = [_seed_stats(rows, cond, env, key) or (np.nan, 0.0) for env in ENVS]
            means = [s[0] for s in stats]
            stds = [s[1] for s in stats]
            pos = x - 0.4 + width * (i + 0.5)
            ax.bar(pos, means, width, color=COLORS[cond], edgecolor=SURFACE, linewidth=2, label=LABELS[cond])
            ax.errorbar(pos, means, yerr=stds, fmt="none", ecolor=INK2, elinewidth=1, capsize=0)
            if key == "reward_mean" and cond == highlight:
                # label one condition only (the newest method), above its error bar
                for p, m, s in zip(pos, means, stds):
                    if np.isfinite(m):
                        ax.annotate(f"{m:.0f}", (p, m + s), xytext=(0, 3), textcoords="offset points",
                                    ha="center", va="bottom", fontsize=8, color=INK)
        ref = [_seed_stats(rows, "A_ref", env, key) for env in ENVS]
        for xi, r in zip(x, ref):
            if r:
                ax.hlines(r[0], xi - 0.42, xi + 0.42, color=MUTED, linewidth=2, label=LABELS["A_ref"] if xi == 0 else None)
        # the narrow right panel only gets the short names
        ax.set_xticks(x, [ENV_LABELS[e] if key == "reward_mean" else e for e in ENVS])
        ax.set_title(title, loc="left")
        _style(ax)
    axes[1].set_ylim(0, 1)
    axes[0].set_ylabel("reward (mean over 3 seeds, bar = seed std)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.06), fontsize=9)
    _save(fig, "main_comparison.png")


def fig_d_vs_base(rows: list[dict]):
    """Effect of the correction itself: D against its own frozen base B@2000 and against B@3000."""
    if not any(r["condition"] == "D" for r in rows):
        return
    conds = ["B2000", "B", "D"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.8 / len(conds)
    x = np.arange(len(ENVS))
    for i, cond in enumerate(conds):
        stats = [_seed_stats(rows, cond, env, "reward_mean") or (np.nan, 0.0) for env in ENVS]
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, [s[0] for s in stats], width, color=COLORS[cond], edgecolor=SURFACE, linewidth=2, label=LABELS[cond])
        ax.errorbar(pos, [s[0] for s in stats], yerr=[s[1] for s in stats], fmt="none", ecolor=INK2, elinewidth=1)
    ax.set_xticks(x, [ENV_LABELS[e] for e in ENVS])
    ax.set_ylabel("reward")
    ax.set_title("Same budget split: B@2000 (D's base), B@3000, D = B@2000 + 1000 it residual", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    _style(ax)
    _save(fig, "d_vs_base.png")


def fig_heatmaps(grid_rows: list[dict]):
    """Friction x roughness: reward per condition, and the difference D - B."""
    conds = [c for c in ("A", "B", "D", "E") if any(r["condition"] == c for r in grid_rows)]
    if not conds:
        return
    best = "E" if "E" in conds else "D"
    frictions = sorted({float(r["friction"]) for r in grid_rows})
    heights = sorted({float(r["max_height"]) for r in grid_rows})

    def matrix(cond):
        m = np.full((len(frictions), len(heights)), np.nan)
        for i, mu in enumerate(frictions):
            for j, h in enumerate(heights):
                vals = [float(r["reward_mean"]) for r in grid_rows
                        if r["condition"] == cond and float(r["friction"]) == mu and float(r["max_height"]) == h]
                if vals:
                    m[i, j] = np.mean(vals)
        return m

    mats = {c: matrix(c) for c in conds}
    vmax = max(np.nanmax(m) for m in mats.values())
    panels = conds + ([f"{best}-B"] if best in mats and "B" in mats else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6), squeeze=False)
    for ax, name in zip(axes[0], panels):
        if name.endswith("-B"):
            m = mats[best] - mats["B"]
            lim = np.nanmax(np.abs(m))
            img = ax.imshow(m, cmap=DIVERGING, vmin=-lim, vmax=lim, origin="lower", aspect="auto")
            ax.set_title(f"{best} - B (blue: {best} better)", loc="left")
        else:
            m = mats[name]
            img = ax.imshow(m, cmap=SEQUENTIAL, vmin=0, vmax=vmax, origin="lower", aspect="auto")
            ax.set_title(SHORT_LABELS[name], loc="left")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isfinite(m[i, j]):
                    rgba = img.cmap(img.norm(m[i, j]))
                    light = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] > 0.55
                    ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=8, color=INK if light else "white")
        ax.set_xticks(range(len(heights)), [f"{h:.2f}" for h in heights])
        ax.set_yticks(range(len(frictions)), [f"{mu:.1f}" for mu in frictions])
        ax.set_xlabel("max bump height (m)")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0][0].set_ylabel("ground friction")
    fig.suptitle("Reward on the friction x roughness grid (200 envs per run, mean of 3 seeds)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    _save(fig, "heatmap_friction_roughness.png")


def fig_alpha():
    """alpha over time when the friction drops (switch at 5 s), and alpha per test environment."""
    traces = {c: [np.load(p) for p in sorted(glob.glob(os.path.join(common.RESULTS_DIR, "raw", "switch", f"{c}_seed*_traces.npz")))]
              for c in ("B", "D")}
    if not traces["D"]:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    dt = float(traces["D"][0]["dt"])

    def mean_series(arrays, key):
        series = []
        for a in arrays:
            vals = np.where(a["active"], a[key], np.nan)
            series.append(np.nanmean(vals, axis=1))
        # drop the final steps: the episode ends at 16 s and the reset zeroes the traced values
        n = min(len(s) for s in series) - 2
        stack = np.stack([s[:n] for s in series])
        return np.arange(n) * dt, np.nanmean(stack, 0), np.nanmin(stack, 0), np.nanmax(stack, 0)

    t, mean, lo, hi = mean_series(traces["D"], "alpha")
    ax = axes[0]
    ax.plot(t, mean, color=COLORS["D"], linewidth=2, label="D: alpha (mean of seeds)")
    ax.fill_between(t, lo, hi, color=COLORS["D"], alpha=0.1, linewidth=0)
    ax.axvline(5.0, color=MUTED, linewidth=1)
    ax.annotate("friction 1.0 -> 0.25", (5.0, 1.0), xytext=(4, -2), textcoords="offset points", fontsize=8, color=INK2, va="top")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("time (s)")
    ax.set_title("Correction strength alpha", loc="left")
    _style(ax)

    ax = axes[1]
    for cond in ("B", "D"):
        if traces[cond]:
            t, mean, lo, hi = mean_series(traces[cond], "vel_x")
            ax.plot(t, mean, color=COLORS[cond], linewidth=2, label=LABELS[cond])
            ax.fill_between(t, lo, hi, color=COLORS[cond], alpha=0.1, linewidth=0)
    ax.axvline(5.0, color=MUTED, linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("forward velocity (m/s)")
    ax.set_title("Forward velocity (traced envs)", loc="left")
    ax.legend(fontsize=8, loc="lower left")
    _style(ax)

    ax = axes[2]
    data = []
    for env in ENVS:
        vals = []
        for path in glob.glob(os.path.join(common.RESULTS_DIR, "raw", "main", f"D_seed*_{env}.json")):
            with open(path) as f:
                vals += [v for v in json.load(f)["per_env"]["alpha_mean"] if v is not None]
        data.append(vals)
    if all(data):
        parts = ax.boxplot(data, widths=0.35, patch_artist=True, showfliers=False,
                           medianprops={"color": INK, "linewidth": 1.5}, whiskerprops={"color": INK2},
                           capprops={"color": INK2})
        for box in parts["boxes"]:
            box.set_facecolor(COLORS["D"])
            box.set_edgecolor(SURFACE)
        ax.set_xticks(range(1, len(ENVS) + 1), list(ENVS), fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_title("Episode-mean alpha of D (300 envs)", loc="left")
        _style(ax)
    _save(fig, "alpha_analysis.png")


def _smooth(values: np.ndarray, window: int = 25) -> np.ndarray:
    """Trailing moving average (keeps the length; the first points average over fewer values)."""
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    counts = np.minimum(np.arange(1, len(values) + 1), window)
    return (cumsum[1:] - cumsum[np.maximum(np.arange(1, len(values) + 1) - window, 0)]) / counts


def fig_training_curves():
    """Training reward per iteration. D is drawn from iteration 2000 because it starts from B@2000."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for cond in CONDITIONS:
        curves = []
        for seed in common.SEEDS:
            try:
                run = common.find_run(f"{cond}_seed{seed}")
            except FileNotFoundError:
                continue
            ea = EventAccumulator(run, size_guidance={"scalars": 0})
            ea.Reload()
            if "Train/mean_reward" not in ea.Tags()["scalars"]:
                continue
            events = ea.Scalars("Train/mean_reward")
            curves.append(np.array([[e.step, e.value] for e in events]))
        if not curves:
            continue
        n = min(len(c) for c in curves)
        # D's first iterations average over very few finished episodes; skip them instead of drawing a fake drop
        start = 10 if cond == "D" else 0
        steps = curves[0][start:n, 0] + (2000 if cond == "D" else 0)
        values = np.stack([_smooth(c[start:n, 1]) for c in curves])
        ax.plot(steps, values.mean(0), color=COLORS[cond], linewidth=2, label=f"{LABELS[cond]} ({len(curves)} seeds)")
        ax.fill_between(steps, values.min(0), values.max(0), color=COLORS[cond], alpha=0.1, linewidth=0)
    ax.axvline(2000, color=MUTED, linewidth=1)
    ax.set_xlabel("PPO iteration (D: B's 2000 + its own 1000)")
    ax.set_ylabel("mean training reward")
    ax.set_title("Training curves on each condition's own training environment", loc="left")
    ax.legend(fontsize=8, loc="lower right")
    _style(ax)
    _save(fig, "training_curves.png")


def fig_calibration(seed: int = 42):
    """Window-mean prediction error on nominal ground vs the DR training environment, with the alpha thresholds."""
    path = os.path.join(common.RESULTS_DIR, "dynamics", f"window_errors_seed{seed}.pt")
    cal_path = os.path.join(common.RESULTS_DIR, "dynamics", f"calibration_seed{seed}.json")
    if not (os.path.exists(path) and os.path.exists(cal_path)):
        return
    data = torch.load(path, weights_only=False)
    with open(cal_path) as f:
        cal = json.load(f)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    flat, dr = data["flat"].numpy(), data["dr"].numpy()
    bins = np.logspace(np.log10(max(min(flat.min(), dr.min()), 1e-4)), np.log10(max(flat.max(), dr.max())), 80)
    ax.hist(flat, bins=bins, color=COLORS["A"], alpha=0.85, label="nominal ground (flat, friction 1.0)", density=True)
    ax.hist(dr, bins=bins, color=COLORS["B"], alpha=0.6, label="DR training environment", density=True)
    top = ax.get_ylim()[1]
    for value, name, dy, ha, dx in ((cal["e_lo"], "e_lo: alpha = 0", -12, "right", -3), (cal["e_hi"], "e_hi: alpha = 1", -26, "left", 3)):
        ax.axvline(value, color=INK2, linewidth=1)
        ax.annotate(name, (value, top), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK2, ha=ha)
    ax.set_xscale("log")
    ax.set_xlabel(f"window-mean prediction error (normalized MSE, {cal['window']}-step window)")
    ax.set_ylabel("density")
    ax.set_title(f"Dynamics-model error separates the environments (seed {seed}, AUROC {cal['auroc_flat_vs_dr']:.2f})", loc="left")
    ax.legend(fontsize=8)
    _style(ax, grid_axis="y")
    _save(fig, f"calibration_seed{seed}.png")


def fig_spawn_height():
    """T1 diagnosis: mean walking distance of each terrain column against the column's spawn height."""
    path = os.path.join(common.RESULTS_DIR, "diagnosis", "spawn_T1.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        spawn = json.load(f)
    z, col = np.array(spawn["origin_z"]), np.array(spawn["column"])
    conds = [c for c in ("A", "B", "D", "E0", "E")
             if glob.glob(os.path.join(common.RESULTS_DIR, "raw", "main", f"{c}_seed*_T1.json"))]
    fig, axes = plt.subplots(1, len(conds), figsize=(3.1 * len(conds), 3.3), sharey=True, squeeze=False)
    for ax, cond in zip(axes[0], conds):
        dist = []
        for p in sorted(glob.glob(os.path.join(common.RESULTS_DIR, "raw", "main", f"{cond}_seed*_T1.json"))):
            with open(p) as f:
                dist.append(np.array(json.load(f)["per_env"]["distance_m"]))
        dist = np.mean(dist, axis=0)
        cz = np.array([z[col == c].mean() for c in np.unique(col)])
        cd = np.array([dist[col == c].mean() for c in np.unique(col)])
        order = np.argsort(cz)
        ax.plot(cz[order], cd[order], color=COLORS[cond], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.set_title(SHORT_LABELS[cond], loc="left")
        ax.set_xlabel("spawn height (m)")
        _style(ax)
    axes[0][0].set_ylabel("distance walked (m)")
    fig.suptitle("T1: distance walked against spawn height (terrain-column means, 3 seeds)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    _save(fig, "t1_spawn_height.png")


def main():
    rows = _read_csv("main_runs.csv")
    if rows:
        fig_main(rows)
        fig_d_vs_base(rows)
    grid_rows = _read_csv("grid_runs.csv")
    if grid_rows:
        fig_heatmaps(grid_rows)
    fig_alpha()
    fig_spawn_height()
    fig_training_curves()
    for seed in common.SEEDS:
        fig_calibration(seed)


if __name__ == "__main__":
    main()
