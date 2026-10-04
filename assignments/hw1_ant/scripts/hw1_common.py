# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Paths and helpers shared by the HW1 scripts.

Scripts that do not need the simulator (export, dynamics training, calibration, initialization) load
``ant_robust/mdp/models.py`` directly from its file. Importing it through ``isaaclab_tasks`` would import every task
package, which requires a running Isaac Sim app.
"""

import glob
import importlib.util
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HW1_DIR = os.path.join(REPO_ROOT, "assignments", "hw1_ant")
DATA_DIR = os.path.join(HW1_DIR, "data")
RESULTS_DIR = os.path.join(HW1_DIR, "results")
FIGURES_DIR = os.path.join(HW1_DIR, "figures")
ANT_ROBUST_DIR = os.path.join(
    REPO_ROOT, "source", "isaaclab_tasks", "isaaclab_tasks", "manager_based", "classic", "ant_robust"
)
WEIGHTS_DIR = os.path.join(ANT_ROBUST_DIR, "weights")
LOG_ROOT = os.path.join(REPO_ROOT, "logs", "rsl_rl", "ant_hw1")
# train.py ignores --experiment_name, so condition A (trained on the original task) logs under "ant"
LOG_ROOTS = (LOG_ROOT, os.path.join(REPO_ROOT, "logs", "rsl_rl", "ant"))
SEEDS = (42, 43, 44)


def load_models_module():
    """Import ``ant_robust/mdp/models.py`` without importing ``isaaclab_tasks``."""
    spec = importlib.util.spec_from_file_location("ant_robust_models", os.path.join(ANT_ROBUST_DIR, "mdp", "models.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_run(run_name: str) -> str:
    """Latest run folder ``logs/rsl_rl/{ant_hw1,ant}/<timestamp>_<run_name>``."""
    runs = sorted(
        (path for root in LOG_ROOTS for path in glob.glob(os.path.join(root, f"*_{run_name}"))),
        key=os.path.basename,
    )
    if not runs:
        raise FileNotFoundError(f"No run named '{run_name}' in {LOG_ROOTS}")
    return runs[-1]


def final_checkpoint(run_dir: str) -> str:
    """Checkpoint with the highest iteration in a run folder."""
    checkpoints = glob.glob(os.path.join(run_dir, "model_*.pt"))
    return max(checkpoints, key=lambda p: int(os.path.basename(p)[len("model_") : -len(".pt")]))
