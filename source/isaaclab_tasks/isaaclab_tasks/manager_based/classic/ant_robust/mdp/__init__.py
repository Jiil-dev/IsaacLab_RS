# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms of the HW1 Ant tasks, on top of the terms of the original Ant task."""

from isaaclab_tasks.manager_based.classic.humanoid.mdp import *  # noqa: F401, F403

from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .residual_action import *  # noqa: F401, F403
