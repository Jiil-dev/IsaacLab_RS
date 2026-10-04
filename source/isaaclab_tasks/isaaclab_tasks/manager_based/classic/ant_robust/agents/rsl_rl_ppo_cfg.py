# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""PPO settings of the HW1 conditions. Everything not listed here equals the Isaac-Ant-v0 baseline."""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg


@configclass
class AntRobustPPORunnerCfg(AntPPORunnerCfg):
    """B and C (and evaluation of every condition): the baseline settings, 3000 iterations."""

    max_iterations = 3000
    experiment_name = "ant_hw1"


@configclass
class AntResidualPPORunnerCfg(AntPPORunnerCfg):
    """D: 1000 iterations on top of B's iteration-2000 policy, small exploration noise and learning rate.

    The actor's last layer starts at zero and the critic starts from B's critic (see ``make_residual_init.py``),
    so training resumes from a checkpoint instead of a random initialization.
    """

    max_iterations = 1000
    experiment_name = "ant_hw1"

    def __post_init__(self):
        self.policy.init_noise_std = 0.2
        self.algorithm.learning_rate = 1.0e-4
