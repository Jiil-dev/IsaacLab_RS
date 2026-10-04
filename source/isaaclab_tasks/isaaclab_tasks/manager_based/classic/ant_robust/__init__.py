# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""HW1: an Ant that keeps walking on unseen terrain.

Task ids (all share the rewards, terminations and robot of Isaac-Ant-v0):

* ``Isaac-Ant-DR-v0``: B, training with domain-randomized terrain and friction.
* ``Isaac-Ant-Hist-DR-v0`` / ``Isaac-Ant-Hist-v0``: C (history observation), training / original scene.
* ``Isaac-Ant-Residual-DR-v0`` / ``Isaac-Ant-Residual-v0``: D (residual correction), training / original scene.
* ``Isaac-Ant[-Hist|-Residual]-{T1,T2,T3,Grid,Switch}-v0``: held-out test and analysis terrains.
"""

import gymnasium as gym

from . import agents, env_cfgs

_ENV_CFGS = env_cfgs.__name__
_AGENT_CFG = f"{agents.__name__}.rsl_rl_ppo_cfg:AntRobustPPORunnerCfg"
_RESIDUAL_AGENT_CFG = f"{agents.__name__}.rsl_rl_ppo_cfg:AntResidualPPORunnerCfg"


def _register(task_id: str, env_cfg_entry_point, rsl_rl_cfg_entry_point: str = _AGENT_CFG):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": env_cfg_entry_point, "rsl_rl_cfg_entry_point": rsl_rl_cfg_entry_point},
    )


# training
_register("Isaac-Ant-DR-v0", f"{_ENV_CFGS}:AntDREnvCfg")
_register("Isaac-Ant-Hist-DR-v0", f"{_ENV_CFGS}:AntHistDREnvCfg")
_register("Isaac-Ant-Residual-DR-v0", f"{_ENV_CFGS}:AntResidualDREnvCfg", _RESIDUAL_AGENT_CFG)

# evaluation on the scene of the original task (follows any terrain change made in ant/ant_env_cfg.py)
_register("Isaac-Ant-Hist-v0", f"{_ENV_CFGS}:AntHistEnvCfg")
_register("Isaac-Ant-Residual-v0", f"{_ENV_CFGS}:AntResidualEnvCfg")

# held-out test and analysis terrains
for _task_id, _cfg_cls in env_cfgs.TEST_TASK_CFGS.items():
    _register(_task_id, _cfg_cls)
