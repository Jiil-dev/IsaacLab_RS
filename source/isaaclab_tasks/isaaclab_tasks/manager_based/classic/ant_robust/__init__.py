# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""HW1: an Ant that keeps walking on unseen terrain.

Task ids (all share the rewards, terminations and robot of Isaac-Ant-v0):

* ``Isaac-Ant-DR-v0``: B, training with domain-randomized terrain and friction.
* ``Isaac-Ant-Hist-DR-v0`` / ``Isaac-Ant-Hist-v0``: C (history observation), training / original scene.
* ``Isaac-Ant-Residual-DR-v0`` / ``Isaac-Ant-Residual-v0``: D (residual correction), training / original scene.
* ``Isaac-Ant-RelHeight-*`` / ``Isaac-Ant-Scan-*``: E0 and E (terrain sensing), first exploratory round.
* ``Isaac-Ant-WideScan-DR-v0`` / ``-DRHard-v0`` / ``-v0``: E2 stage 1 / stage 2 / original scene, second exploratory
  round; ``Isaac-Ant-WideScan-Oracle-v0`` is its ceiling reference (trained on the test shapes, never submitted).
* ``Isaac-Ant[-<obs>]-{T1,T2,T3,T4,T5,Grid,Switch}-v0``: held-out test and analysis terrains.
"""

import gymnasium as gym

from . import agents, env_cfgs

_ENV_CFGS = env_cfgs.__name__
_AGENT_CFG = f"{agents.__name__}.rsl_rl_ppo_cfg:AntRobustPPORunnerCfg"
_RESIDUAL_AGENT_CFG = f"{agents.__name__}.rsl_rl_ppo_cfg:AntResidualPPORunnerCfg"
_E2_AGENT_CFG = f"{agents.__name__}.rsl_rl_ppo_cfg:AntE2PPORunnerCfg"


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

# exploratory conditions (terrain sensing): E0 and E
_register("Isaac-Ant-RelHeight-DR-v0", f"{_ENV_CFGS}:AntRelHeightDREnvCfg")
_register("Isaac-Ant-Scan-DR-v0", f"{_ENV_CFGS}:AntScanDREnvCfg")

# second exploratory round: E2 (two training stages) and its ceiling reference
_register("Isaac-Ant-WideScan-DR-v0", f"{_ENV_CFGS}:AntWideScanDREnvCfg", _E2_AGENT_CFG)
_register("Isaac-Ant-WideScan-DRHard-v0", f"{_ENV_CFGS}:AntWideScanDRHardEnvCfg", _E2_AGENT_CFG)
_register("Isaac-Ant-WideScan-Oracle-v0", f"{_ENV_CFGS}:AntWideScanOracleEnvCfg", _E2_AGENT_CFG)

# evaluation on the scene of the original task (follows any terrain change made in ant/ant_env_cfg.py)
_register("Isaac-Ant-Hist-v0", f"{_ENV_CFGS}:AntHistEnvCfg")
_register("Isaac-Ant-Residual-v0", f"{_ENV_CFGS}:AntResidualEnvCfg")
_register("Isaac-Ant-RelHeight-v0", f"{_ENV_CFGS}:AntRelHeightEnvCfg")
_register("Isaac-Ant-Scan-v0", f"{_ENV_CFGS}:AntScanEnvCfg")
_register("Isaac-Ant-WideScan-v0", f"{_ENV_CFGS}:AntWideScanEnvCfg", _E2_AGENT_CFG)

# held-out test and analysis terrains
for _task_id, _cfg_cls in env_cfgs.TEST_TASK_CFGS.items():
    _register(_task_id, _cfg_cls)
