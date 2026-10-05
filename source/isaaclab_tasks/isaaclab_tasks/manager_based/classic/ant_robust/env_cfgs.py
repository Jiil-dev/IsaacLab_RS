# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configurations of the HW1 Ant tasks.

Everything inherits from the original ``AntEnvCfg`` (``ant/ant_env_cfg.py``), which is not modified. Rewards,
terminations, episode length and the robot are therefore identical to Isaac-Ant-v0. The configurations only change:

* the observation (C: + command-response history, D: + history and the base policy's action),
* the action (D: residual correction on top of a frozen base policy),
* the terrain and the friction (training: domain randomization; test: held-out terrains).

Evaluation configurations without a terrain change (``AntHistEnvCfg``, ``AntResidualEnvCfg``) keep the scene of the
original task, so a terrain edit in ``ant/ant_env_cfg.py`` also applies to them.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg, EventCfg, MySceneCfg, ObservationsCfg

from . import mdp, terrains

HISTORY_LENGTH = 15
"""Steps of command-response history in the observation (15 x 1/60 s = 0.25 s)."""

DR_FRICTION_RANGE = (0.3, 1.2)
"""Effective friction range of the training environments."""

##
# Observations
##


@configclass
class HistObservationsCfg:
    """Observations of condition C."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Original 60-dim observation followed by 15 steps of response (22) and command (8) history: 510."""

        proprio_hist = ObsTerm(func=mdp.proprio_state, history_length=HISTORY_LENGTH)
        action_hist = ObsTerm(func=mdp.last_action, history_length=HISTORY_LENGTH)

    policy: PolicyCfg = PolicyCfg()


@configclass
class ResidualObservationsCfg:
    """Observations of condition D."""

    @configclass
    class PolicyCfg(HistObservationsCfg.PolicyCfg):
        """Observation of C followed by the action the frozen base policy proposes: 518."""

        base_action = ObsTerm(func=mdp.base_action)

    policy: PolicyCfg = PolicyCfg()


##
# Terrain sensing (exploratory conditions E0 and E, see lab notes addendum)
##

HEIGHT_PROBE = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.0, 0.0)),
    mesh_prim_paths=["/World/ground"],
    max_distance=100.0,
)
"""One ray straight down from above the torso: height of the torso above the ground below it."""

HEIGHT_SCAN = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso",
    offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(1.6, 1.0)),
    mesh_prim_paths=["/World/ground"],
    max_distance=100.0,
)
"""9 x 6 rays every 0.2 m, from 0.4 m behind to 1.2 m ahead of the torso and 0.5 m to each side (yaw aligned)."""


HEIGHT_SCAN_WIDE = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso",
    offset=RayCasterCfg.OffsetCfg(pos=(1.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.25, size=(4.0, 2.5)),
    mesh_prim_paths=["/World/ground"],
    max_distance=100.0,
)
"""17 x 11 rays every 0.25 m, from 1.0 m behind to 3.0 m ahead of the torso and 1.25 m to each side (yaw aligned).

Sized for the Ant (E2): its feet touch down 0.3-0.95 m to the side and up to 0.9 m ahead of and behind the torso
(measured with E, which saw only 11-16 % of these points), and 3 m ahead give about 1 s of preview at 2.5 m/s.
"""


@configclass
class ProbeSceneCfg(MySceneCfg):
    """Original scene (terrain, robot, light) plus the downward height probe."""

    height_probe = HEIGHT_PROBE


@configclass
class ScanSceneCfg(ProbeSceneCfg):
    """Original scene plus the height probe and the height scan."""

    height_scan = HEIGHT_SCAN


@configclass
class WideScanSceneCfg(ProbeSceneCfg):
    """Original scene plus the height probe and the wide height scan (E2)."""

    height_scan = HEIGHT_SCAN_WIDE


@configclass
class RelHeightObservationsCfg:
    """Observations of E0."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Original 60 dims, with the world-z torso height replaced by the height above the ground below."""

        base_height = ObsTerm(func=mdp.ground_relative_heights, params={"sensor_cfg": SceneEntityCfg("height_probe")})

    policy: PolicyCfg = PolicyCfg()


@configclass
class ScanObservationsCfg:
    """Observations of E."""

    @configclass
    class PolicyCfg(RelHeightObservationsCfg.PolicyCfg):
        """Observation of E0 followed by the 9 x 6 height scan: 114 dims."""

        height_scan = ObsTerm(func=mdp.ground_relative_heights, params={"sensor_cfg": SceneEntityCfg("height_scan")})

    policy: PolicyCfg = PolicyCfg()


##
# Actions
##


@configclass
class ResidualActionsCfg:
    """Action of condition D: correction on top of a frozen base policy (same term name as the original)."""

    joint_effort = mdp.ResidualJointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=7.5)


##
# Events
##


@configclass
class DREventCfg(EventCfg):
    """Training events: the original reset events plus spawn spread and per-environment friction."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}, "velocity_range": {}},
    )

    robot_friction = EventTerm(
        func=mdp.randomize_robot_friction,
        mode="startup",
        params={"friction_range": DR_FRICTION_RANGE, "num_buckets": 64},
    )


@configclass
class OracleEventCfg(DREventCfg):
    """Ceiling reference of E2: a friction range that also covers T2 (0.2) and T3 (0.4)."""

    robot_friction = EventTerm(
        func=mdp.randomize_robot_friction,
        mode="startup",
        params={"friction_range": terrains.ORACLE_FRICTION_RANGE, "num_buckets": 64},
    )


@configclass
class T5EventCfg(EventCfg):
    """Original events plus one spawn tile per robot (T5, T6 and the in-domain obstacle test)."""

    spread_origins = EventTerm(
        func=mdp.spread_env_origins, mode="startup", params={"spawn_rows": terrains.T5_SPAWN_ROWS}
    )


##
# Modifiers
##


def use_dr_training(cfg: AntEnvCfg):
    """Domain-randomized training: random terrain tiles and per-environment friction."""
    cfg.scene.terrain = terrains.dr_terrain()
    cfg.events = DREventCfg()
    # many more contacts on uneven ground than on a plane
    cfg.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


def use_dr_hard_training(cfg: AntEnvCfg):
    """Stage 2 of E2: like ``use_dr_training`` with the higher terrain."""
    use_dr_training(cfg)
    cfg.scene.terrain = terrains.dr_hard_terrain()


def use_dr_harder_training(cfg: AntEnvCfg):
    """Stage 3 (E3): like ``use_dr_training`` with higher blocks, steeper slopes and harder tiles."""
    use_dr_training(cfg)
    cfg.scene.terrain = terrains.dr_harder_terrain()


def use_obstacles_hard_training(cfg: AntEnvCfg):
    """Obstacle specialist (third addendum): like ``use_dr_training`` with only the stage-2 blocks."""
    use_dr_training(cfg)
    cfg.scene.terrain = terrains.obstacles_hard_terrain()


def use_oracle_training(cfg: AntEnvCfg):
    """Ceiling reference of E2: the shapes and frictions of the test terrains T1-T4 (other random tiles)."""
    use_dr_training(cfg)
    cfg.scene.terrain = terrains.oracle_terrain()
    cfg.events = OracleEventCfg()


def use_t1_terrain(cfg: AntEnvCfg):
    cfg.scene.terrain = terrains.t1_terrain()


def use_t2_terrain(cfg: AntEnvCfg):
    cfg.scene.terrain = terrains.t2_terrain()


def use_t3_terrain(cfg: AntEnvCfg):
    cfg.scene.terrain = terrains.t3_terrain()


def use_t4_terrain(cfg: AntEnvCfg):
    cfg.scene.terrain = terrains.t4_terrain()


def use_t5_terrain(cfg: AntEnvCfg):
    cfg.scene.terrain = terrains.t5_terrain()
    cfg.events = T5EventCfg()


def use_t6_terrain(cfg: AntEnvCfg):
    cfg.scene.terrain = terrains.t6_terrain()
    cfg.events = T5EventCfg()


def use_obstacles_test_terrain(cfg: AntEnvCfg):
    cfg.scene.terrain = terrains.obstacles_test_terrain()
    cfg.events = T5EventCfg()


def use_grid_terrain(cfg: AntEnvCfg):
    cfg.scene.terrain = terrains.grid_terrain()
    cfg.scene.num_envs = 200


def use_switch_terrain(cfg: AntEnvCfg):
    cfg.scene.terrain = terrains.switch_terrain()


##
# Environment configurations
##


@configclass
class AntHistEnvCfg(AntEnvCfg):
    """C on the original scene (evaluation)."""

    observations: HistObservationsCfg = HistObservationsCfg()


@configclass
class AntResidualEnvCfg(AntEnvCfg):
    """D on the original scene (evaluation, submission)."""

    observations: ResidualObservationsCfg = ResidualObservationsCfg()
    actions: ResidualActionsCfg = ResidualActionsCfg()


@configclass
class AntRelHeightEnvCfg(AntEnvCfg):
    """E0 on the original scene (evaluation): height above the ground instead of world z."""

    # Ray-caster sensors look up every robot on the USD stage, so robots must not be cloned only in Fabric.
    scene: ProbeSceneCfg = ProbeSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    observations: RelHeightObservationsCfg = RelHeightObservationsCfg()


@configclass
class AntScanEnvCfg(AntEnvCfg):
    """E on the original scene (evaluation): height above the ground plus a height scan."""

    # Ray-caster sensors look up every robot on the USD stage, so robots must not be cloned only in Fabric.
    scene: ScanSceneCfg = ScanSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    observations: ScanObservationsCfg = ScanObservationsCfg()


@configclass
class AntWideScanEnvCfg(AntEnvCfg):
    """E2 on the original scene (evaluation): height above the ground plus the wide height scan (60 + 187 dims)."""

    # Ray-caster sensors look up every robot on the USD stage, so robots must not be cloned only in Fabric.
    scene: WideScanSceneCfg = WideScanSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    observations: ScanObservationsCfg = ScanObservationsCfg()


@configclass
class AntWideScanDREnvCfg(AntWideScanEnvCfg):
    """E2 stage 1: the training terrain of B, C, D, E0 and E."""

    def __post_init__(self):
        super().__post_init__()
        use_dr_training(self)


@configclass
class AntWideScanDRHardEnvCfg(AntWideScanEnvCfg):
    """E2 stage 2: the same terrain types with about twice the height range."""

    def __post_init__(self):
        super().__post_init__()
        use_dr_hard_training(self)


@configclass
class AntWideScanDRHarderEnvCfg(AntWideScanEnvCfg):
    """E3 stage 3: higher blocks, steeper slopes and harder tiles than stage 2."""

    def __post_init__(self):
        super().__post_init__()
        use_dr_harder_training(self)


@configclass
class AntWideScanObstHardEnvCfg(AntWideScanEnvCfg):
    """Obstacle specialist (diagnosis, never submitted): only the blocks of the stage-2 terrain."""

    def __post_init__(self):
        super().__post_init__()
        use_obstacles_hard_training(self)


@configclass
class AntWideScanOracleEnvCfg(AntWideScanEnvCfg):
    """Ceiling reference of E2 (never submitted): stage 2 on the shapes and frictions of T1-T4."""

    def __post_init__(self):
        super().__post_init__()
        use_oracle_training(self)


@configclass
class AntRelHeightDREnvCfg(AntRelHeightEnvCfg):
    """E0: domain-randomized training (same terrain, friction and events as B)."""

    def __post_init__(self):
        super().__post_init__()
        use_dr_training(self)


@configclass
class AntScanDREnvCfg(AntScanEnvCfg):
    """E: domain-randomized training (same terrain, friction and events as B)."""

    def __post_init__(self):
        super().__post_init__()
        use_dr_training(self)


@configclass
class AntDREnvCfg(AntEnvCfg):
    """B: original observation and action, domain-randomized training."""

    def __post_init__(self):
        super().__post_init__()
        use_dr_training(self)


@configclass
class AntHistDREnvCfg(AntHistEnvCfg):
    """C: history observation, domain-randomized training."""

    def __post_init__(self):
        super().__post_init__()
        use_dr_training(self)


@configclass
class AntResidualDREnvCfg(AntResidualEnvCfg):
    """D: residual action, domain-randomized training."""

    def __post_init__(self):
        super().__post_init__()
        use_dr_training(self)


def _with_modifier(base_cls: type, modifier, name: str) -> type:
    """Subclass of ``base_cls`` whose ``__post_init__`` additionally applies ``modifier``."""

    @configclass
    class VariantCfg(base_cls):
        def __post_init__(self):
            super().__post_init__()
            modifier(self)

    VariantCfg.__name__ = VariantCfg.__qualname__ = name
    return VariantCfg


# Held-out test and analysis tasks: every observation design (A/B, C, D, E0, E, E2) on every test terrain.
# For example "Isaac-Ant-Residual-T2-v0" is D on the low-friction test terrain.
TEST_TASK_CFGS: dict[str, type] = {}
for _obs_name, _base_cls in {
    "": AntEnvCfg,
    "Hist": AntHistEnvCfg,
    "Residual": AntResidualEnvCfg,
    "RelHeight": AntRelHeightEnvCfg,
    "Scan": AntScanEnvCfg,
    "WideScan": AntWideScanEnvCfg,
}.items():
    for _terrain_name, _modifier in {
        "T1": use_t1_terrain,
        "T2": use_t2_terrain,
        "T3": use_t3_terrain,
        "T4": use_t4_terrain,
        "T5": use_t5_terrain,
        "T6": use_t6_terrain,
        "Obst": use_obstacles_test_terrain,
        "Grid": use_grid_terrain,
        "Switch": use_switch_terrain,
    }.items():
        _task_id = "-".join(filter(None, ["Isaac-Ant", _obs_name, _terrain_name, "v0"]))
        TEST_TASK_CFGS[_task_id] = _with_modifier(_base_cls, _modifier, f"Ant{_obs_name}{_terrain_name}EnvCfg")
