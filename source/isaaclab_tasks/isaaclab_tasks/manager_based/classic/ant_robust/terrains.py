# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Terrains for the HW1 Ant tasks.

* ``dr_terrain``: the domain-randomized training terrain (B, C, D, E0, E and stage 1 of E2 are trained on it).
* ``dr_hard_terrain``: the same terrain types with about twice the height range (stage 2 of E2).
* ``t1_terrain`` .. ``t5_terrain``: held-out test terrains (never used for training).
* ``oracle_terrain``: the shapes of T1 and T4 on other random tiles, only for the ceiling reference of E2.
* ``grid_terrain``: four columns of increasing roughness for the friction x roughness heatmap.
* ``plane_terrain``: a flat plane with a chosen friction (T2 and the friction-switch analysis).

Two constraints of the original Ant task shape these terrains:

* The episode terminates when the torso is below 0.31 m in *world* z (``root_height_below_minimum``), and a trained
  Ant walks with its torso only 0.38-0.50 m high (measured in phase 0). Every terrain therefore keeps the ground at
  or above zero: dips or pits would end episodes even if the robot walks well.
* The Ant runs about 130 m in a 16 s episode, so robots spawn at the start of a strip that is long enough.

Friction is set through the ground material. Its combine mode is ``multiply`` wherever the effective friction
matters: PhysX resolves mixed modes by priority (average < min < multiply < max), so the contact friction becomes
``robot friction x ground friction``. The robot keeps the default friction 1.0 unless an event changes it.
"""

from __future__ import annotations

import numpy as np
import trimesh

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.terrains.trimesh import mesh_terrains
from isaaclab.utils import configclass

##
# Custom sub-terrains (non-negative versions of built-in ones)
##


@height_field_to_mesh
def scaled_random_uniform_terrain(difficulty: float, cfg: ScaledRandomUniformTerrainCfg) -> np.ndarray:
    """Random uniform bumps whose maximum height grows linearly with the difficulty.

    The built-in random uniform terrain ignores ``difficulty``. Here the height noise is drawn from ``(0, h)`` where
    ``h`` is interpolated within ``cfg.max_height_range``. The spline upsampling can undershoot slightly below zero,
    so heights are clipped at zero.
    """
    low, high = cfg.max_height_range
    max_height = low + difficulty * (high - low)
    # __wrapped__ is the height-field function before the mesh conversion
    heights = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg.replace(noise_range=(0.0, max_height)))
    return np.clip(heights, 0, None)


@configclass
class ScaledRandomUniformTerrainCfg(terrain_gen.HfRandomUniformTerrainCfg):
    """Configuration for :func:`scaled_random_uniform_terrain`."""

    function = scaled_random_uniform_terrain

    max_height_range: tuple[float, float] = (0.0, 0.1)
    """Maximum bump height (m) at difficulty 0 and 1."""

    noise_range: tuple[float, float] = (0.0, 0.1)
    """Overwritten from :attr:`max_height_range` when the terrain is generated."""

    noise_step: float = 0.005
    downsampled_scale: float | None = 0.2
    border_width: float = 0.25


@height_field_to_mesh
def raised_wave_terrain(difficulty: float, cfg: RaisedWaveTerrainCfg) -> np.ndarray:
    """Built-in sinusoidal waves shifted up so that the troughs touch zero (heights in ``[0, 2 * amplitude]``)."""
    heights = hf_terrains.wave_terrain.__wrapped__(difficulty, cfg)
    return heights - heights.min()


@configclass
class RaisedWaveTerrainCfg(terrain_gen.HfWaveTerrainCfg):
    """Configuration for :func:`raised_wave_terrain`."""

    function = raised_wave_terrain


def raised_random_grid_terrain(difficulty: float, cfg: RaisedRandomGridTerrainCfg) -> tuple[list, np.ndarray]:
    """Built-in random box grid lifted by the grid height so that no box top is below zero.

    The built-in grid moves every box top by ``U(-h, h)``; after the lift the tops lie in ``[0, 2h]``. The tile border
    is lifted too, which leaves a step of ``h`` at the tile edge.
    """
    meshes, origin = mesh_terrains.random_grid_terrain(difficulty, cfg)
    low, high = cfg.grid_height_range
    lift = low + difficulty * (high - low)
    shift = trimesh.transformations.translation_matrix((0.0, 0.0, lift))
    for mesh in meshes:
        mesh.apply_transform(shift)
    return meshes, origin + np.array([0.0, 0.0, lift])


@configclass
class RaisedRandomGridTerrainCfg(terrain_gen.MeshRandomGridTerrainCfg):
    """Configuration for :func:`raised_random_grid_terrain`."""

    function = raised_random_grid_terrain


##
# Ground materials
##


def ground_material(friction: float = 1.0, combine_mode: str = "average") -> sim_utils.RigidBodyMaterialCfg:
    """Ground physics material with equal static and dynamic friction (restitution as in the original task)."""
    return sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode=combine_mode,
        restitution_combine_mode="average",
        static_friction=friction,
        dynamic_friction=friction,
        restitution=0.0,
    )


GROUND_VISUAL = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.44, 0.47), roughness=0.9)
"""Neutral gray for generated terrains so that the terrain shape is visible in videos."""

##
# Training terrain (B, C, D)
##

DR_TERRAIN_GENERATOR = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=24,  # 192 m along +x
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    curriculum=False,  # every tile gets a random type and a random difficulty
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
        "rough": ScaledRandomUniformTerrainCfg(proportion=0.30, max_height_range=(0.01, 0.08)),
        # heights in [0, 2 * amplitude], i.e. up to 0.08 m like the bumps
        "wave": RaisedWaveTerrainCfg(proportion=0.20, amplitude_range=(0.01, 0.04), num_waves=4, border_width=0.25),
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.15,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.3, 0.8),
            obstacle_height_range=(0.02, 0.10),
            num_obstacles=30,
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.15), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Training terrain. Stairs and box grids are held out for the test terrain T1."""

DR_SPAWN_ROWS = 4
"""Robots spawn on the first rows only, leaving at least 164 m (+20 m border) ahead."""

DR_GROUND_FRICTION = 1.0
"""Ground friction for training. With ``multiply``, the randomized robot friction is the effective friction."""


def training_terrain(generator: TerrainGeneratorCfg) -> TerrainImporterCfg:
    """Training terrain: spawn on the first rows, ground friction 1.0 with ``multiply`` (robot friction decides)."""
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=generator,
        max_init_terrain_level=DR_SPAWN_ROWS - 1,
        collision_group=-1,
        physics_material=ground_material(DR_GROUND_FRICTION, "multiply"),
        visual_material=GROUND_VISUAL,
        debug_vis=False,
    )


def dr_terrain() -> TerrainImporterCfg:
    return training_terrain(DR_TERRAIN_GENERATOR)


DR_HARD_TERRAIN_GENERATOR = DR_TERRAIN_GENERATOR.replace(
    seed=5,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
        "rough": ScaledRandomUniformTerrainCfg(proportion=0.30, max_height_range=(0.01, 0.12)),
        "wave": RaisedWaveTerrainCfg(proportion=0.20, amplitude_range=(0.01, 0.06), num_waves=4, border_width=0.25),
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.15,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.3, 0.8),
            obstacle_height_range=(0.02, 0.20),
            num_obstacles=30,
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.30), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Stage 2 of E2: the training terrain types with a larger height range (bumps and waves up to 0.12 m, blocks up to
0.20 m, slopes up to 0.30). Bumps stay below the 0.13-0.16 m of T1, and no test shape (stairs, box grid, raised box,
rails, cylinders, cones, tilted boxes) is added."""


def dr_hard_terrain() -> TerrainImporterCfg:
    return training_terrain(DR_HARD_TERRAIN_GENERATOR)


##
# Held-out test terrains
##

T1_DIFFICULTY_RANGE = (0.5, 1.0)
T1_HEIGHT_SCALE = 1.0
T2_FRICTION = 0.2
T3_FRICTION = 0.4


def t1_terrain_generator(
    difficulty_range: tuple[float, float] = T1_DIFFICULTY_RANGE, height_scale: float = T1_HEIGHT_SCALE
) -> TerrainGeneratorCfg:
    """Shapes never seen in training: pyramid stairs, random box grids and rougher bumps."""
    s = height_scale
    return TerrainGeneratorCfg(
        seed=1,
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=25,  # 200 m along +x
        num_cols=10,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=difficulty_range,
        curriculum=False,
        use_cache=False,
        sub_terrains={
            "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=1.0 / 3.0,
                step_height_range=(0.04 * s, 0.10 * s),
                step_width=0.4,
                platform_width=2.0,
                border_width=1.0,
            ),
            # box tops in [0, 2 * h], h = 0.02 -> 0.06 m
            "boxes": RaisedRandomGridTerrainCfg(
                proportion=1.0 / 3.0, grid_width=0.45, grid_height_range=(0.02 * s, 0.06 * s), platform_width=2.0
            ),
            "rough_high": ScaledRandomUniformTerrainCfg(
                proportion=1.0 / 3.0, max_height_range=(0.10 * s, 0.16 * s)
            ),
        },
    )


def generator_terrain(
    generator: TerrainGeneratorCfg, friction: float = 1.0, combine_mode: str = "average"
) -> TerrainImporterCfg:
    """Generated terrain where every robot spawns on the first row (one column per group of robots)."""
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=generator,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=ground_material(friction, combine_mode),
        visual_material=GROUND_VISUAL,
        debug_vis=False,
    )


def plane_terrain(friction: float = 1.0, combine_mode: str = "multiply") -> TerrainImporterCfg:
    """Infinite flat plane (same layout as the original task) with a chosen ground friction."""
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=ground_material(friction, combine_mode),
        debug_vis=False,
    )


def t1_terrain() -> TerrainImporterCfg:
    """T1 (shape): unseen shapes, original friction setting (1.0, average)."""
    return generator_terrain(t1_terrain_generator(), friction=1.0, combine_mode="average")


def t2_terrain() -> TerrainImporterCfg:
    """T2 (friction): flat ground with an effective friction below the training range."""
    return plane_terrain(friction=T2_FRICTION, combine_mode="multiply")


def t3_terrain() -> TerrainImporterCfg:
    """T3 (compound): T1 shapes with reduced friction."""
    return generator_terrain(t1_terrain_generator(), friction=T3_FRICTION, combine_mode="multiply")


def t4_terrain_generator() -> TerrainGeneratorCfg:
    """Second held-out set, defined after the A-D results but before training E0/E (addendum section 4).

    Shapes never used in training or in the diagnosis: a raised box to step down from (spawn on top), rails to step
    over and scattered cylinders.
    """
    return TerrainGeneratorCfg(
        seed=3,
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=25,
        num_cols=10,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(0.5, 1.0),
        curriculum=False,
        use_cache=False,
        sub_terrains={
            "box": terrain_gen.MeshBoxTerrainCfg(proportion=1.0 / 3.0, box_height_range=(0.15, 0.30), platform_width=2.5),
            "rails": terrain_gen.MeshRailsTerrainCfg(
                proportion=1.0 / 3.0, rail_thickness_range=(0.05, 0.10), rail_height_range=(0.05, 0.12), platform_width=2.0
            ),
            "cylinders": terrain_gen.MeshRepeatedCylindersTerrainCfg(
                proportion=1.0 / 3.0,
                object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                    num_objects=20, height=0.05, radius=0.25
                ),
                object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                    num_objects=40, height=0.12, radius=0.35
                ),
                platform_width=2.0,
            ),
        },
    )


def t4_terrain() -> TerrainImporterCfg:
    """T4 (second held-out set): new shapes, original friction setting (1.0, average)."""
    return generator_terrain(t4_terrain_generator(), friction=1.0, combine_mode="average")


T5_SPAWN_ROWS = 10
"""T5 spreads the 100 robots over 10 rows x 10 columns, one robot per tile (see ``mdp.spread_env_origins``)."""


def t5_terrain_generator() -> TerrainGeneratorCfg:
    """Third held-out set, defined before training E2 (second addendum).

    Shapes never used in training, in the diagnosis or in T1-T4: cones, tilted boxes and long waves twice as high as
    the stage-2 waves.
    """
    cones = terrain_gen.MeshRepeatedPyramidsTerrainCfg
    boxes = terrain_gen.MeshRepeatedBoxesTerrainCfg
    return TerrainGeneratorCfg(
        seed=4,
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=25,
        num_cols=10,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(0.5, 1.0),
        curriculum=False,
        use_cache=False,
        sub_terrains={
            "cones": cones(
                proportion=1.0 / 3.0,
                object_params_start=cones.ObjectCfg(num_objects=20, height=0.08, radius=0.40),
                object_params_end=cones.ObjectCfg(num_objects=40, height=0.18, radius=0.50),
                platform_width=2.0,
            ),
            "tilted_boxes": boxes(
                proportion=1.0 / 3.0,
                object_params_start=boxes.ObjectCfg(num_objects=20, height=0.06, size=(0.5, 0.5), max_yx_angle=10.0),
                object_params_end=boxes.ObjectCfg(num_objects=40, height=0.14, size=(0.7, 0.7), max_yx_angle=25.0),
                platform_width=2.0,
            ),
            # heights in [0, 2 * amplitude] = up to 0.2 m, wavelength 4 m
            "long_waves": RaisedWaveTerrainCfg(
                proportion=1.0 / 3.0, amplitude_range=(0.04, 0.10), num_waves=2, border_width=0.25
            ),
        },
    )


def t5_terrain() -> TerrainImporterCfg:
    """T5 (third held-out set): new shapes, original friction setting (1.0, average)."""
    return generator_terrain(t5_terrain_generator(), friction=1.0, combine_mode="average")


##
# Ceiling reference of E2 (never a submission candidate)
##

ORACLE_FRICTION_RANGE = (0.15, 1.2)
"""Robot friction range of the ceiling reference: covers the 0.2 of T2 and the 0.4 of T3."""


def oracle_terrain() -> TerrainImporterCfg:
    """The shapes of T1 and T4 (other random tiles, seed 7, difficulty 0-1) plus flat ground.

    Training on it answers "how high can this robot score on T1-T4 at all", so that the score of E2 can be judged.
    """
    sub_terrains = {"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10)}
    for name, cfg in {**t1_terrain_generator().sub_terrains, **t4_terrain_generator().sub_terrains}.items():
        sub_terrains[name] = cfg.replace(proportion=0.15)
    return training_terrain(DR_TERRAIN_GENERATOR.replace(seed=7, sub_terrains=sub_terrains))


##
# Analysis terrains
##

GRID_MAX_HEIGHTS = (0.0, 0.05, 0.10, 0.15)
"""Maximum bump height of each column of the heatmap terrain (one column per value)."""


def grid_terrain_generator() -> TerrainGeneratorCfg:
    """Columns of fixed roughness. With ``curriculum=True`` the columns follow the order of ``sub_terrains``."""
    sub_terrains = {}
    for height in GRID_MAX_HEIGHTS:
        name = f"h{round(height * 100):03d}"
        if height == 0.0:
            sub_terrains[name] = terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)
        else:
            sub_terrains[name] = ScaledRandomUniformTerrainCfg(proportion=1.0, max_height_range=(height, height))
    return TerrainGeneratorCfg(
        seed=2,
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=25,
        num_cols=len(GRID_MAX_HEIGHTS),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        curriculum=True,
        use_cache=False,
        sub_terrains=sub_terrains,
    )


def grid_terrain() -> TerrainImporterCfg:
    """Heatmap terrain. Set the friction per run, e.g. ``env.scene.terrain.physics_material.static_friction=0.3``."""
    return generator_terrain(grid_terrain_generator(), friction=1.0, combine_mode="multiply")


def switch_terrain() -> TerrainImporterCfg:
    """Flat plane whose effective friction equals the robot friction (changed mid-episode by ``eval.py``)."""
    return plane_terrain(friction=1.0, combine_mode="multiply")
