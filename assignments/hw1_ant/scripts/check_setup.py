# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase-0 setup check for HW1.

Prints the facts the HW1 design relies on:

1. library versions and the task registry,
2. the physics materials of the robot, the terrain and the scene default (friction and combine modes),
3. torso height statistics while a trained policy walks (to bound how deep a terrain may go),
4. resident memory of this Isaac Sim process (to decide how many runs fit in RAM at once).

Example:

    ./isaaclab.sh -p assignments/hw1_ant/scripts/check_setup.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="HW1 phase-0 setup check.")
parser.add_argument("--task", type=str, default="Isaac-Ant-v0", help="Task to inspect.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="logs/rsl_rl/ant/2026-09-17_13-19-56_ant_baseline/model_999.pt",
    help="RSL-RL checkpoint whose actor is rolled out to measure the torso height.",
)
parser.add_argument("--steps", type=int, default=600, help="Number of policy steps to roll out.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import importlib.metadata as metadata
import torch

import isaaclab.sim as sim_utils
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _read_rss_gb() -> tuple[float, float]:
    """Current and peak resident memory of this process in GB."""
    values = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmRSS", "VmHWM")):
                key, value = line.split(":")
                values[key] = int(value.split()[0]) / 1024**2
    return values.get("VmRSS", 0.0), values.get("VmHWM", 0.0)


def _describe_material(prim: Usd.Prim) -> str:
    """Friction, restitution and combine modes of a physics material prim."""
    mat_api = UsdPhysics.MaterialAPI(prim)
    physx_api = PhysxSchema.PhysxMaterialAPI(prim)
    fields = {
        "static": mat_api.GetStaticFrictionAttr().Get(),
        "dynamic": mat_api.GetDynamicFrictionAttr().Get(),
        "restitution": mat_api.GetRestitutionAttr().Get(),
        "friction_combine": physx_api.GetFrictionCombineModeAttr().Get() if physx_api else None,
        "restitution_combine": physx_api.GetRestitutionCombineModeAttr().Get() if physx_api else None,
    }
    return ", ".join(f"{k}={v}" for k, v in fields.items())


def _load_actor(path: str, device: str) -> torch.nn.Sequential:
    """Rebuild the RSL-RL actor MLP (60 -> 400 -> 200 -> 100 -> 8, ELU) from a checkpoint."""
    state_dict = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
    actor_sd = {k.removeprefix("actor."): v for k, v in state_dict.items() if k.startswith("actor.")}
    linear_ids = sorted({int(k.split(".")[0]) for k in actor_sd})
    layers = []
    for i, idx in enumerate(linear_ids):
        weight = actor_sd[f"{idx}.weight"]
        layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
        if i < len(linear_ids) - 1:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).to(device)
    # remap rsl_rl indices (0, 2, 4, 6) to the contiguous Sequential indices
    remapped = {}
    for i, idx in enumerate(linear_ids):
        remapped[f"{2 * i}.weight"] = actor_sd[f"{idx}.weight"]
        remapped[f"{2 * i}.bias"] = actor_sd[f"{idx}.bias"]
    actor.load_state_dict(remapped)
    return actor.eval()


def main():
    # -- 1. versions and registry
    print("=" * 80)
    for pkg in ["isaacsim", "isaaclab", "isaaclab_tasks", "isaaclab_rl", "rsl-rl-lib", "torch"]:
        try:
            print(f"[VERSION] {pkg}: {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            print(f"[VERSION] {pkg}: not found")
    print(f"[CUDA] visible devices: {torch.cuda.device_count()}, current: {torch.cuda.get_device_name(0)}")
    ant_tasks = sorted(t for t in gym.registry.keys() if t.startswith("Isaac-Ant"))
    print(f"[REGISTRY] {len(ant_tasks)} Ant tasks: {ant_tasks}")

    # -- create the environment
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]

    # -- 2. physics materials
    print("=" * 80)
    print(f"[ROBOT] bodies: {robot.body_names}")
    print(f"[ROBOT] joints: {robot.joint_names}")
    materials = robot.root_physx_view.get_material_properties()
    print(f"[ROBOT] material tensor shape (envs, shapes, [static, dynamic, restitution]): {tuple(materials.shape)}")
    print(f"[ROBOT] env 0 per-shape materials:\n{materials[0]}")
    stage = sim_utils.get_current_stage()
    robot_root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    bound = set()
    for prim in Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies()):
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
        if material:
            bound.add(str(material.GetPath()))
    print(f"[ROBOT] physics-purpose material bindings under env_0/Robot: {sorted(bound) if bound else 'none'}")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith("/World/envs/env_") and not path.startswith("/World/envs/env_0/"):
            continue
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            print(f"[MATERIAL] {path}: {_describe_material(prim)}")

    # -- 3. torso height while the trained policy walks
    print("=" * 80)
    actor = _load_actor(args_cli.checkpoint, unwrapped.device)
    obs, _ = env.reset()
    heights, done_count = [], 0
    with torch.inference_mode():
        for step in range(args_cli.steps):
            actions = actor(obs["policy"])
            obs, _, terminated, truncated, _ = env.step(actions)
            done_count += int(terminated.sum().item())
            if step >= 60:
                heights.append(robot.data.root_pos_w[:, 2].clone())
    heights = torch.cat(heights)
    q = torch.quantile(heights, torch.tensor([0.001, 0.01, 0.05, 0.5, 0.95], device=heights.device))
    print(f"[TORSO] height while walking (after 1 s): min={heights.min():.3f} q0.1%={q[0]:.3f} q1%={q[1]:.3f}"
          f" q5%={q[2]:.3f} median={q[3]:.3f} q95%={q[4]:.3f} m")
    print(f"[TORSO] falls (torso < 0.31 m) during {args_cli.steps} steps x {args_cli.num_envs} envs: {done_count}")

    # -- 4. memory
    rss, peak = _read_rss_gb()
    print(f"[MEMORY] process RSS={rss:.2f} GB, peak={peak:.2f} GB")
    if torch.cuda.is_available():
        print(f"[MEMORY] torch peak CUDA allocated={torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
