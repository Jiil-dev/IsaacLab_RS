# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Small networks and buffers used by the residual action term (condition D) and its offline tools.

Only ``torch`` is needed here, so the offline scripts (export, dynamics training, calibration) can import this
module without starting Isaac Sim. The same :class:`TransitionHistory` and :class:`ErrorWindow` are used at run time
and during data collection / calibration, so the dynamics model sees identically built inputs everywhere.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn

STATE_DIM = 22
"""Size of ``proprio_state``."""

ACTION_DIM = 8
"""Number of actuated joints."""

TARGET_SLICE = slice(8, 19)
"""Entries of ``proprio_state`` predicted by the dynamics model: joint velocities (x0.2) and base angular velocity."""

TARGET_DIM = TARGET_SLICE.stop - TARGET_SLICE.start

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
"""Folder of the frozen networks shipped with the task (``ant_robust/weights``)."""


def resolve_weights_path(path: str) -> str:
    """Absolute paths are used as given; bare file names are looked up in :data:`WEIGHTS_DIR`."""
    return path if os.path.isabs(path) else os.path.join(WEIGHTS_DIR, path)


def build_mlp(in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
    """Linear layers with ELU activations in between (same layout as the RSL-RL actor)."""
    layers: list[nn.Module] = []
    dims = [in_dim, *hidden_dims]
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(d_in, d_out), nn.ELU()]
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


##
# Frozen base policy
##


def actor_from_rsl_rl_checkpoint(checkpoint_path: str) -> tuple[nn.Sequential, dict]:
    """Rebuild the actor of an RSL-RL ``ActorCritic`` checkpoint as a plain MLP.

    RSL-RL stores the actor as ``actor.0``, ``actor.2``, ... (Linear layers; the ELU layers have no weights).
    Observation normalization must be off (true for all HW1 runs), otherwise a normalizer would be needed too.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("actor_obs_normalizer.") for k in state_dict):
        raise ValueError("Checkpoints with observation normalization are not supported.")
    actor_sd = {k.removeprefix("actor."): v for k, v in state_dict.items() if k.startswith("actor.")}
    layer_ids = sorted({int(k.split(".")[0]) for k in actor_sd})
    weights = [actor_sd[f"{i}.weight"] for i in layer_ids]
    hidden_dims = [w.shape[0] for w in weights[:-1]]
    actor = build_mlp(weights[0].shape[1], hidden_dims, weights[-1].shape[0])
    # Sequential indices are contiguous (0, 2, 4, 6 for Linear) just like in RSL-RL
    actor.load_state_dict({f"{i}.{p}": actor_sd[f"{i}.{p}"] for i in layer_ids for p in ("weight", "bias")})
    info = {
        "obs_dim": weights[0].shape[1],
        "act_dim": weights[-1].shape[0],
        "hidden_dims": hidden_dims,
        "iteration": checkpoint.get("iter"),
        "source": os.path.abspath(checkpoint_path),
    }
    return actor, info


def save_frozen_policy(actor: nn.Sequential, info: dict, path: str):
    torch.save({"state_dict": actor.state_dict(), "info": info}, path)


def load_frozen_policy(path: str, device: str) -> nn.Sequential:
    data = torch.load(resolve_weights_path(path), map_location="cpu", weights_only=False)
    info = data["info"]
    actor = build_mlp(info["obs_dim"], info["hidden_dims"], info["act_dim"])
    actor.load_state_dict(data["state_dict"])
    return _freeze(actor.to(device))


##
# Dynamics model and the inputs it needs
##


class TransitionHistory:
    """The last ``history`` (state, action) pairs of every environment, oldest first.

    After a reset, the first push fills the older slots with that first state and zero actions. This matches the
    zero ``last_action`` of the action manager after a reset.
    """

    def __init__(self, num_envs: int, history: int, device: str):
        self.states = torch.zeros(num_envs, history, STATE_DIM, device=device)
        self.actions = torch.zeros(num_envs, history, ACTION_DIM, device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        """True for environments whose current episode has no pushed step yet."""

    def reset(self, env_ids=None):
        self.fresh[slice(None) if env_ids is None else env_ids] = True

    def last_state(self) -> torch.Tensor:
        return self.states[:, -1]

    def push(self, state: torch.Tensor, action: torch.Tensor):
        fresh = self.fresh
        self.states[fresh] = state[fresh].unsqueeze(1)
        self.actions[fresh] = 0.0
        self.states[:] = torch.roll(self.states, shifts=-1, dims=1)
        self.actions[:] = torch.roll(self.actions, shifts=-1, dims=1)
        self.states[:, -1] = state
        self.actions[:, -1] = action
        self.fresh[:] = False

    def features(self) -> torch.Tensor:
        """Flattened ``[s_{t-k+1}, a_{t-k+1}, ..., s_t, a_t]``. Shape (num_envs, history * 30)."""
        return torch.cat((self.states, self.actions), dim=-1).flatten(1)


class ErrorWindow:
    """Mean of the last ``size`` prediction errors of every environment."""

    def __init__(self, num_envs: int, size: int, device: str):
        self.size = size
        self.errors = torch.zeros(num_envs, size, device=device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.errors[ids] = 0.0
        self.count[ids] = 0

    def push(self, error: torch.Tensor, mask: torch.Tensor):
        """Append ``error`` for the environments in ``mask`` (others are left unchanged)."""
        shifted = torch.roll(self.errors, shifts=-1, dims=1)
        shifted[:, -1] = error
        self.errors[:] = torch.where(mask.unsqueeze(1), shifted, self.errors)
        self.count[:] = torch.where(mask, (self.count + 1).clamp(max=self.size), self.count)

    def mean(self) -> torch.Tensor:
        # empty slots are zero, so the sum only contains valid errors
        return self.errors.sum(dim=1) / self.count.clamp(min=1)

    def full(self) -> torch.Tensor:
        return self.count >= self.size


class DynamicsModel(nn.Module):
    """Predicts the next change of joint velocities and base angular velocity from recent states and actions.

    It is trained only on nominal conditions (flat ground, friction 1.0). A large prediction error therefore means
    "the robot is not responding the way it does on nominal ground", which drives the correction strength alpha.
    """

    def __init__(self, history: int = 3, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.history = history
        self.hidden_dims = list(hidden_dims)
        in_dim = history * (STATE_DIM + ACTION_DIM)
        self.net = build_mlp(in_dim, self.hidden_dims, TARGET_DIM)
        # normalization statistics of the training data
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std", torch.ones(in_dim))
        self.register_buffer("out_mean", torch.zeros(TARGET_DIM))
        self.register_buffer("out_std", torch.ones(TARGET_DIM))
        # calibration of alpha: mean error at which alpha starts to rise (e_lo) and reaches one (e_hi)
        self.register_buffer("e_lo", torch.tensor(0.0))
        self.register_buffer("e_hi", torch.tensor(1.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Normalized prediction of ``s_{t+1}[TARGET_SLICE] - s_t[TARGET_SLICE]``."""
        return self.net((features - self.in_mean) / self.in_std)

    def prediction_error(self, prediction: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
        """Mean squared error in normalized units. ``target_delta`` is in raw units. Shape (num_envs,)."""
        target = (target_delta - self.out_mean) / self.out_std
        return (prediction - target).square().mean(dim=-1)

    def alpha(self, mean_error: torch.Tensor) -> torch.Tensor:
        return ((mean_error - self.e_lo) / (self.e_hi - self.e_lo)).clamp(0.0, 1.0)


def save_dynamics(model: DynamicsModel, path: str, info: dict | None = None):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "history": model.history,
            "hidden_dims": model.hidden_dims,
            "info": info or {},
        },
        path,
    )


def load_dynamics(path: str, device: str) -> DynamicsModel:
    data = torch.load(resolve_weights_path(path), map_location="cpu", weights_only=False)
    model = DynamicsModel(data["history"], tuple(data["hidden_dims"]))
    model.load_state_dict(data["state_dict"])
    model.info = data.get("info", {})  # training report and alpha calibration (window, e_lo, e_hi, AUROC)
    return _freeze(model.to(device))
