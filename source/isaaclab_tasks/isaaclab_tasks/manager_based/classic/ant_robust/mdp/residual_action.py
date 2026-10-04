# Copyright (c) 2026, Jiil-dev. HW1 (Robotics Simulation) - robust Ant locomotion.
# SPDX-License-Identifier: BSD-3-Clause

"""Residual joint-effort action of condition D.

The PPO policy outputs a correction ``delta`` and this term turns it into joint torques::

    a_t    = pi_base(o_t) + alpha_t * delta_t      (pi_base: frozen policy of condition B)
    torque = 7.5 * a_t                               (same scale as the original JointEffortAction)

``alpha_t`` is not learned. It comes from a frozen dynamics model trained on nominal ground: the larger its recent
prediction error, the less "nominal" the current environment, and the more of the correction is applied.

The final action ``a_t`` is written back into the action manager buffer. The reward terms ``action_l2`` and
``energy`` and the ``last_action`` observation read that buffer, so they see the action that was actually applied,
exactly as in the original task.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import models
from .observations import base_obs, proprio_state, resolved_feet_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ResidualJointEffortAction(ActionTerm):
    """Joint effort action ``scale * (pi_base(o) + alpha * delta)`` with a frozen base policy."""

    cfg: ResidualJointEffortActionCfg
    _asset: Articulation

    def __init__(self, cfg: ResidualJointEffortActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.alpha_mode not in ("error", "zero", "one"):
            raise ValueError(f"Unknown alpha_mode '{cfg.alpha_mode}'. Use 'error', 'zero' or 'one'.")
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # buffers (always updated in place: rollouts run under torch.inference_mode)
        self._raw_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._final_actions = torch.zeros_like(self._raw_actions)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self.base_actions = torch.zeros_like(self._raw_actions)
        """Action proposed by the frozen base policy in this step."""
        self.alpha = torch.zeros(self.num_envs, device=self.device)
        """Correction strength in [0, 1] used in this step."""
        self.pred_error = torch.zeros(self.num_envs, device=self.device)
        """Mean prediction error over the window (before mapping to alpha)."""

        # frozen networks
        self._feet_cfg = resolved_feet_cfg(env)
        self.base_policy = models.load_frozen_policy(cfg.base_policy_file, self.device)
        self.dynamics: models.DynamicsModel | None = None
        if cfg.alpha_mode == "error":
            self.dynamics = models.load_dynamics(cfg.dynamics_file, self.device)
            calibrated_window = self.dynamics.info.get("calibration", {}).get("window")
            if calibrated_window != cfg.error_window:
                raise ValueError(
                    f"'{cfg.dynamics_file}' was calibrated with an error window of {calibrated_window} steps,"
                    f" but the action term uses {cfg.error_window}."
                )
            self._history = models.TransitionHistory(self.num_envs, self.dynamics.history, self.device)
            self._window = models.ErrorWindow(self.num_envs, cfg.error_window, self.device)
            self._prediction = torch.zeros(self.num_envs, models.TARGET_DIM, device=self.device)

        # columns of this term inside the action manager buffer (resolved at the first step)
        self._action_slice: slice | None = None

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        """Correction ``delta`` from the PPO policy."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Joint torques."""
        return self._processed_actions

    @property
    def final_actions(self) -> torch.Tensor:
        """Applied action ``pi_base(o) + alpha * delta``."""
        return self._final_actions

    """
    Operations.
    """

    def compute_base_action(self, last_action: torch.Tensor) -> torch.Tensor:
        """Base policy action for the current state, given the action applied in the previous step."""
        return self.base_policy(base_obs(self._env, last_action, self._feet_cfg))

    def process_actions(self, actions: torch.Tensor):
        if self._action_slice is None:
            self._action_slice = self._find_action_slice()
        self._raw_actions[:] = actions
        state = proprio_state(self._env)

        # 1) correction strength from the error of the prediction made one step ago
        if self.dynamics is not None:
            has_prediction = ~self._history.fresh
            target_delta = state[:, models.TARGET_SLICE] - self._history.last_state()[:, models.TARGET_SLICE]
            error = self.dynamics.prediction_error(self._prediction, target_delta)
            self._window.push(error, has_prediction)
            self.pred_error[:] = self._window.mean()
            # alpha stays zero until the window is full (about 0.27 s after a reset)
            self.alpha[:] = torch.where(self._window.full(), self.dynamics.alpha(self.pred_error), 0.0)
        elif self.cfg.alpha_mode == "one":
            self.alpha[:] = 1.0
        else:
            self.alpha[:] = 0.0

        # 2) base action; the manager's prev_action holds the final action written back in the previous step
        last_action = self._env.action_manager.prev_action[:, self._action_slice]
        self.base_actions[:] = self.compute_base_action(last_action)

        # 3) final action, written back so that rewards and observations see what is applied
        self._final_actions[:] = self.base_actions + self.alpha.unsqueeze(-1) * self._raw_actions
        self._env.action_manager.action[:, self._action_slice] = self._final_actions
        self._processed_actions[:] = self._final_actions * self.cfg.scale

        # 4) predict the change of the next step from the updated history
        if self.dynamics is not None:
            self._history.push(state, self._final_actions)
            self._prediction[:] = self.dynamics(self._history.features())

    def apply_actions(self):
        self._asset.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self._final_actions[ids] = 0.0
        self.alpha[ids] = 0.0
        self.pred_error[ids] = 0.0
        if self.dynamics is not None:
            self._history.reset(env_ids)
            self._window.reset(env_ids)

    """
    Helpers.
    """

    def _find_action_slice(self) -> slice:
        manager = self._env.action_manager
        start = 0
        for name, dim in zip(manager.active_terms, manager.action_term_dim):
            if manager.get_term(name) is self:
                return slice(start, start + dim)
            start += dim
        raise RuntimeError("Residual action term is not registered in the action manager.")


@configclass
class ResidualJointEffortActionCfg(ActionTermCfg):
    """Configuration for :class:`ResidualJointEffortAction`."""

    class_type: type[ActionTerm] = ResidualJointEffortAction

    joint_names: list[str] = [".*"]
    scale: float = 7.5
    """Torque per unit action (the original Ant task uses 7.5)."""

    base_policy_file: str = "base_policy_seed42.pt"
    """Frozen base policy (file name inside ``ant_robust/weights`` or an absolute path)."""

    dynamics_file: str = "dynamics_seed42.pt"
    """Frozen dynamics model with its alpha calibration (file name inside ``ant_robust/weights`` or absolute)."""

    error_window: int = 60
    """Number of recent prediction errors averaged before mapping to alpha (60 steps = 1 s).

    The plan used 15 steps. Gate G3 (flat vs DR separation, AUROC >= 0.70) failed with 15 and, for one seed, with 30;
    60 is the shortest window that passes for all seeds. Must equal the window stored in the dynamics file by
    ``calibrate_alpha.py`` (checked at start-up)."""

    alpha_mode: str = "error"
    """``"error"``: alpha from the prediction error. ``"zero"``: base policy only (equivalence test).
    ``"one"``: correction always fully applied (ablation without gating)."""
