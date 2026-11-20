"""Compatibility adapter for Asimov-MJLab's precomputed-body MotionLoader.

The initial Asimov GenCoord port intentionally tracks the pelvis root plus all
12 actuated joint coordinates.  It does not pretend to provide differentiable
FK for the 14 leg-link bodies.  A full-link FK adapter is a separate follow-up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from types import MethodType

import numpy as np
import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg


def _motion_body_state(self, motion_times: torch.Tensor):
  """Return tracked body states at integer reference-frame indices."""
  frames = torch.as_tensor(
    motion_times, dtype=torch.long, device=self.joint_pos.device
  ).clamp(0, self.time_step_total - 1)

  return (
    self.body_pos_w[frames],
    self.body_quat_w[frames],
    self.body_lin_vel_w[frames],
    self.body_ang_vel_w[frames],
  )


class _RootOnlyForwardKinematics:
  """Differentiable FK contract for a single tracked root body."""

  def __init__(self, joint_names: tuple[str, ...]) -> None:
    self.joint_names = joint_names

  def __call__(
    self,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    joint_pos: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # GenCoord's root state is already the complete state of the one tracked
    # body. Joint coordinates remain represented and rewarded separately.
    del joint_pos, joint_vel
    return (
      root_pos_w.unsqueeze(-2),
      root_quat_w.unsqueeze(-2),
      root_lin_vel_w.unsqueeze(-2),
      root_ang_vel_w.unsqueeze(-2),
    )


class SuperMimicAsimovMotionCommand(MotionCommand):
  """Expose the extra MotionLoader contract expected by SuperMimic GenCoord.

  Asimov-MJLab's MotionLoader stores selected body transforms and joint states,
  whereas SuperMimic also expects root_* aliases, motion_ends and ``fk``.
  This adapter supplies those fields for a single-root tracked body.
  """

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv) -> None:
    # Asimov-MJLab accepts one NPZ path. SuperMimic's CLI also allows a list
    # when multiple comma-separated files are supplied; explicitly reject that
    # case rather than letting np.load fail with an opaque type error.
    if isinstance(cfg.motion_file, (list, tuple)):
      if len(cfg.motion_file) != 1:
        raise ValueError(
          "The first Asimov integration supports one motion NPZ at a time; "
          f"received {len(cfg.motion_file)} files."
        )
      cfg.motion_file = cfg.motion_file[0]

    super().__init__(cfg, env)

    if cfg.anchor_body_name != "pelvis_link" or tuple(cfg.body_names) != ("pelvis_link",):
      raise ValueError(
        "The initial Asimov GenCoord adapter tracks only body_names=('pelvis_link',) "
        "with anchor_body_name='pelvis_link'. Full-link tracking needs a separate FK port."
      )

    # Entity.body_names excludes MuJoCo's world body and starts with the root
    # body. Therefore pelvis_link resolves to entity-local body index zero.
    if self.body_indexes.tolist() != [0]:
      raise RuntimeError(
        "Expected Asimov pelvis_link to be entity body index 0; got "
        f"{self.body_indexes.tolist()} with body_names={self.robot.body_names}."
      )

    motion = self.motion
    if motion.joint_pos.ndim != 2 or motion.joint_pos.shape[1] != self.robot.num_joints:
      raise ValueError(
        "Asimov motion joint_pos must have shape (frames, 12); got "
        f"{tuple(motion.joint_pos.shape)}."
      )
    if motion.joint_vel.shape != motion.joint_pos.shape:
      raise ValueError(
        "Asimov motion joint_vel must have the same shape as joint_pos; got "
        f"{tuple(motion.joint_vel.shape)} vs {tuple(motion.joint_pos.shape)}."
      )

    # Require explicit joint-name metadata to catch column-order mistakes.
    with np.load(cfg.motion_file, allow_pickle=False) as motion_file:
      if "joint_names" not in motion_file.files:
        raise ValueError(
          "Motion file is missing joint_names metadata. Recreate it with "
          "scripts/convert_asimov_walking_reference.py."
        )
      if "body_names" not in motion_file.files:
        raise ValueError(
          "Motion file is missing body_names metadata. Recreate it with "
          "scripts/convert_asimov_walking_reference.py."
        )
      file_joint_names = tuple(str(x) for x in motion_file["joint_names"].tolist())
      file_body_names = tuple(str(x) for x in motion_file["body_names"].tolist())
      if file_body_names != ("pelvis_link",):
        raise ValueError(
          "Root-only Asimov motion must declare body_names=('pelvis_link',); "
          f"got {file_body_names}."
        )
    robot_joint_names = tuple(self.robot.joint_names)
    if file_joint_names != robot_joint_names:
      raise ValueError(
        "Motion joint order does not match Asimov MJLab joint order.\n"
        f"Motion: {file_joint_names}\nRobot:  {robot_joint_names}"
      )

    # Asimov's loader indexes bodies using the configured body_names. Our
    # one-body task therefore expects a one-body motion array at index zero.
    if motion.body_pos_w.shape[1] != 1:
      raise ValueError(
        "Root-only Asimov motion must contain exactly one tracked body "
        "(pelvis_link); got "
        f"{motion.body_pos_w.shape[1]} bodies."
      )

    motion.root_pos_w = motion.body_pos_w[:, 0]
    motion.root_quat_w = motion.body_quat_w[:, 0]
    motion.root_lin_vel_w = motion.body_lin_vel_w[:, 0]
    motion.root_ang_vel_w = motion.body_ang_vel_w[:, 0]
    motion.motion_ends = torch.tensor(
      [motion.time_step_total], dtype=torch.long, device=self.device
    )
    motion.fk = _RootOnlyForwardKinematics(robot_joint_names)
    motion.body_state = MethodType(_motion_body_state, motion)

    # Latched for one environment step before MotionCommand resamples
    # the frame counter at the end of a clip.
    self.clip_end = torch.zeros_like(self.time_steps, dtype=torch.bool)

  def _update_command(self) -> None:
    self.clip_end = (
      self.time_steps + 1 >= self.motion.time_step_total
    )
    super()._update_command()
