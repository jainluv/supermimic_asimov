"""Gaussian-kernel tracking reward on world-frame bodies.

Mirrors ``mjlab.tasks.tracking.tracking_env_cfg.rewards``: each tracking term is
``weight * exp(-err_sq / std**2)``, plus the action-rate and joint-limit
penalties; body pos/ori targets are re-anchored
to the predicted root xy + yaw, matching
``MotionCommand.update_relative_body_poses``.
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_inv, quat_mul, yaw_quat

from supermimic.utils.helpers import rotation_error_magnitude, small_matmul


def _reanchor_targets(
  target_pos: torch.Tensor,
  target_quat: torch.Tensor,
  pred_root_pos: torch.Tensor,
  pred_root_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Re-anchor target body poses to predicted root xy + yaw.

  Target bodies are expressed in a frame whose xy + yaw come from the
  predicted root and whose z + roll/pitch come from the target root.
  """
  anchor_pos = target_pos[..., 0:1, :]

  delta_pos = torch.cat([pred_root_pos[..., :2], target_pos[..., 0, 2:3]], dim=-1)
  delta_mat = matrix_from_quat(
    yaw_quat(quat_mul(pred_root_quat, quat_inv(target_quat[..., 0, :])))
  ).unsqueeze(-3)

  rel_pos = target_pos - anchor_pos
  tgt_pos = (
    delta_pos.unsqueeze(-2) + small_matmul(delta_mat, rel_pos.unsqueeze(-1))[..., 0]
  )
  tgt_mat = small_matmul(delta_mat, matrix_from_quat(target_quat))
  return tgt_pos, tgt_mat


def tracking_reward(
  aux_state: torch.Tensor,
  pred_pos: torch.Tensor,
  pred_rot_mat: torch.Tensor,
  pred_lin_vel: torch.Tensor,
  pred_ang_vel: torch.Tensor,
  joint_state: torch.Tensor,
  action: torch.Tensor,
  last_action: torch.Tensor,
  target_pos: torch.Tensor,
  target_quat: torch.Tensor,
  target_lin_vel: torch.Tensor,
  target_ang_vel: torch.Tensor,
  target_joint_pos: torch.Tensor,
  target_joint_vel: torch.Tensor,
  soft_joint_pos_limits: torch.Tensor,
  dt: float,
) -> torch.Tensor:
  """Gaussian-kernel reward matching the motion-tracking env.

  ``pred_*`` are the predicted bodies in the world frame and ``aux_state`` the
  predicted root pose the targets are re-anchored to.
  """
  tgt_pos, tgt_mat = _reanchor_targets(
    target_pos, target_quat, aux_state[..., :3], aux_state[..., 3:7]
  )

  terms: list[tuple[torch.Tensor, float, float]] = [
    # Absolute pelvis tracking is essential: this is the term that
    # penalizes a policy that remains stationary while the reference moves.
    (torch.sum((pred_pos[..., 0, :] - target_pos[..., 0, :]) ** 2, dim=-1), 0.5, 0.12),
    (
      rotation_error_magnitude(
        pred_rot_mat[..., 0, :, :], matrix_from_quat(target_quat[..., 0, :])
      )
      ** 2,
      0.5,
      0.25,
    ),
    (torch.sum((pred_lin_vel - target_lin_vel) ** 2, dim=-1).mean(-1), 1.0, 0.15),
    (torch.sum((pred_ang_vel - target_ang_vel) ** 2, dim=-1).mean(-1), 1.0, 0.75),
  ]

  # Relative body tracking is meaningful only when actual non-root bodies
  # are supplied. With pelvis_link alone, re-anchoring makes these terms
  # self-comparisons and they provide no useful supervision.
  if pred_pos.shape[-2] > 1:
    terms.extend([
      (torch.sum((pred_pos - tgt_pos) ** 2, dim=-1).mean(-1), 1.0, 0.3),
      ((rotation_error_magnitude(pred_rot_mat, tgt_mat) ** 2).mean(-1), 1.0, 0.4),
    ])

  joint_pos, joint_vel = joint_state.chunk(2, dim=-1)
  joint_pos_error = torch.mean((joint_pos - target_joint_pos) ** 2, dim=-1)
  joint_vel_error = torch.mean((joint_vel - target_joint_vel) ** 2, dim=-1)

  # Joint imitation must distinguish the reference gait from a stationary pose.
  terms.extend([
    (joint_pos_error, 2.0, 0.20),
    (joint_vel_error, 0.5, 0.50),
  ])

  # Wide-kernel companions: keep a gradient when far from the reference,
  # while the tight terms above still separate walking from standing.
  terms.extend([
    (torch.sum((pred_pos[..., 0, :] - target_pos[..., 0, :]) ** 2, dim=-1), 0.5, 0.6),
    (torch.sum((pred_lin_vel - target_lin_vel) ** 2, dim=-1).mean(-1), 0.5, 1.5),
    (joint_pos_error, 1.0, 0.5),
  ])

  reward = sum(w * torch.exp(-err / std**2) for err, w, std in terms)
  action_rate = torch.sum((action - last_action) ** 2, dim=-1)
  lo, hi = soft_joint_pos_limits.unbind(-1)
  joint_limit = ((lo - joint_pos).clamp(min=0) + (joint_pos - hi).clamp(min=0)).sum(-1)
  return dt * (reward - 0.1 * action_rate - 2.0 * joint_limit)
