"""SuperMimic GenCoord environment for the 12-DOF Asimov biped.

This uses Asimov-MJLab's own tracking base config and robot asset. The first
port tracks the pelvis root and all 12 joints; it does not claim full-link
MaxCoord tracking.
"""

from __future__ import annotations

import mjlab.envs.mdp as mdp
from mjlab.asset_zoo.robots.asimov.asimov_constants import (
  ASIMOV_ACTION_SCALE,
  get_asimov_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from supermimic.backends.asimov_motion import SuperMimicAsimovMotionCommand
from supermimic.representations import PROPRIO_NOISE_HALF_WIDTHS
from supermimic.tasks.env_cfgs import (
  joint_pos,
  joint_vel,
  motion_anchor_ori_b,
  motion_anchor_pos_b,
  motion_command_window,
  root_ang_vel,
  root_lin_vel,
  root_ori,
  root_pos,
)


def _asimov_actor_group() -> ObservationGroupCfg:
  """Build the same sensor layout as SuperMimic, without G1-only bias args."""
  terms: dict[str, ObservationTermCfg] = {}

  def add(name: str, func, **params) -> None:
    half_width = PROPRIO_NOISE_HALF_WIDTHS[name]
    kwargs: dict = {"func": func}
    if params:
      kwargs["params"] = params
    if half_width > 0:
      kwargs["noise"] = Unoise(n_min=-half_width, n_max=half_width)
    terms[name] = ObservationTermCfg(**kwargs)

  add("projected_gravity", mdp.projected_gravity)
  add("base_ang_vel", mdp.builtin_sensor, sensor_name="robot/imu_ang_vel")
  # Asimov-MJLab joint_pos_rel has no ``biased`` parameter and this fork does
  # not define the G1 encoder_bias startup event.
  add("joint_pos", mdp.joint_pos_rel)
  add("joint_vel", mdp.joint_vel_rel)
  add("actions", mdp.last_action)
  return ObservationGroupCfg(
    terms=terms, concatenate_terms=True, enable_corruption=False
  )


def _asimov_goal_group() -> ObservationGroupCfg:
  return ObservationGroupCfg(
    terms={
      "command": ObservationTermCfg(func=motion_command_window),
      "motion_anchor_pos_b": ObservationTermCfg(func=motion_anchor_pos_b),
      "motion_anchor_ori_b": ObservationTermCfg(func=motion_anchor_ori_b),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )


def asimov_gencoord_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Asimov GenCoord task using Asimov-MJLab's tracking stack."""
  cfg = make_tracking_env_cfg()
  cfg.scene.entities = {"robot": get_asimov_robot_cfg()}

  action_cfg = cfg.actions["joint_pos"]
  assert isinstance(action_cfg, JointPositionActionCfg)
  action_cfg.scale = ASIMOV_ACTION_SCALE

  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.asset_name = "robot"
  motion_cmd.anchor_body_name = "pelvis_link"
  # One tracked root body keeps GenCoord's world-model FK contract valid
  # without inventing an unverified differentiable leg-link FK implementation.
  motion_cmd.body_names = ("pelvis_link",)
  motion_cmd.class_type = SuperMimicAsimovMotionCommand

  cfg.viewer.body_name = "pelvis_link"

  # The Asimov fork calls this observation group "policy". SuperMimic's
  # runner consumes "actor", so replace the former with the latter.
  cfg.observations.pop("policy", None)
  cfg.observations["actor"] = _asimov_actor_group()
  cfg.observations["full_state"] = ObservationGroupCfg(
    terms={
      "root_pos": ObservationTermCfg(func=root_pos),
      "root_ori": ObservationTermCfg(func=root_ori),
      "joint_pos": ObservationTermCfg(func=joint_pos),
      "root_lin_vel": ObservationTermCfg(func=root_lin_vel),
      "root_ang_vel": ObservationTermCfg(func=root_ang_vel),
      "joint_vel": ObservationTermCfg(func=joint_vel),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.observations["goal"] = _asimov_goal_group()

  # Keep Asimov's own physical randomization hooks, but remove interval pushes
  # during the first integration baseline.
  cfg.events.pop("push_robot", None)
  if "base_com" in cfg.events:
    cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis_link",)
  if "foot_friction" in cfg.events:
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
      "left_ankle_roll_link_collision",
      "right_ankle_roll_link_collision",
    )

  cfg.rewards.pop("self_collisions", None)
  if "joint_limit" in cfg.rewards:
    cfg.rewards["joint_limit"].weight = -2.0
  cfg.terminations.pop("ee_body_pos", None)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
