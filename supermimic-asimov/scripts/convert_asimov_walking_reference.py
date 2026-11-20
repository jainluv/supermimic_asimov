#!/usr/bin/env python3
"""Convert Asimov's joint CSV into the single-root NPZ expected by this port.

This produces a *walk-in-place/interface-smoke-test* clip: pelvis position and
orientation are held fixed, while the 12 reference joint trajectories are
replayed. It is not a retargeter for travelling human locomotion.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np


def _joint_names_in_model_order(model: mujoco.MjModel, csv_columns: set[str]) -> list[str]:
  names: list[str] = []
  for joint_id in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if name is None:
      continue
    if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    if name in csv_columns:
      names.append(name)
  return names


def convert(
  csv_path: Path,
  mjcf_path: Path,
  output_path: Path,
  fps: float,
  root_height: float,
  forward_speed: float,
) -> None:
  if fps <= 0:
    raise ValueError("fps must be positive")
  model = mujoco.MjModel.from_xml_path(str(mjcf_path))
  root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
  if root_body_id < 1:
    raise ValueError("MJCF does not contain a pelvis_link body")

  with csv_path.open(newline="") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
      raise ValueError(f"CSV has no header: {csv_path}")
    rows = list(reader)
    columns = set(reader.fieldnames)

  joint_names = _joint_names_in_model_order(model, columns)
  if len(joint_names) != 12:
    raise ValueError(
      f"Expected 12 Asimov hinge joints in CSV and MJCF; found {len(joint_names)}: {joint_names}"
    )
  if len(rows) < 3:
    raise ValueError("Need at least three CSV frames to calculate joint velocities")

  joint_pos = np.asarray(
    [[float(row[name]) for name in joint_names] for row in rows], dtype=np.float32
  )
  joint_vel = np.gradient(joint_pos, 1.0 / fps, axis=0, edge_order=2).astype(np.float32)
  frame_count = joint_pos.shape[0]

  # Entity.body_names omits MuJoCo worldbody and starts with its root body.
  # The task intentionally configures body_names=('pelvis_link',), index 0.
  body_pos_w = np.zeros((frame_count, 1, 3), dtype=np.float32)
  body_pos_w[:, 0, 2] = root_height
  time = np.arange(frame_count, dtype=np.float32) / fps
  body_pos_w[:, 0, 0] = forward_speed * time
  time = np.arange(frame_count, dtype=np.float32) / fps
  body_pos_w[:, 0, 0] = forward_speed * time
  time = np.arange(frame_count, dtype=np.float32) / fps
  body_pos_w[:, 0, 0] = forward_speed * time
  body_quat_w = np.zeros((frame_count, 1, 4), dtype=np.float32)
  body_quat_w[:, 0, 0] = 1.0
  body_lin_vel_w = np.zeros((frame_count, 1, 3), dtype=np.float32)
  body_lin_vel_w[:, 0, 0] = forward_speed
  body_lin_vel_w[:, 0, 0] = forward_speed
  body_lin_vel_w[:, 0, 0] = forward_speed
  body_ang_vel_w = np.zeros((frame_count, 1, 3), dtype=np.float32)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output_path,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
    joint_names=np.asarray(joint_names, dtype="U64"),
    body_names=np.asarray(["pelvis_link"], dtype="U64"),
    fps=np.asarray(fps, dtype=np.float32),
  )
  print(f"Wrote: {output_path}")
  print(f"Frames: {frame_count}; duration: {(frame_count - 1) / fps:.3f}s; fps: {fps:g}")
  print("Joint order:")
  for index, name in enumerate(joint_names):
    print(f"  {index:2d}: {name}")
  print(f"Forward root speed: {forward_speed:.3f} m/s (provisional).")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--csv", type=Path, required=True)
  parser.add_argument("--mjcf", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--fps", type=float, default=50.0)
  parser.add_argument("--root-height", type=float, default=0.75)
  parser.add_argument("--forward-speed", type=float, default=0.30)
  args = parser.parse_args()
  convert(args.csv, args.mjcf, args.output, args.fps, args.root_height, args.forward_speed)


if __name__ == "__main__":
  main()
