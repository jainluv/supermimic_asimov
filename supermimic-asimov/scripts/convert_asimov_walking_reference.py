#!/usr/bin/env python3
"""Convert Asimov joint CSV data into a fixed-root SuperMimic GenCoord NPZ.

The source CSV contains 12 joint-position columns and four Cartesian marker
triplets (rf, lf, rk, lk). The current GenCoord task consumes the joint
trajectory and a pelvis anchor. Since this CSV contains no pelvis trajectory,
this converter deliberately does not invent forward root motion. Marker data
is preserved verbatim for later coordinate-frame validation; it is not
silently treated as world-frame body poses.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

MARKER_PREFIXES = ("rf", "lf", "rk", "lk")
MARKER_NAMES = ("right_foot", "left_foot", "right_knee", "left_knee")


def _joint_names_in_model_order(
    model: mujoco.MjModel, csv_columns: set[str]
) -> list[str]:
    names: list[str] = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None or model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
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
) -> None:
    if fps <= 0:
        raise ValueError(f"fps must be positive; received {fps}")
    if not np.isfinite(root_height):
        raise ValueError("root_height must be finite")

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    root_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link"
    )
    if root_body_id < 1:
        raise ValueError("MJCF does not contain a pelvis_link body")

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        rows = list(reader)
        columns = set(reader.fieldnames)

    joint_names = _joint_names_in_model_order(model, columns)
    if len(joint_names) != 12:
        raise ValueError(
            "Expected exactly 12 Asimov actuated joint columns matching the MJCF; "
            f"found {len(joint_names)}: {joint_names}"
        )
    if len(rows) < 3:
        raise ValueError("Need at least three CSV frames to calculate joint velocities")

    joint_pos = np.asarray(
        [[float(row[name]) for name in joint_names] for row in rows],
        dtype=np.float32,
    )
    if not np.isfinite(joint_pos).all():
        raise ValueError("Joint reference contains NaN or infinite values")
    joint_vel = np.gradient(
        joint_pos, 1.0 / fps, axis=0, edge_order=2
    ).astype(np.float32)

    marker_columns = [f"{prefix}_{axis}" for prefix in MARKER_PREFIXES for axis in "xyz"]
    present = [name in columns for name in marker_columns]
    if any(present) and not all(present):
        missing = [name for name, is_present in zip(marker_columns, present) if not is_present]
        raise ValueError(f"CSV has an incomplete Cartesian marker set; missing: {missing}")

    marker_pos_source = None
    if all(present):
        marker_pos_source = np.asarray(
            [
                [
                    [float(row[f"{prefix}_{axis}"]) for axis in "xyz"]
                    for prefix in MARKER_PREFIXES
                ]
                for row in rows
            ],
            dtype=np.float32,
        )
        if not np.isfinite(marker_pos_source).all():
            raise ValueError("Cartesian source markers contain NaN or infinite values")

    frame_count = joint_pos.shape[0]

    # This source file has no root trajectory. Use the specified home height and
    # a stationary pelvis instead of fabricating a constant forward speed.
    body_pos_w = np.zeros((frame_count, 1, 3), dtype=np.float32)
    body_pos_w[:, 0, 2] = float(root_height)
    body_quat_w = np.zeros((frame_count, 1, 4), dtype=np.float32)
    body_quat_w[:, 0, 0] = 1.0  # MuJoCo quaternion order: w, x, y, z.
    body_lin_vel_w = np.zeros((frame_count, 1, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((frame_count, 1, 3), dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "joint_names": np.asarray(joint_names, dtype="U64"),
        "body_names": np.asarray(["pelvis_link"], dtype="U64"),
        "fps": np.asarray(fps, dtype=np.float32),
        "root_height": np.asarray(root_height, dtype=np.float32),
        "reference_mode": np.asarray("joint_trajectory_fixed_root", dtype="U64"),
    }
    if marker_pos_source is not None:
        arrays["marker_pos_source"] = marker_pos_source
        arrays["marker_names"] = np.asarray(MARKER_NAMES, dtype="U32")

    np.savez_compressed(output_path, **arrays)
    print(f"Wrote: {output_path}")
    print(f"Frames: {frame_count}; duration: {(frame_count - 1) / fps:.3f}s; fps: {fps:g}")
    print("Joint order (MJCF order):")
    for index, name in enumerate(joint_names):
        print(f"  {index:2d}: {name}")
    print(f"Root: fixed at (0, 0, {root_height:.4f}) m; no synthetic forward motion.")
    if marker_pos_source is not None:
        print("Preserved raw CSV marker groups:", ", ".join(MARKER_NAMES))
        print("Marker coordinates are retained as source data, not yet used as training targets.")
    else:
        print("No Cartesian marker triplets found; saved joint-only reference.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--root-height", type=float, default=0.75)
    args = parser.parse_args()
    convert(args.csv, args.mjcf, args.output, args.fps, args.root_height)


if __name__ == "__main__":
    main()
