#!/usr/bin/env python3
"""Validate the structure and fixed-root invariants of an Asimov motion NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def validate(path: Path) -> None:
    required = {
        "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
        "body_lin_vel_w", "body_ang_vel_w", "joint_names", "body_names", "fps",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Missing required arrays: {sorted(missing)}")
        q = data["joint_pos"]
        qd = data["joint_vel"]
        pos = data["body_pos_w"]
        quat = data["body_quat_w"]
        lin = data["body_lin_vel_w"]
        ang = data["body_ang_vel_w"]
        joint_names = tuple(str(x) for x in data["joint_names"].tolist())
        body_names = tuple(str(x) for x in data["body_names"].tolist())
        fps = float(np.asarray(data["fps"]).item())

        if q.ndim != 2 or q.shape[1] != 12:
            raise ValueError(f"joint_pos must have shape (T, 12); got {q.shape}")
        t = q.shape[0]
        if t < 3 or qd.shape != q.shape:
            raise ValueError(f"joint_vel must match joint_pos and T >= 3; got {qd.shape}")
        expected_shapes = {
            "body_pos_w": (t, 1, 3),
            "body_quat_w": (t, 1, 4),
            "body_lin_vel_w": (t, 1, 3),
            "body_ang_vel_w": (t, 1, 3),
        }
        for name, shape in expected_shapes.items():
            if data[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}; got {data[name].shape}")
        if len(joint_names) != 12 or len(set(joint_names)) != 12:
            raise ValueError(f"joint_names must contain 12 unique names; got {joint_names}")
        if body_names != ("pelvis_link",):
            raise ValueError(f"body_names must be ('pelvis_link',); got {body_names}")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"fps must be finite and positive; got {fps}")
        for name in required - {"joint_names", "body_names", "fps"}:
            if not np.isfinite(data[name]).all():
                raise ValueError(f"{name} contains NaN or infinite values")
        if not np.allclose(pos, pos[:1], atol=1e-7):
            raise ValueError("Root reference is not fixed; regenerate with the corrected converter")
        if not np.allclose(lin, 0.0, atol=1e-7) or not np.allclose(ang, 0.0, atol=1e-7):
            raise ValueError("Fixed-root reference must have zero root linear/angular velocity")
        if "marker_pos_source" in data.files:
            markers = data["marker_pos_source"]
            names = tuple(str(x) for x in data["marker_names"].tolist())
            if markers.shape != (t, 4, 3):
                raise ValueError(f"marker_pos_source must have shape {(t, 4, 3)}; got {markers.shape}")
            if names != ("right_foot", "left_foot", "right_knee", "left_knee"):
                raise ValueError(f"Unexpected marker order: {names}")
            if not np.isfinite(markers).all():
                raise ValueError("marker_pos_source contains NaN or infinite values")
        print(f"Reference OK: {path}")
        print(f"Frames={t}, duration={(t - 1) / fps:.3f}s, fps={fps:g}")
        print("Joint order:", ", ".join(joint_names))
        print("Root is fixed; markers preserved but not interpreted as training targets.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    args = parser.parse_args()
    validate(args.npz)


if __name__ == "__main__":
    main()
