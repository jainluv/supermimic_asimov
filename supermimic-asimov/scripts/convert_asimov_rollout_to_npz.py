import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

def short_name(name):
    return (
        name.replace("_joint", "")
        .replace("left_", "L_")
        .replace("right_", "R_")
    )

source = Path("../asimov-mjlab/logs/2026-11-13_20-09-40/sim2sim.csv")

if not source.is_file():
    raise FileNotFoundError(source)

with source.open(newline="") as f:
    rows = list(csv.DictReader(f))

if len(rows) < 100:
    raise ValueError(f"Insufficient rollout frames: {len(rows)}")

times = np.array([float(r["time"]) for r in rows])
dt = np.diff(times)

if np.any(dt <= 0):
    raise ValueError("Invalid timestamps.")

fps = 1.0 / np.median(dt)

root_pos = np.column_stack([
    [float(r[k]) for r in rows]
    for k in ("pelvis_x", "pelvis_y", "pelvis_z")
])

quat = np.column_stack([
    [float(r[k]) for r in rows]
    for k in ("quat_w", "quat_x", "quat_y", "quat_z")
])

quat /= np.linalg.norm(quat, axis=1, keepdims=True)

joint_pos = np.column_stack([
    [float(r[f"act_{short_name(j)}"]) for r in rows]
    for j in JOINTS
])

# Use logged velocities if present, otherwise differentiate recorded positions.
velocity_columns = [
    f"vel_{short_name(j)}" for j in JOINTS
]

if all(k in rows[0] for k in velocity_columns):
    joint_vel = np.column_stack([
        [float(r[k]) for r in rows] for k in velocity_columns
    ])
else:
    joint_vel = np.gradient(joint_pos, times, axis=0)

# Root linear velocity in world coordinates.
root_lin_vel = np.gradient(root_pos, times, axis=0)

# Root angular velocity in world coordinates.
rotations = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
root_ang_vel = np.zeros((len(rows), 3))

for i in range(len(rows)):
    a = max(0, i - 1)
    b = min(len(rows) - 1, i + 1)
    delta = rotations[b] * rotations[a].inv()
    root_ang_vel[i] = delta.as_rotvec() / (times[b] - times[a])

arrays = [
    root_pos, quat, root_lin_vel, root_ang_vel,
    joint_pos, joint_vel
]

if not all(np.isfinite(a).all() for a in arrays):
    raise ValueError("Non-finite values in rollout.")

displacement = root_pos[-1] - root_pos[0]

path_length = np.linalg.norm(
    np.diff(root_pos[:, :2], axis=0), axis=1
).sum()

net_distance = np.linalg.norm(displacement[:2])
straightness = net_distance / max(path_length, 1e-8)

print("\n========== ROLLOUT VALIDATION ==========")
print("Frames:", len(rows))
print(f"Duration: {times[-1] - times[0]:.2f}s")
print(f"Frequency: {fps:.2f} Hz")
print(f"Forward displacement X: {displacement[0]:.3f} m")
print(f"Lateral displacement Y: {displacement[1]:.3f} m")
print(f"XY path length: {path_length:.3f} m")
print(f"Path straightness: {straightness:.3f}")
print(f"Mean forward velocity: {displacement[0] / (times[-1]-times[0]):.3f} m/s")
print(f"Minimum pelvis height: {root_pos[:, 2].min():.3f} m")

if displacement[0] < 0.5:
    raise ValueError("Insufficient forward displacement. Refusing conversion.")

if straightness < 0.50:
    print("WARNING: Trajectory is highly curved. Continuing conversion anyway.")

if root_pos[:, 2].min() < 0.35:
    raise ValueError("Robot appears to have fallen. Refusing conversion.")

output = Path("../asimov-mjlab/asimov_ppo_rollout.npz")

np.savez_compressed(
    output,
    fps=np.float32(fps),
    body_names=np.asarray(["pelvis_link"]),
    joint_names=np.asarray(JOINTS),
    body_pos_w=root_pos[:, None, :].astype(np.float32),
    body_quat_w=quat[:, None, :].astype(np.float32),
    body_lin_vel_w=root_lin_vel[:, None, :].astype(np.float32),
    body_ang_vel_w=root_ang_vel[:, None, :].astype(np.float32),
    joint_pos=joint_pos.astype(np.float32),
    joint_vel=joint_vel.astype(np.float32),
    motion_ends=np.asarray([len(rows)], dtype=np.int64),
)

print("\nSUCCESS: SuperMimic reference created.")
print(output.resolve())
