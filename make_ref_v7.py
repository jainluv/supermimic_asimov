import sys
import numpy as np
J = ["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
     "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint"]
d = np.genfromtxt("new_imitation_data_walking_1.25Hz_50Hz.csv", delimiter=",", names=True)
N, fps = len(d), 50.0
t = np.arange(N) / fps
jp = np.column_stack([d[j] for j in J]); jv = np.gradient(jp, t, axis=0)
rf = np.column_stack([d["rf_x"], d["rf_y"], d["rf_z"]]); lf = np.column_stack([d["lf_x"], d["lf_y"], d["lf_z"]])
print("mean foot z: rf", rf[:, 2].mean().round(3), "lf", lf[:, 2].mean().round(3))
if max(rf[:, 2].mean(), lf[:, 2].mean()) > -0.3:
    sys.exit("STOP: feet not clearly below the pelvis. Tell Claude.")
w = 1 / (1 + np.exp(-300 * (lf[:, 2] - rf[:, 2])))
vel = -(w[:, None] * np.gradient(rf, t, axis=0) + (1 - w)[:, None] * np.gradient(lf, t, axis=0))
pos = np.cumsum(vel, axis=0) / fps; pos -= pos[0]
pos[:, 1] -= np.polyval(np.polyfit(t, pos[:, 1], 1), t)
h = -(w * rf[:, 2] + (1 - w) * lf[:, 2])
if abs(h.mean() - 0.739) > 0.08:
    print("height mean", h.mean().round(3), "-> shifting to 0.739"); h = h - h.mean() + 0.739
pos[:, 2] = h
vx = pos[-1, 0] / t[-1]
print("forward speed", round(vx, 3), "m/s | net x", round(pos[-1, 0], 2), "m | sway p-p", round(np.ptp(pos[:, 1]), 3), "m | height", round(h.mean(), 3), "+/-", round(h.std(), 4))
if not 0.1 < vx < 0.8: sys.exit("STOP: speed implausible. Tell Claude.")
lin = np.gradient(pos, t, axis=0); quat = np.tile([1.0, 0, 0, 0], (N, 1))
np.savez_compressed("asimov_csv_walk.npz", fps=np.float32(fps), body_names=np.asarray(["pelvis_link"]), joint_names=np.asarray(J),
    body_pos_w=pos[:, None].astype(np.float32), body_quat_w=quat[:, None].astype(np.float32),
    body_lin_vel_w=lin[:, None].astype(np.float32), body_ang_vel_w=np.zeros((N, 1, 3), np.float32),
    joint_pos=jp.astype(np.float32), joint_vel=jv.astype(np.float32), motion_ends=np.asarray([N], np.int64))
print("saved asimov_csv_walk.npz")
