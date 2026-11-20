# Motion reference data

## Role

This directory stores motion references used as imitation targets. It is not the replay buffer used to fit the neural world model.

The world model is trained from state/action/next-state transitions collected by stepping the Asimov MJLab environment. The reference motion supplies the task goal and reference-tracking signal.

## Current source CSV

```text
asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/assets/walking_reference.csv
```

The current CSV header contains:
- 12 actuated joint position channels;
- 4 Cartesian marker triplets: right foot (`rf`), left foot (`lf`), right knee (`rk`), and left knee (`lk`).

The marker coordinate frame/body-site mapping must be validated before using those marker positions as world-model or motion-tracking targets.

## Convert

```bash
cd ~/supermimic_ws/supermimic-asimov
uv run python scripts/convert_asimov_walking_reference.py \
  --csv ../asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/assets/walking_reference.csv \
  --mjcf ../asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/xmls/asimov.xml \
  --output ../data/asimov_walking_rootonly.npz \
  --fps 50 \
  --root-height 0.75
```

The CSV does not contain a pelvis trajectory. The corrected converter therefore keeps the pelvis fixed at the specified home height. It does not add an invented constant forward speed. The output is a joint-trajectory / walk-in-place reference, not a travelling locomotion sequence.

The NPZ contains joint position/velocity, a fixed pelvis body pose, required metadata, and (when present) the original marker positions under `marker_pos_source`. Marker coordinates are preserved verbatim and are not currently consumed by the GenCoord task.

Validate:

```bash
uv run python scripts/validate_asimov_reference.py ../data/asimov_walking_rootonly.npz
```

## Inspect the file

```bash
python - <<'PY'
import numpy as np
p = "../data/asimov_walking_rootonly.npz"
with np.load(p, allow_pickle=False) as d:
    for key in d.files:
        x = d[key]
        print(f"{key}: shape={getattr(x, 'shape', None)}, dtype={getattr(x, 'dtype', type(x).__name__)}")
PY
```

## Reference-generation utility

`~/supermimic_ws/make_ref_v7.py` is a separate utility. Inspect its actual arguments before using it:

```bash
cd ~/supermimic_ws
python make_ref_v7.py --help
```

Keep original CSV inputs separate from converted NPZ outputs. Do not describe a fixed-root joint reference as full travelling locomotion data.
