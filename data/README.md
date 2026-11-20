# Motion-reference Data

## Purpose

This directory stores source and generated motion-reference datasets used by the Asimov SuperMimic workflow. It is the hand-off point between the Asimov walking-reference source and the NPZ motion file loaded by training.

## Current data flow

```text
asimov-mjlab/.../walking_reference.csv
                |
                v
supermimic-asimov/scripts/convert_asimov_walking_reference.py
                |
                v
data/asimov_walking_rootonly.npz
                |
                v
SuperMimic GenCoord training
```

## Source files

- CSV: `../asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/assets/walking_reference.csv`
- Robot MJCF: `../asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/xmls/asimov.xml`

## Convert the reference

Run the converter from the SuperMimic Asimov project:

```bash
cd ~/supermimic_ws/supermimic-asimov
uv run python scripts/convert_asimov_walking_reference.py \
  --csv ../asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/assets/walking_reference.csv \
  --mjcf ../asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/xmls/asimov.xml \
  --output ../data/asimov_walking_rootonly.npz \
  --fps 50 \
  --root-height 0.75
```

Check the generated file:

```bash
ls -lh ~/supermimic_ws/data/asimov_walking_rootonly.npz
```

The current conversion creates a stationary-root reference with the 12 joint trajectories and computed velocities. It should not be described as a complete travelling-locomotion retargeting pipeline.

## Inspect files and dataset arrays

```bash
cd ~/supermimic_ws
find data -maxdepth 4 -type f -printf '%p (%k KB)\n' | sort
find . -type f -name '*.csv' -not -path '*/.venv/*' | sort
find . -type f -name '*.npz' -not -path '*/.venv/*' | sort
```

Inspect the current NPZ keys and shapes:

```bash
cd ~/supermimic_ws
python - <<'PY'
import numpy as np

path = "data/asimov_walking_rootonly.npz"
with np.load(path) as data:
    print("File:", path)
    print("Keys:", data.files)
    for key in data.files:
        array = data[key]
        print(f"{key}: shape={array.shape}, dtype={array.dtype}")
PY
```

## Other reference-generation utility

The workspace also contains `~/supermimic_ws/make_ref_v7.py`. Inspect the script before assuming its input format or output options:

```bash
cd ~/supermimic_ws
sed -n '1,240p' make_ref_v7.py
```

## Data-handling rules

- Preserve original reference CSV files.
- Keep generated references in this directory.
- Do not overwrite source data with converted outputs.
- Record conversion FPS and root-height settings when creating a new reference.
- Inspect NPZ keys and array shapes before training.
- Keep model checkpoints and training logs separate from motion-reference data.
