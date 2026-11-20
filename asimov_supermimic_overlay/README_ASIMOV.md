# SuperMimic + Asimov-MJLab integration overlay

This is an isolated first-pass port for **SuperMimic GenCoord on the 12-DOF Asimov leg model**. It does not edit the upstream repositories. Apply it to a separate copy of SuperMimic.

## What this overlay changes

- Points the integration copy's `mjlab` dependency at the sibling `asimov-mjlab` repository.
- Adapts SuperMimic's manager-term imports to the Asimov fork's module layout.
- Adds `SuperMimic-GenCoord-Asimov-12DOF` using Asimov-MJLab's own `make_tracking_env_cfg()` and `get_asimov_robot_cfg()`.
- Adds a motion-command adapter to provide the root aliases, one-clip `motion_ends`, and GenCoord FK contract expected by SuperMimic.
- Adds a converter for Asimov's `walking_reference.csv` to the Asimov-MJLab NPZ motion format.

## Intentional scope limits

1. **GenCoord only.** MaxCoord is not registered.
2. **Root-body tracking plus 12 joint coordinates.** The command tracks `pelvis_link` as its one body. The root-only FK adapter is consistent for that configured single body, but does not produce link-level FK or foot-body losses.
3. **The CSV converter makes a stationary-root gait clip.** It preserves the 12 joint trajectories and computes their velocities, but fixes pelvis position/orientation. It is an interface smoke test, not a travelling locomotion retargeter or a sim2real dataset.
4. **Uniform sampling is required for the first port.** SuperMimic's default `return` sampling maps to MJLab's `external` mode and requires `motion_end` plus `set_sampling_distribution`. Asimov-MJLab's MotionCommand supports `adaptive`, `uniform` and `start`, but not that external interface. Train with `--runner.sampling-mode uniform`; do not use the default `return` mode yet.
5. This overlay is syntax-checked but cannot be runtime-tested against the user's local GPU/checkouts from this build environment. The import, environment-construction and one-step smoke tests below are mandatory before training.

## Apply without modifying the original checkouts

From `~/supermimic_ws`:

```bash
rsync -a \
  --exclude='.venv' --exclude='logs' --exclude='__pycache__' \
  supermimic/ supermimic-asimov/
```

Unpack this overlay somewhere convenient, then run:

```bash
python /path/to/asimov_supermimic_overlay/apply_asimov_overlay.py \
  --project ~/supermimic_ws/supermimic-asimov \
  --asimov-mjlab ../asimov-mjlab
```

The script refuses to overwrite an existing overlay file and checks that its patch anchors are unique.

## Create a separate Python environment

```bash
cd ~/supermimic_ws/supermimic-asimov
uv lock
UV_PROJECT_ENVIRONMENT=../.venv-asimov uv sync
source ~/supermimic_ws/.venv-asimov/bin/activate
```

This keeps the original `~/supermimic_ws/.venv` untouched. If `uv sync` proposes a major CUDA/PyTorch change, stop and inspect the resolution instead of accepting it blindly.

## Convert the supplied Asimov walking CSV

The MJCF joint order is read from the actual Asimov model, and the CSV columns are reordered by joint name. This avoids assuming the CSV's left/right-interleaved header order matches the MuJoCo articulation order.

```bash
cd ~/supermimic_ws/supermimic-asimov
python scripts/convert_asimov_walking_reference.py \
  --csv ../asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/assets/walking_reference.csv \
  --mjcf ../asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/xmls/asimov.xml \
  --output ../data/asimov_walking_rootonly.npz \
  --fps 50 \
  --root-height 0.75
```

`50 FPS` matches the Asimov tracking environment's 0.005 s simulator timestep and decimation 4, but verify the original CSV sampling rate before treating the generated velocity labels as ground truth.

## First import/registry check

```bash
cd ~/supermimic_ws/supermimic-asimov
source ~/supermimic_ws/.venv-asimov/bin/activate
WANDB_MODE=disabled python - <<'PY'
import mjlab.tasks
import supermimic.tasks
from mjlab.tasks.registry import load_env_cfg

cfg = load_env_cfg("SuperMimic-GenCoord-Asimov-12DOF", play=True)
print("MJLab:", __import__("mjlab").__file__)
print("Robot:", cfg.scene.entities["robot"])
print("Motion bodies:", cfg.commands["motion"].body_names)
print("Motion command class:", cfg.commands["motion"].class_type)
print("Observation groups:", tuple(cfg.observations))
PY
```

If this passes, next create the environment with `--motion_file ../data/asimov_walking_rootonly.npz` and run a one-environment, CPU playback/reset smoke test before any training.

## First training command (only after smoke test)

```bash
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
uv run --active src/supermimic/scripts/train.py \
  SuperMimic-GenCoord-Asimov-12DOF \
  --motion_file ../data/asimov_walking_rootonly.npz \
  --num_envs 64 \
  --runner.run_name asimov_gencoord_rootonly_smoke \
  --runner.sampling-mode uniform
```

`uniform` is mandatory until an Asimov-specific external reset-sampling adapter and `motion_end` termination are implemented.

Start with 64 environments on the 6 GB RTX 4050 rather than 256. Increase only after measuring peak VRAM and confirming rollout stability.
