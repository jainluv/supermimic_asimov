# SuperMimic Asimov

## Role in this workspace

This is the main working project for adapting SuperMimic to the Asimov robot's 12-DoF legs. Use it to convert motion references, register and validate the Asimov task, train GenCoord, run the PPO alternative, play back policies, and benchmark saved models.

The original framework checkout is the sibling `../supermimic/`. The Asimov project uses the sibling `../asimov-mjlab/` checkout as its local MJLab dependency.

## Important files

```text
scripts/convert_asimov_walking_reference.py
src/supermimic/scripts/train.py
src/supermimic/scripts/train_ppo.py
src/supermimic/scripts/play.py
src/supermimic/scripts/benchmark.py
src/supermimic/tasks/
pyproject.toml
```

| File/location | Purpose |
|---|---|
| `scripts/convert_asimov_walking_reference.py` | CSV-to-NPZ motion-reference conversion. |
| `src/supermimic/scripts/train.py` | Main SuperMimic/GenCoord training entry point. |
| `src/supermimic/scripts/train_ppo.py` | PPO training alternative. |
| `src/supermimic/scripts/play.py` | Policy playback. |
| `src/supermimic/scripts/benchmark.py` | Model benchmarking. |
| `src/supermimic/tasks/` | Task implementations and registration. |

## 1. Set up the environment

```bash
cd ~/supermimic_ws/supermimic-asimov
uv sync
uv run python -c "import mjlab, supermimic; print('Imports OK')"
```

This project is configured for Python 3.13 and uses the local `../asimov-mjlab` checkout as an editable dependency.

## 2. Convert the Asimov walking reference

Inputs:

- CSV: `../asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/assets/walking_reference.csv`
- MJCF: `../asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/xmls/asimov.xml`

Run:

```bash
cd ~/supermimic_ws/supermimic-asimov
uv run python scripts/convert_asimov_walking_reference.py \
  --csv ../asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/assets/walking_reference.csv \
  --mjcf ../asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/xmls/asimov.xml \
  --output ../data/asimov_walking_rootonly.npz \
  --fps 50 \
  --root-height 0.75
```

Inspect converter options with:

```bash
uv run python scripts/convert_asimov_walking_reference.py --help
```

Verify the output:

```bash
ls -lh ../data/asimov_walking_rootonly.npz
```

The current converter produces a stationary-root reference containing the 12 joint trajectories and computed velocities. It is not a complete travelling-locomotion retargeting pipeline.

## 3. Validate task registration

```bash
cd ~/supermimic_ws/supermimic-asimov
uv run python -c "import mjlab.tasks; import supermimic.tasks; print('Task imports OK')"
```

Current Asimov task ID:

`SuperMimic-GenCoord-Asimov-12DOF`

## 4. Train GenCoord

Start with a smoke test:

```bash
cd ~/supermimic_ws/supermimic-asimov
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
uv run src/supermimic/scripts/train.py \
  SuperMimic-GenCoord-Asimov-12DOF \
  --motion_file ../data/asimov_walking_rootonly.npz \
  --num_envs 64 \
  --runner.run_name asimov_gencoord_rootonly_smoke \
  --runner.sampling-mode uniform
```

See all supported options:

```bash
uv run src/supermimic/scripts/train.py --help
```

- `--motion_file` points to the converted local motion reference.
- `--num_envs` sets the number of parallel environments.
- `--runner.run_name` labels the run.
- `--runner.sampling-mode uniform` selects uniform motion sampling.
- `WANDB_MODE=disabled` disables WandB logging for this run.
- `CUDA_VISIBLE_DEVICES=0` selects GPU 0.

The local motion-file path disables WandB registry motion loading. The current Asimov integration uses uniform sampling. Do not use `return` sampling as a substitute for uniform sampling in this workflow. Increase environment count only after checking GPU memory and confirming the smoke test.

## 5. PPO alternative

A separate PPO entry point is available:

```bash
uv run src/supermimic/scripts/train_ppo.py --help
```

Read its help and configuration before launching a long run. It is a different runner from the main GenCoord path.

## 6. Find checkpoints and logs

```bash
cd ~/supermimic_ws/supermimic-asimov

# Model files
find . -type f \( -name actor.pt -o -name world_model.pt \) \
  -not -path '*/.venv/*' -printf '%T+ %p\n' | sort -r | head -30

# Log directories and TensorBoard event files
find . -type d -name logs -not -path '*/.venv/*'
find . -type f -name 'events.out.tfevents.*' -not -path '*/.venv/*'

# GPU use
nvidia-smi
```

The SuperMimic runners save model files such as `actor.pt` and `world_model.pt`.

## 7. Play back a trained policy

First inspect the installed playback interface:

```bash
uv run src/supermimic/scripts/play.py --help
```

The local-checkpoint workflow uses a directory containing `actor.pt` and `world_model.pt`. Command shape:

```bash
uv run src/supermimic/scripts/play.py \
  --checkpoint_dir /absolute/path/to/checkpoint/directory \
  --motion_file ../data/asimov_walking_rootonly.npz
```

Replace the example path with the real checkpoint directory found in Section 6. Check `--help` for any additional required options. Viewer choices include `auto`, `native`, and `viser`.

## 8. Benchmark

```bash
uv run src/supermimic/scripts/benchmark.py --help
```

Use the options shown by the benchmark script and provide the model directory containing the required model files.

## 9. Resume-training caution

Do not assume that `actor.pt` and `world_model.pt` contain optimizer state or the complete state needed to resume training. Verify the runner's checkpoint implementation before relying on a resume workflow. Playback from saved model files is a separate operation from resuming training.

## 10. Current scope

The Asimov integration currently focuses on GenCoord, pelvis/root tracking, and the 12 joint coordinates. Do not assume MaxCoord, full link-level forward-kinematics losses, foot-body tracking losses, travelling-locomotion retargeting, or real-robot deployment validation are already implemented.
