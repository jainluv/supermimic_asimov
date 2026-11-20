# SuperMimic on Asimov Legs

## Project objective

This workspace adapts SuperMimic's motion-imitation framework to the Asimov robot's 12-DoF legs. The intended result is a simulated Asimov robot that learns to reproduce reference leg motion, with policy playback and evaluation used to inspect the learned behaviour.

The project combines:
- **SuperMimic:** motion imitation and policy learning.
- **Asimov:** target robot description, actuated joints, and reference walking data.
- **MJLab:** MuJoCo-based simulation and robot environments.
- **GenCoord:** the current SuperMimic training approach used by the Asimov integration.
- **Motion references:** source CSV trajectories converted to NPZ files for training.

Simulation success alone does not establish sim-to-real readiness.

## 1. Workspace map

```text
supermimic_ws/
├── asimov-mjlab/                  # Asimov model, MJLab tasks, conventional PPO
│   └── src/mjlab/
│       ├── asset_zoo/robots/asimov/
│       │   ├── assets/walking_reference.csv
│       │   └── xmls/asimov.xml
│       ├── tasks/velocity/config/asimov/
│       │   └── rl_cfg.py
│       └── scripts/
├── supermimic/                    # Original upstream SuperMimic checkout
├── supermimic-asimov/             # Working Asimov SuperMimic adaptation
│   ├── scripts/convert_asimov_walking_reference.py
│   └── src/supermimic/
│       ├── scripts/{train,train_ppo,play,benchmark}.py
│       └── tasks/
├── asimov_supermimic_overlay/     # Asimov integration overlay
│   ├── apply_asimov_overlay.py
│   └── README_ASIMOV.md
├── mjlab_research/                # Research and experiments
├── data/                          # Source and converted motion data
├── make_ref_v7.py                 # Separate reference-generation utility
└── README.md                      # This guide
```

This is a map of the important project locations, not an exhaustive listing of every source file.

## 2. What each directory is for

| Directory | Role | Work here when you want to... |
|---|---|---|
| `asimov-mjlab/` | Robot and simulator foundation | Inspect the Asimov MJCF, robot assets, MJLab task configuration, or run conventional PPO. |
| `supermimic/` | Upstream framework | Read or compare original SuperMimic implementation. |
| `supermimic-asimov/` | Main working adaptation | Convert motion references, train the Asimov GenCoord task, play policies, or benchmark checkpoints. |
| `asimov_supermimic_overlay/` | Integration changes | Inspect or apply Asimov-specific changes to the working adaptation. |
| `mjlab_research/` | Experiments | Keep exploratory code separate from the main training path. |
| `data/` | Motion inputs/outputs | Store and inspect CSV/NPZ motion-reference data. |

**Important:** `supermimic/` is the upstream checkout. `supermimic-asimov/` is the working adaptation. Run Asimov SuperMimic commands from `supermimic-asimov/`, not from the upstream directory.

## 3. End-to-end pipeline

```text
Asimov MJCF + walking reference CSV
                 |
                 v
      Asimov reference converter
                 |
                 v
       data/*.npz motion reference
                 |
                 v
      SuperMimic Asimov GenCoord task
                 |
                 v
          MJLab simulation
                 |
                 v
          Policy training
                 |
                 v
       actor.pt + world_model.pt
                 |
                 v
          Policy playback
                 |
                 v
       Inspect motion and evaluate
```

The current converter produces a stationary-root reference containing the 12 joint trajectories and computed velocities. It is not a complete travelling-locomotion retargeting pipeline.

## 4. Set up the two main environments

The projects use separate Python environments. Run setup separately:

```bash
cd ~/supermimic_ws/asimov-mjlab
uv sync

cd ~/supermimic_ws/supermimic-asimov
uv sync
```

Check imports:

```bash
cd ~/supermimic_ws/asimov-mjlab
uv run python -c "import mjlab; print('MJLab OK')"

cd ~/supermimic_ws/supermimic-asimov
uv run python -c "import mjlab, supermimic; print('SuperMimic and MJLab OK')"
```

## 5. Convert the Asimov walking reference

Source reference:

`asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/assets/walking_reference.csv`

Robot MJCF:

`asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/xmls/asimov.xml`

Run the converter:

```bash
cd ~/supermimic_ws/supermimic-asimov
uv run python scripts/convert_asimov_walking_reference.py \
  --csv ../asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/assets/walking_reference.csv \
  --mjcf ../asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/xmls/asimov.xml \
  --output ../data/asimov_walking_rootonly.npz \
  --fps 50 \
  --root-height 0.75
```

Check that the file was created:

```bash
ls -lh ~/supermimic_ws/data/asimov_walking_rootonly.npz
```

## 6. Validate task registration

```bash
cd ~/supermimic_ws/supermimic-asimov
uv run python -c "import mjlab.tasks; import supermimic.tasks; print('Task imports OK')"
```

The Asimov task ID used by this workspace is:

`SuperMimic-GenCoord-Asimov-12DOF`

## 7. Train the Asimov GenCoord policy

Start with a modest environment count to validate the pipeline:

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

Inspect the script's supported arguments before changing settings:

```bash
uv run src/supermimic/scripts/train.py --help
```

The local `--motion_file` workflow disables WandB registry motion loading. The current Asimov integration uses uniform sampling; do not substitute the `return` mode for this workflow. Increase `--num_envs` only after checking GPU memory and confirming that the smaller run behaves correctly.

## 8. Find logs and checkpoints

```bash
cd ~/supermimic_ws/supermimic-asimov

# Find model files
find . -type f \( -name actor.pt -o -name world_model.pt \) \
  -not -path '*/.venv/*' -printf '%T+ %p\n' | sort -r | head -30

# Find log directories and TensorBoard events
find . -type d -name logs -not -path '*/.venv/*'
find . -type f -name 'events.out.tfevents.*' -not -path '*/.venv/*'

# GPU status
nvidia-smi
```

## 9. View a trained policy

Inspect the playback interface:

```bash
cd ~/supermimic_ws/supermimic-asimov
uv run src/supermimic/scripts/play.py --help
```

The local playback workflow uses a checkpoint directory containing `actor.pt` and `world_model.pt`. The command shape is:

```bash
uv run src/supermimic/scripts/play.py \
  --checkpoint_dir /absolute/path/to/checkpoint/directory \
  --motion_file ../data/asimov_walking_rootonly.npz
```

Replace the example checkpoint directory with the actual directory found in Section 8. Check `--help` for the exact options required by the installed checkout. The viewer configuration supports `auto`, `native`, and `viser`.

## 10. Benchmark

```bash
cd ~/supermimic_ws/supermimic-asimov
uv run src/supermimic/scripts/benchmark.py --help
```

Use the arguments shown by the benchmark script and point it at the model directory containing the required checkpoint files.

## 11. Conventional MJLab PPO is a separate workflow

To list conventional MJLab environments:

```bash
cd ~/supermimic_ws/asimov-mjlab
uv run list_envs
```

Inspect train and playback options:

```bash
uv run train --help
uv run play --help
```

Train a task using its exact registered ID from `list_envs`:

```bash
uv run train TASK_ID
```

This is not the SuperMimic GenCoord training command.

## 12. Current integration scope and limitations

The current Asimov integration focuses on GenCoord, pelvis/root tracking, and the 12 joint coordinates. Do not assume that it already implements MaxCoord, full link-level forward-kinematics losses, foot-body tracking losses, a complete travelling-locomotion retargeter, or a validated real-robot deployment pipeline.

A saved actor/world-model pair is sufficient for the documented playback workflow; it does not prove that optimizer state and all training state can be resumed. Verify the runner's checkpoint implementation before relying on training resume.

## 13. Where to go next

- Asimov robot model and conventional PPO: `asimov-mjlab/README.md`
- Main Asimov GenCoord training and playback: `supermimic-asimov/README.md`
- Upstream framework checkout: `supermimic/WORKSPACE_GUIDE.md`
- Integration overlay: `asimov_supermimic_overlay/README.md`
- Motion-reference data: `data/README.md`
