# Asimov MJLab

## Role in this workspace

This directory supplies the Asimov robot and MJLab simulation side of the project. It contains the robot description, assets, task configurations, and conventional PPO entry points. The sibling `supermimic-asimov/` project uses this checkout as its local MJLab dependency and reads the Asimov robot MJCF and walking-reference data from here.

## Important locations

```text
src/mjlab/asset_zoo/robots/asimov/
├── assets/walking_reference.csv
└── xmls/asimov.xml

src/mjlab/tasks/velocity/config/asimov/
└── rl_cfg.py

src/mjlab/scripts/
├── train.py
├── play.py
└── list_envs.py
```

The tree shows the key files for this workspace, not every file in the repository.

## Setup

```bash
cd ~/supermimic_ws/asimov-mjlab
uv sync
uv run python -c "import mjlab; print('MJLab import OK')"
```

## List available tasks

Do not guess task IDs. Ask the installed project to list them:

```bash
cd ~/supermimic_ws/asimov-mjlab
uv run list_envs
```

## Train a conventional PPO task

Inspect supported options:

```bash
uv run train --help
```

Then select the exact Asimov task ID printed by `uv run list_envs`:

```bash
uv run train TASK_ID
```

Replace `TASK_ID` with the actual registered ID. This workflow is conventional MJLab PPO, not SuperMimic GenCoord.

## View a policy

Inspect the playback interface first:

```bash
uv run play --help
```

Use the task and checkpoint arguments supported by that help output. Find likely model files with:

```bash
find logs -type f \( -name '*.pt' -o -name '*.pth' \) \
  -printf '%T+ %p\n' 2>/dev/null | sort -r | head -30
```

## Important configuration locations

- Asimov velocity task configurations: `src/mjlab/tasks/velocity/config/asimov/`
- PPO runner configuration: `src/mjlab/tasks/velocity/config/asimov/rl_cfg.py`
- Runner configuration definitions: `src/mjlab/rl/config.py`

The runner configuration includes `resume`, `load_run`, and `load_checkpoint` fields. Inspect the actual CLI and configuration before using overrides; do not assume a generic `--resume` flag exists:

```bash
uv run train --help
grep -R -n -E 'resume|load_run|load_checkpoint' src/mjlab/rl src/mjlab/tasks/velocity/config/asimov
```

## Logs and checkpoints

```bash
find logs -maxdepth 4 -type d 2>/dev/null | sort
find logs -type f \( -name '*.pt' -o -name '*.pth' \) \
  -printf '%T+ %p\n' 2>/dev/null | sort -r | head -30
find logs -type f -name 'events.out.tfevents.*' 2>/dev/null
```

If TensorBoard is available in this environment:

```bash
uv run tensorboard --logdir logs
```

## Connection to SuperMimic

The Asimov-specific imitation project is in `../supermimic-asimov/`. Its converter uses:

- `src/mjlab/asset_zoo/robots/asimov/assets/walking_reference.csv`
- `src/mjlab/asset_zoo/robots/asimov/xmls/asimov.xml`

The converted NPZ is stored in the workspace-level `data/` directory. See the root workspace README for the full CSV-to-NPZ-to-GenCoord pipeline.
