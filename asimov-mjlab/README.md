# Asimov MJLab simulation foundation

## Role in this workspace

This checkout supplies the Asimov robot description and MJLab simulation environment used by the SuperMimic adaptation. It is not the project's training entry point. The Asimov-specific conventional PPO velocity-task package is removed from this workspace by the cleanup installer.

The main algorithmic workflow lives in the sibling directory `../supermimic-asimov/` and is launched through `src/supermimic/scripts/train.py` there.

## Important robot files

```text
src/mjlab/asset_zoo/robots/asimov/
├── asimov_constants.py
├── assets/
│   └── walking_reference.csv
└── xmls/
    └── asimov.xml
```

- `xmls/asimov.xml`: MJCF kinematic tree, body/joint definitions, actuators, collision geometry and sensors.
- `asimov_constants.py`: Asimov robot configuration, actuator definitions, action scales and home keyframe.
- `assets/walking_reference.csv`: current source trajectory. Its header contains 12 joint channels and four 3D marker groups (`rf`, `lf`, `rk`, `lk`).

The current home keyframe in `asimov_constants.py` uses a root height of 0.75 m. The reference converter uses this as its fixed-root default unless explicitly overridden.

## Setup and import check

```bash
cd ~/supermimic_ws/asimov-mjlab
uv sync
uv run python -c "from mjlab.asset_zoo.robots.asimov.asimov_constants import get_asimov_robot_cfg; print(get_asimov_robot_cfg())"
```

## Inspect the robot and source reference

```bash
cd ~/supermimic_ws/asimov-mjlab

# Inspect the robot model
sed -n '1,220p' src/mjlab/asset_zoo/robots/asimov/xmls/asimov.xml

# Inspect CSV header and a few frames
head -5 src/mjlab/asset_zoo/robots/asimov/assets/walking_reference.csv
```

## How SuperMimic uses this project

`supermimic-asimov/pyproject.toml` points to this checkout as its local MJLab dependency. SuperMimic creates and steps the Asimov simulation, collects state/action/next-state transitions, and trains its differentiable dynamics model from those transitions.

The CSV is used as a motion-imitation target. It is **not** the world-model training dataset. The world-model training data is collected online from MJLab.

## What not to do

Do not launch a separate conventional locomotion-training command as a substitute for the SuperMimic workflow. The cleanup removes `src/mjlab/tasks/velocity/config/asimov/`, which registered Asimov PPO velocity tasks. Generic MJLab simulator, task-registry, and RSL-RL model APIs remain because the simulator integration and SuperMimic network construction depend on them.

For the main experiment, use `../supermimic-asimov/README.md`.
