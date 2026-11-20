# SuperMimic Asimov adaptation

## Purpose

This is the working port of SuperMimic's GenCoord motion-imitation method to the Asimov 12-DoF legs. The core training target is a learned, differentiable neural dynamics model of the Asimov MJLab environment, followed by policy optimization through imagined rollouts of that model.

## Algorithm used

The supported training path is:

1. Run the current policy in the real MJLab simulation.
2. Store state, action, next-state, termination and simulator-context data in SuperMimic's replay buffer.
3. Train `WorldModelLearner` on replay windows to fit the neural dynamics model.
4. Roll out the learned model inside the differentiable world-model environment.
5. Update the actor by backpropagating the imagined return through the frozen learned dynamics (BPTT). The default actor-critic variant also fits a critic using GAE; that is not PPO.
6. Save the learned dynamics/context parameters and policy checkpoints.

`train.py` is the only supported training entry point for this project.

## Key files

```text
scripts/
├── convert_asimov_walking_reference.py
├── validate_asimov_reference.py
└── inspect_world_model.py

src/supermimic/
├── scripts/
│   ├── train.py
│   ├── play.py
│   └── benchmark.py
├── runners/
│   └── supermimic_runner.py
├── learners/
│   ├── wm_learner.py
│   └── policy_learner.py
├── models.py
├── replay_buffer.py
├── sim_context.py
├── representations/
├── backends/
└── tasks/
    ├── asimov_env_cfgs.py
    └── ...
```

This checkout does not include a standalone Asimov PPO runner or `train_ppo.py` entry point, and the Asimov MJLab PPO velocity-task registration package is absent. The `ActorCriticPolicyLearner` used by `SuperMimicRunner` remains because it inherits from `BpttPolicyLearner` and differentiates the actor's imagined return through the learned model; GAE is used only for the critic update, not as a PPO policy update.

## Setup

```bash
cd ~/supermimic_ws/supermimic-asimov
uv sync
uv run python -c "import mjlab.tasks, supermimic.tasks; print('imports OK')"
```

## Convert the Asimov reference

The source CSV is located in the MJLab checkout and includes 12 joint trajectories plus four Cartesian marker groups. The current supported GenCoord reference keeps the pelvis fixed and uses the 12 joint trajectories. The Cartesian columns are preserved in the NPZ for review but are not currently used as training loss terms.

```bash
cd ~/supermimic_ws/supermimic-asimov
uv run python scripts/convert_asimov_walking_reference.py \
  --csv ../asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/assets/walking_reference.csv \
  --mjcf ../asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/xmls/asimov.xml \
  --output ../data/asimov_walking_rootonly.npz \
  --fps 50 \
  --root-height 0.75
```

The converter does not fabricate root translation. Because the source file does not provide a root trajectory, this output is a fixed-root joint-motion reference, not a travelling locomotion clip.

Validate:

```bash
uv run python scripts/validate_asimov_reference.py ../data/asimov_walking_rootonly.npz
```

## Validate task registration

```bash
uv run python -c "import mjlab.tasks, supermimic.tasks; from mjlab.tasks.registry import list_tasks; print([t for t in list_tasks() if 'Asimov' in t])"
```

Expected Asimov task:

```text
SuperMimic-GenCoord-Asimov-12DOF
```

## Train the world model and motion policy

Start with a smoke run:

```bash
cd ~/supermimic_ws/supermimic-asimov
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
uv run src/supermimic/scripts/train.py \
  SuperMimic-GenCoord-Asimov-12DOF \
  --motion_file ../data/asimov_walking_rootonly.npz \
  --num_envs 64 \
  --runner.run_name asimov_gencoord_world_model_smoke \
  --runner.sampling-mode uniform
```

The command intentionally uses `src/supermimic/scripts/train.py`, which constructs `SuperMimicRunner`. It does not call the deleted standalone PPO runner.

Inspect available options:

```bash
uv run src/supermimic/scripts/train.py --help
```

## World-model checkpoint

The runner builds the `Dynamics` network from the Asimov environment dimensions. Its default hidden widths are `(2048, 2048, 2048)`. It learns from simulator transitions stored in the replay buffer.

The runner creates timestamped local directories:

```text
logs/SuperMimic-GenCoord-Asimov-12DOF/<timestamp>/
```

At each save interval (default 1000 iterations), it writes:

- `world_model.pt`: `dynamics` and `sim_context` state dictionaries.
- `actor.pt`: learned actor state dictionary.
- `critic.pt`: critic state dictionary when the actor-critic learner is selected.

Find and inspect the newest world model:

```bash
find logs/SuperMimic-GenCoord-Asimov-12DOF -type f -name world_model.pt \
  -printf '%T+ %p\n' 2>/dev/null | sort -r | head -10

uv run python scripts/inspect_world_model.py \
  logs/SuperMimic-GenCoord-Asimov-12DOF/<timestamp>/world_model.pt
```

The inspection script checks required state-dictionary groups, tensor counts and finite values. It does not calculate held-out prediction error. Before claiming that the world model learned accurate dynamics, evaluate its one-step and multi-step predictions against transitions held out from MJLab.

## Playback

```bash
uv run src/supermimic/scripts/play.py --help
```

Use the local checkpoint options documented by that CLI and point it to the run directory containing the required model files.

## Benchmark

```bash
uv run src/supermimic/scripts/benchmark.py --help
```

## Logs and checkpoints

```bash
# Latest world-model checkpoints
find logs/SuperMimic-GenCoord-Asimov-12DOF -type f -name world_model.pt \
  -printf '%T+ %p\n' 2>/dev/null | sort -r | head -20

# All actor/critic/world-model files
find logs/SuperMimic-GenCoord-Asimov-12DOF -type f \
  \( -name actor.pt -o -name critic.pt -o -name world_model.pt \) \
  -printf '%T+ %p\n' 2>/dev/null | sort -r | head -50

nvidia-smi
```

## What is still not complete

- The current GenCoord port tracks the pelvis as its anchor and represents the legs through the 12 generalized joint coordinates. It is not a MaxCoord full-body Cartesian implementation.
- The current CSV marker groups are preserved but are not used in the training loss.
- The reference contains no root trajectory, so the current converted motion is fixed-root.
- A saved checkpoint does not prove world-model prediction accuracy or imitation success.
- Held-out one-step/multi-step model validation and quantitative policy evaluation must be run before reporting success.

Do not describe the current integration as a validated full locomotion or sim-to-real port until these checks pass.
