# SuperMimic on Asimov 12-DoF Legs

## Project objective

Port and validate SuperMimic's GenCoord motion-imitation method on the Asimov 12-DoF biped in MJLab. The central learned object is the differentiable neural dynamics model of the Asimov simulation. The policy is improved through imagined rollouts of that learned model using backpropagation through time (BPTT).

This workspace is **not** a conventional PPO locomotion project.

## The actual learning loop

```text
Asimov MJLab environment
          |
          | current policy acts in the simulator
          v
real state/action/next-state transitions
          |
          v
SuperMimic replay buffer
          |
          +------------------------------+
          |                              |
          v                              |
WorldModelLearner fits the neural       |
dynamics model on replay windows       |
          |                              |
          v                              |
Differentiable learned dynamics         |
          |                              |
          v                              |
Imagined rollouts + BPTT policy update <+
          |
          v
world_model.pt + actor.pt (+ critic.pt)
```

**The motion reference is the imitation target. It is not the dynamics-training dataset.** Dynamics transitions are collected by stepping the Asimov MJLab simulator and are inserted into SuperMimic's replay buffer. The runner then trains the world model from those transitions and improves the actor through imagined rollouts.

## What is implemented in this checkout

- `supermimic-asimov/src/supermimic/scripts/train.py` constructs `SuperMimicRunner`, not `PpoRunner`.
- `SuperMimicRunner` collects real MJLab transitions, updates `WorldModelLearner`, updates the policy learner using the learned world-model environment, and saves `world_model.pt` and `actor.pt`.
- The default `ActorCriticPolicyLearner` is **not PPO**. It subclasses `BpttPolicyLearner`: the actor's return objective is differentiated through the imagined rollout and the frozen learned dynamics. Its critic is fitted separately using GAE.
- `world_model.pt` stores the `dynamics` and `sim_context` state dictionaries.
- The Asimov GenCoord state contains root state and all 12 joint positions/velocities. The current motion command uses the pelvis as the tracked anchor.

## Important current limitation

The Asimov motion adapter currently provides a root-only FK interface and the GenCoord task tracks the pelvis anchor plus the 12 generalized joint coordinates. This is **not** a MaxCoord/full-body Cartesian tracking implementation. The CSV also contains four Cartesian marker groups (`rf`, `lf`, `rk`, `lk`). The converter in this package preserves those source columns for analysis, but they are not currently part of the SuperMimic training loss. Do not claim foot/knee marker tracking until their frames, body/site correspondences, and losses are implemented and validated.

The CSV does not contain a pelvis trajectory. The corrected converter therefore keeps the pelvis fixed at the Asimov home height instead of inventing a forward velocity. Its output is a joint-trajectory / walk-in-place reference for the current GenCoord smoke workflow, not a travelling-walking reference.

## Workspace layout

| Directory | Role |
|---|---|
| `asimov-mjlab/` | Asimov MJCF, assets, actuator configuration, and MJLab simulation foundation. |
| `supermimic/` | Upstream SuperMimic checkout retained for source comparison. Its upstream README is left unchanged. |
| `supermimic-asimov/` | Working Asimov adaptation, GenCoord task, world-model runner, conversion and playback scripts. |
| `asimov_supermimic_overlay/` | Integration overlay and application utility. |
| `mjlab_research/` | Research checkout. |
| `data/` | Converted motion references and other explicitly documented motion data. |
| `make_ref_v7.py` | Separate reference-generation utility; inspect its CLI before using it. |

## Setup

The MJLab and SuperMimic adaptation use separate environments:

```bash
cd ~/supermimic_ws/asimov-mjlab
uv sync

cd ~/supermimic_ws/supermimic-asimov
uv sync
```

Verify imports:

```bash
cd ~/supermimic_ws/supermimic-asimov
uv run python -c "import mjlab.tasks, supermimic.tasks; print('Asimov SuperMimic imports OK')"
```

## Prepare the current joint-motion reference

The source file is:

`asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/assets/walking_reference.csv`

The CSV contains 12 joint trajectories and four Cartesian marker groups. It does not provide a root trajectory. Generate a fixed-root reference:

```bash
cd ~/supermimic_ws/supermimic-asimov
uv run python scripts/convert_asimov_walking_reference.py \
  --csv ../asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/assets/walking_reference.csv \
  --mjcf ../asimov-mjlab/src/mjlab/asset_zoo/robots/asimov/xmls/asimov.xml \
  --output ../data/asimov_walking_rootonly.npz \
  --fps 50 \
  --root-height 0.75
```

Validate the generated file:

```bash
uv run python scripts/validate_asimov_reference.py \
  ../data/asimov_walking_rootonly.npz
```

The validator checks shapes, finite values, joint metadata, marker preservation (when present), and that the root is fixed. It does not prove the reference is physically stable or validate the marker coordinate frame.

## Train SuperMimic, not PPO

Registered Asimov task:

`SuperMimic-GenCoord-Asimov-12DOF`

Start with a small smoke run:

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

This is the main training entry point. It trains the neural dynamics model from MJLab transitions and optimizes the motion-tracking policy through the learned model. There is no PPO training step in this command.

## Where the learned parameters are saved

The runner creates timestamped local directories under:

`supermimic-asimov/logs/SuperMimic-GenCoord-Asimov-12DOF/`

At each configured save interval (default `save_interval=1000`), the runner writes:

- `world_model.pt`: `dynamics` and `sim_context` state dictionaries.
- `actor.pt`: actor parameters.
- `critic.pt`: saved when the selected policy learner has a critic.

Find the newest world model:

```bash
cd ~/supermimic_ws/supermimic-asimov
find logs/SuperMimic-GenCoord-Asimov-12DOF -type f -name world_model.pt \
  -printf '%T+ %p\n' 2>/dev/null | sort -r | head -10
```

Inspect a saved world model:

```bash
uv run python scripts/inspect_world_model.py \
  logs/SuperMimic-GenCoord-Asimov-12DOF/<timestamp>/world_model.pt
```

This confirms that the expected parameter groups exist and contain finite tensors. It is a checkpoint-integrity check, **not** a dynamics-accuracy evaluation. Low world-model training loss alone is also not proof of multi-step predictive accuracy; a held-out real-simulation rollout comparison is still required.

## Playback

Inspect the playback CLI:

```bash
uv run src/supermimic/scripts/play.py --help
```

The local playback path expects a directory containing the required model files. Use the options shown by the actual CLI for that checkout; do not assume a checkpoint directory flag without checking `--help`.

## Remove/avoid the unrelated PPO experiment

The cleanup patch removes the separate Asimov-adaptation PPO entry point and runner:

- `supermimic-asimov/src/supermimic/scripts/train_ppo.py`
- `supermimic-asimov/src/supermimic/runners/ppo_runner.py`
- `asimov-mjlab/src/mjlab/tasks/velocity/config/asimov/` (Asimov-specific PPO velocity-task registrations and configs)
- `asimov-mjlab/asimov_ppo_rollout.npz` (the explicitly named PPO rollout artifact). It does **not** delete generic MJLab or upstream framework internals. Those are simulator/framework dependencies and are not the algorithm used by `SuperMimicRunner`. In particular, do not remove `rsl_rl` merely because it also provides PPO: this checkout's SuperMimic runner imports `rsl_rl.models.MLPModel` for network construction. The generic MJLab training APIs remain in the underlying dependency, but no Asimov PPO task is registered by this workspace after the patch.

## Current validation boundary

Before claiming a successful Asimov port, validate at least:

1. The task builds and the robot exposes exactly 12 actions and 12 joints in the expected order.
2. The world-model replay buffer receives real Asimov MJLab transitions.
3. `world_model.pt` is written and reloads with finite parameters.
4. Held-out one-step and multi-step world-model predictions are compared against real MJLab transitions.
5. The actor is evaluated against the motion reference with joint-position/velocity tracking metrics and a rendered rollout.
6. Any future foot/knee Cartesian tracking is separately validated against the Asimov MJCF and marker coordinate conventions.

A completed training run or a saved `world_model.pt` is not, by itself, proof that the model learned accurate Asimov dynamics or that the policy imitates locomotion successfully.
