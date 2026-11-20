# Asimov SuperMimic integration overlay

This directory contains the Asimov-specific integration overlay. It is not the training directory.

The working training project is `../supermimic-asimov/`. The supported algorithm path is `src/supermimic/scripts/train.py` → `SuperMimicRunner` → real MJLab transition collection → `WorldModelLearner` → imagined rollouts and BPTT policy updates.

Do not apply an overlay merely to refresh documentation. The existing `apply_asimov_overlay.py` modifies the target checkout; inspect its `--help`, review local changes, and back up the target before using it.

```bash
cd ~/supermimic_ws/asimov_supermimic_overlay
python apply_asimov_overlay.py --help
```

After any intentional overlay application, validate the adaptation:

```bash
cd ~/supermimic_ws/supermimic-asimov
uv sync
uv run python -c "import mjlab.tasks, supermimic.tasks; print('imports OK')"
```

The current workspace does not contain the standalone Asimov `PpoRunner`, its `train_ppo.py` entry point, the Asimov MJLab PPO velocity-task registration package, or the named PPO rollout artifact. Generic MJLab simulator and RSL-RL model APIs are retained because this project depends on them; they are not the project's training algorithm. The supported training route is the SuperMimic world-model/BPTT pipeline described above.
