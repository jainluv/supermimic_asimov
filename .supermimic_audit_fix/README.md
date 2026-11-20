# SuperMimic-Asimov repository audit fix

This bundle applies a focused cleanup to the `~/supermimic_ws` workspace.

## What it fixes

1. **MuJoCo resolver source:** removes the `py.mujoco.org` index override and package-source mapping that forced `mujoco` resolution through that index. Pins `mujoco==3.4.0`, the version already smoke-tested in this workspace with the checked-in MuJoCo-Warp revision.
2. **Lockfiles:** regenerates `asimov-mjlab/uv.lock` and `supermimic-asimov/uv.lock` using PyPI as the default index.
3. **Misleading documentation:** rewrites the root README, Asimov MJLab README, adaptation README, and overlay README so they describe the current repository state and the supported SuperMimic/BPTT route. It removes historical "cleanup patch/installer" wording.
4. **Stale Asimov PPO remnants:** removes only the known Asimov-specific entry point, runner, task-registration directory, and named rollout artifact if any remain locally.
5. **Validation:** checks that `train.py` uses `SuperMimicRunner`, that the runner/policy learner retain the world-model + BPTT + GAE-critic pipeline, that Python sources parse, and that documented scripts exist.

## What it deliberately does not do

- It does not delete generic MJLab PPO infrastructure.
- It does not delete `rsl_rl` or `rsl_rl.models.MLPModel`.
- It does not change SuperMimic's BPTT actor objective or GAE critic fitting.
- It does not change motion conversion, reward functions, robot XML, or training hyperparameters.
- It does not run `uv sync`, delete `.venv`, or start training.
- It does not claim that the world model or walking policy has been validated.

## Apply

From the workspace root, extract the bundle into `.supermimic_audit_fix` and run:

```bash
mkdir -p .supermimic_audit_fix
unzip -oq ~/Downloads/SuperMimic_Asimov_Audit_Fix.zip -d .supermimic_audit_fix
bash .supermimic_audit_fix/apply.sh "$PWD"
```

The script creates `.audit_fix_backups/<timestamp>/` before changing files. Lockfile regeneration requires `uv` and network access.

To patch and run static checks without regenerating lockfiles:

```bash
bash .supermimic_audit_fix/apply.sh "$PWD" --skip-lock
```

After applying, review the diff and commit it yourself. The script does not push to GitHub.

## Expected next checks

```bash
cd ~/supermimic_ws/supermimic-asimov
uv run python -c "import mjlab.tasks, supermimic.tasks; print('Asimov SuperMimic imports OK')"
```

Then run the README smoke-training command. Early iterations can be replay-buffer warm-up; do not interpret a zero update time or a saved checkpoint as proof of learned dynamics. Held-out one-step/multi-step model prediction and quantitative actor evaluation remain separate requirements.
