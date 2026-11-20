# Upstream SuperMimic workspace guide

`supermimic/` is retained as the upstream source checkout for comparison. The Asimov-specific working adaptation is `../supermimic-asimov/`.

The upstream `README.md` is intentionally not replaced. Use this guide for the local workspace relationship only.

```text
supermimic_ws/
├── supermimic/          # Upstream source reference
├── supermimic-asimov/   # Asimov implementation and training
└── asimov-mjlab/        # Asimov model and simulator dependency
```

Compare source trees without generated environments and logs:

```bash
cd ~/supermimic_ws
diff -qr --exclude='.venv' --exclude='logs' --exclude='wandb' \
  --exclude='__pycache__' supermimic supermimic-asimov | less
```

Run Asimov training only from `supermimic-asimov/` using `src/supermimic/scripts/train.py`. That entry point constructs `SuperMimicRunner`, which trains the neural dynamics model from MJLab replay transitions and improves the actor through imagined rollouts. Do not use a standalone PPO runner as the project training route.
