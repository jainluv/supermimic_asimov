# Upstream SuperMimic: Workspace Guide

## Purpose

This directory is the original SuperMimic source checkout. It is kept as the upstream framework reference for the Asimov adaptation. The active Asimov implementation and training workflow live in the sibling directory `../supermimic-asimov/`.

The upstream `README.md` is intentionally not replaced by this workspace-specific guide.

## How the checkouts relate

```text
supermimic_ws/
├── supermimic/          # Original framework source
├── supermimic-asimov/   # Working Asimov adaptation
├── asimov-mjlab/        # Asimov robot and MJLab integration
└── asimov_supermimic_overlay/ # Asimov integration overlay
```

## Inspect upstream source

```bash
cd ~/supermimic_ws/supermimic

# Source directories
find src -maxdepth 4 -type d | sort | head -100

# Main workflow scripts
find src -type f \( -name train.py -o -name train_ppo.py -o -name play.py -o -name benchmark.py \) -print | sort

# Task directories
find src -type d -iname '*task*' -print
```

## Compare upstream and Asimov working copy

From the workspace root:

```bash
cd ~/supermimic_ws
diff -qr --exclude='.venv' --exclude='logs' --exclude='wandb' \
  --exclude='__pycache__' supermimic supermimic-asimov | less
```

This helps identify differences between the original framework and the Asimov working adaptation.

## When to use this directory

Use it to inspect original algorithm implementations, compare upstream files, and understand framework behaviour. Do not launch Asimov training from this checkout unless you have deliberately configured it for the Asimov task and dependencies.

For Asimov reference conversion, GenCoord training, checkpoint discovery, playback, and benchmarking, use `../supermimic-asimov/README.md`.
