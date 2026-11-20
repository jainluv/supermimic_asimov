# Asimov SuperMimic Overlay

## Purpose

This directory contains the Asimov-specific integration overlay used to adapt the SuperMimic framework for the Asimov robot. It is an integration and maintenance directory, not the main training directory.

The target working project is `~/supermimic_ws/supermimic-asimov/`. The upstream source checkout is `~/supermimic_ws/supermimic/`.

The existing `README_ASIMOV.md` is preserved.

## Inspect the overlay

```bash
cd ~/supermimic_ws/asimov_supermimic_overlay
find . -maxdepth 4 -type f -not -path '*/.git/*' | sort
less README_ASIMOV.md
python apply_asimov_overlay.py --help
```

## Before applying

Applying the overlay changes the target working project. Review and preserve any local changes you need before running the application script. Do not apply the overlay just to start or resume training.

## Apply/refresh workflow

The following is the integration workflow used in this workspace. Review the target contents and the script's current `--help` before running it:

```bash
cd ~/supermimic_ws

rsync -a \
  --exclude='.venv' \
  --exclude='logs' \
  --exclude='__pycache__' \
  supermimic/ supermimic-asimov/

python asimov_supermimic_overlay/apply_asimov_overlay.py \
  --project "$HOME/supermimic_ws/supermimic-asimov" \
  --asimov-mjlab "$HOME/supermimic_ws/asimov-mjlab"
```

## Validate the target after applying

```bash
cd ~/supermimic_ws/supermimic-asimov
uv sync
uv run python -c "import mjlab.tasks; import supermimic.tasks; print('Integration imports OK')"
grep -R -n 'SuperMimic-GenCoord-Asimov-12DOF' src/supermimic
```

## Integration scope

The current integration focuses on GenCoord, pelvis/root tracking, and the 12 joint coordinates. Do not assume MaxCoord, full link-level forward-kinematics losses, foot-body tracking losses, a complete travelling-locomotion retargeting pipeline, or sim-to-real deployment support are implemented unless the source has been changed and tested.

## Where to train and play

Run Asimov training and playback from `~/supermimic_ws/supermimic-asimov/`. See `../supermimic-asimov/README.md` for the reference-conversion, training, logs, checkpoint, playback, and benchmark commands.
