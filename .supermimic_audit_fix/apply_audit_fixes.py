#!/usr/bin/env python3
"""Apply the SuperMimic-Asimov repository audit fixes safely and reproducibly."""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import shutil
import subprocess
import sys
from pathlib import Path

PYPI_INDEX = "https://pypi.org/simple"


def fail(message: str) -> "NoReturn":
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        fail(f"Required file is missing: {path}")


def require_workspace(root: Path) -> None:
    required = [
        "README.md",
        "asimov-mjlab/pyproject.toml",
        "asimov-mjlab/README.md",
        "supermimic-asimov/pyproject.toml",
        "supermimic-asimov/README.md",
        "supermimic-asimov/src/supermimic/scripts/train.py",
        "supermimic-asimov/src/supermimic/runners/supermimic_runner.py",
        "supermimic-asimov/src/supermimic/learners/policy_learner.py",
        "asimov_supermimic_overlay/README.md",
    ]
    missing = [p for p in required if not (root / p).exists()]
    if missing:
        fail(
            "The supplied directory does not look like the supermimic_asimov "
            "workspace. Missing: " + ", ".join(missing)
        )


def replace_section(text: str, start: str, end: str, replacement: str, label: str) -> str:
    a = text.find(start)
    if a < 0:
        fail(f"Cannot find section heading {start!r} in {label}")
    if end == "":
        return text[:a] + replacement.rstrip() + "\n"
    b = text.find(end, a + len(start))
    if b < 0:
        fail(f"Cannot find following section heading {end!r} in {label}")
    return text[:a] + replacement.rstrip() + "\n\n" + text[b:]


def plan_changes(root: Path) -> dict[Path, str]:
    planned: dict[Path, str] = {}

    # Pin the version already exercised successfully with the pinned MuJoCo-Warp
    # revision, and stop sending MuJoCo package resolution to the special index.
    pyproject_path = root / "asimov-mjlab/pyproject.toml"
    pyproject = read_text(pyproject_path)
    pyproject = pyproject.replace('    "mujoco>=3.4.0",', '    "mujoco==3.4.0",')
    old_index = (
        '[[tool.uv.index]]\n'
        'name = "mujoco"\n'
        'url = "https://py.mujoco.org"\n'
        'explicit = true\n'
    )
    pyproject = pyproject.replace(old_index, "")
    pyproject = pyproject.replace('mujoco = { index = "mujoco" }\n', "")
    if '"mujoco==3.4.0"' not in pyproject:
        fail("Could not safely pin mujoco==3.4.0 in asimov-mjlab/pyproject.toml")
    if "py.mujoco.org" in pyproject or 'mujoco = { index = "mujoco" }' in pyproject:
        fail("The special MuJoCo index override is still present after patch planning")
    planned[pyproject_path] = pyproject

    # Give the adaptation package accurate metadata rather than upstream-generic
    # project wording.
    adap_pyproject_path = root / "supermimic-asimov/pyproject.toml"
    adap_pyproject = read_text(adap_pyproject_path)
    adap_pyproject = adap_pyproject.replace(
        'description = "Humanoid whole-body motion tracking with one differentiable world model."',
        'description = "Asimov 12-DoF GenCoord motion imitation with a differentiable world model."',
    )
    planned[adap_pyproject_path] = adap_pyproject

    gitignore_path = root / ".gitignore"
    if gitignore_path.exists():
        gitignore = read_text(gitignore_path)
        backup_rule = "/.audit_fix_backups/"
        if backup_rule not in gitignore.splitlines():
            gitignore = gitignore.rstrip() + "\n\n# Local safety backups created by repository repair scripts\n" + backup_rule + "\n"
        planned[gitignore_path] = gitignore

    root_readme_path = root / "README.md"
    root_readme = read_text(root_readme_path)
    root_readme = replace_section(
        root_readme,
        "## Remove/avoid the unrelated PPO experiment",
        "## Current validation boundary",
        """## Algorithm and cleanup boundary

The supported Asimov training entry point is `supermimic-asimov/src/supermimic/scripts/train.py`. It constructs `SuperMimicRunner`, which collects real MJLab transitions, trains `WorldModelLearner`, and updates the actor through imagined rollouts and BPTT.

The current workspace does **not** include an Asimov-specific conventional PPO velocity-task package, a standalone Asimov PPO runner, or a separate Asimov PPO training entry point. The legacy paths are:

- `supermimic-asimov/src/supermimic/scripts/train_ppo.py`
- `supermimic-asimov/src/supermimic/runners/ppo_runner.py`
- `asimov-mjlab/src/mjlab/tasks/velocity/config/asimov/`
- `asimov-mjlab/asimov_ppo_rollout.npz`

These historical Asimov-only paths are not part of the supported checkout and must not be restored as an alternative training route. They must not be restored as an alternative training route. This cleanup does **not** remove generic MJLab training infrastructure, generic RSL-RL PPO modules, or `rsl_rl.models.MLPModel`. The SuperMimic runner imports `MLPModel` for network construction; the presence of generic PPO APIs does not make this project a PPO training pipeline.

`ActorCriticPolicyLearner` remains part of the supported algorithm because it inherits from `BpttPolicyLearner`: the actor is optimized through the learned differentiable dynamics, while GAE is used to fit the critic separately. GAE critic fitting is not PPO.""",
        "root README",
    )
    setup_marker = "The MJLab and SuperMimic adaptation use separate environments:\n"
    setup_note = (
        "The Asimov MJLab project pins `mujoco==3.4.0`, the version exercised with "
        "the checked-in MuJoCo-Warp revision. MuJoCo is resolved from the configured "
        "PyPI index; the project must not route it through `py.mujoco.org`.\n\n"
    )
    if setup_marker in root_readme and setup_note not in root_readme:
        root_readme = root_readme.replace(setup_marker, setup_marker + "\n" + setup_note, 1)
    planned[root_readme_path] = root_readme

    mjlab_readme_path = root / "asimov-mjlab/README.md"
    mjlab_readme = read_text(mjlab_readme_path)
    mjlab_readme = replace_section(
        mjlab_readme,
        "## Role in this workspace",
        "## Important robot files",
        """## Role in this workspace

This checkout supplies the Asimov robot description and MJLab simulation environment used by the SuperMimic adaptation. It is not the project's training entry point. The current tree contains no Asimov-specific conventional PPO velocity-task registration package.

The main algorithmic workflow lives in the sibling directory `../supermimic-asimov/` and is launched through `src/supermimic/scripts/train.py` there.

The dependency configuration pins `mujoco==3.4.0` and resolves it from PyPI. This is the MuJoCo version exercised with the checked-in MuJoCo-Warp revision. Do not re-add the `py.mujoco.org` package-source override; it caused the resolver failure encountered in this workspace.""",
        "asimov-mjlab README",
    )
    mjlab_readme = replace_section(
        mjlab_readme,
        "## What not to do",
        "",
        """## Training boundary

Do not launch a separate conventional locomotion-training command as a substitute for the SuperMimic workflow. The current tree intentionally has no Asimov-specific PPO velocity-task registrations. Generic MJLab simulator, task-registry, training APIs, and RSL-RL model APIs remain because the simulation stack and SuperMimic network construction depend on them. They are not the algorithm used by the Asimov adaptation.

For the main experiment, use `../supermimic-asimov/README.md`.""",
        "asimov-mjlab README",
    )
    planned[mjlab_readme_path] = mjlab_readme

    adaptation_readme_path = root / "supermimic-asimov/README.md"
    adaptation_readme = read_text(adaptation_readme_path)
    old_para = (
        "The standalone PPO runner and entry point are removed by the cleanup patch, "
        "along with the separate Asimov MJLab PPO task registrations. The `ActorCriticPolicyLearner` "
        "used by `SuperMimicRunner` remains because it inherits from `BpttPolicyLearner` "
        "and differentiates the actor's imagined return through the learned model."
    )
    new_para = (
        "This checkout does not include a standalone Asimov PPO runner or `train_ppo.py` "
        "entry point, and the Asimov MJLab PPO velocity-task registration package is absent. "
        "The `ActorCriticPolicyLearner` used by `SuperMimicRunner` remains because it inherits "
        "from `BpttPolicyLearner` and differentiates the actor's imagined return through the "
        "learned model; GAE is used only for the critic update, not as a PPO policy update."
    )
    adaptation_readme = adaptation_readme.replace(old_para, new_para)
    planned[adaptation_readme_path] = adaptation_readme

    overlay_readme_path = root / "asimov_supermimic_overlay/README.md"
    overlay_readme = read_text(overlay_readme_path)
    old_line = (
        "The standalone `PpoRunner`, its `train_ppo.py` entry point, the Asimov MJLab "
        "PPO velocity-task registration package, and the named PPO rollout artifact are "
        "removed by the workspace cleanup patch. Generic MJLab simulator and RSL-RL model "
        "APIs are retained because this project depends on them; they are not the project's "
        "training algorithm."
    )
    new_line = (
        "The current workspace does not contain the standalone Asimov `PpoRunner`, its "
        "`train_ppo.py` entry point, the Asimov MJLab PPO velocity-task registration package, "
        "or the named PPO rollout artifact. Generic MJLab simulator and RSL-RL model APIs are "
        "retained because this project depends on them; they are not the project's training "
        "algorithm. The supported training route is the SuperMimic world-model/BPTT pipeline "
        "described above."
    )
    overlay_readme = overlay_readme.replace(old_line, new_line)
    planned[overlay_readme_path] = overlay_readme

    # Do not silently leave historical language claiming an installer/patch is what
    # makes the current branch clean. The only accepted mentions are explicit legacy paths.
    for path, content in planned.items():
        lowered = content.lower()
        if "cleanup installer" in lowered or "removed by the cleanup patch" in lowered or "removed by the workspace cleanup patch" in lowered:
            fail(f"Stale cleanup-history wording remains in {path.relative_to(root)}")

    return {path: content for path, content in planned.items() if path.read_text(encoding="utf-8") != content}


def backup_path(root: Path, backup_root: Path, source: Path) -> None:
    rel = source.relative_to(root)
    target = backup_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    elif source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def remove_legacy_paths(root: Path, backup_root: Path) -> list[str]:
    targets = [
        root / "supermimic-asimov/src/supermimic/scripts/train_ppo.py",
        root / "supermimic-asimov/src/supermimic/runners/ppo_runner.py",
        root / "asimov-mjlab/src/mjlab/tasks/velocity/config/asimov",
        root / "asimov-mjlab/asimov_ppo_rollout.npz",
    ]
    removed: list[str] = []
    for target in targets:
        if target.exists():
            backup_path(root, backup_root, target)
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            removed.append(str(target.relative_to(root)))
    return removed


def scan_python_sources(root: Path) -> list[str]:
    needles = (
        "asimov_ppo_runner_cfg",
        "Mjlab-Velocity-Rough-Asimov",
        "Mjlab-Velocity-Flat-Asimov",
        "Mjlab-Balance-Asimov",
        "asimov_ppo_rollout.npz",
        "mjlab.tasks.velocity.config.asimov",
        "from .asimov import",
    )
    scan_roots = [
        root / "asimov-mjlab/src/mjlab/tasks",
        root / "supermimic-asimov/src/supermimic",
    ]
    hits: list[str] = []
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*.py"):
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for needle in needles:
                if needle in content:
                    hits.append(f"{path.relative_to(root)} contains {needle!r}")
    return hits


def static_checks(root: Path) -> list[str]:
    errors: list[str] = []
    train = read_text(root / "supermimic-asimov/src/supermimic/scripts/train.py")
    runner = read_text(root / "supermimic-asimov/src/supermimic/runners/supermimic_runner.py")
    policy = read_text(root / "supermimic-asimov/src/supermimic/learners/policy_learner.py")
    mjlab_pyproject = read_text(root / "asimov-mjlab/pyproject.toml")
    adaptation_pyproject = read_text(root / "supermimic-asimov/pyproject.toml")

    if "from supermimic.runners.supermimic_runner import SuperMimicRunner" not in train:
        errors.append("train.py does not import SuperMimicRunner")
    if "PpoRunner" in train or "ppo_runner" in train:
        errors.append("train.py still refers to the standalone PPO runner")
    for required in ("WorldModelLearner", "policy_learner.update()", "world_model.pt", "actor.pt", "MLPModel"):
        if required not in runner:
            errors.append(f"SuperMimicRunner missing expected pipeline component: {required}")
    if "class ActorCriticPolicyLearner(BpttPolicyLearner)" not in policy:
        errors.append("ActorCriticPolicyLearner no longer inherits from BpttPolicyLearner")
    for required in ("self.wm_env.step(actions)", "gae_returns(", "self._rollout_actor"):
        if required not in policy:
            errors.append(f"policy_learner.py missing expected BPTT/GAE component: {required}")
    if '"mujoco==3.4.0"' not in mjlab_pyproject:
        errors.append("MJLab dependency is not pinned to mujoco==3.4.0")
    if "py.mujoco.org" in mjlab_pyproject or 'mujoco = { index = "mujoco" }' in mjlab_pyproject:
        errors.append("MJLab still routes mujoco to the py.mujoco.org index")
    if "rsl-rl-lib" not in mjlab_pyproject or "MLPModel" not in runner:
        errors.append("Generic RSL-RL network dependency required by SuperMimic is missing")

    # Parse project Python files without creating __pycache__ directories.
    for base in (root / "supermimic-asimov/src", root / "asimov-mjlab/src"):
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeDecodeError) as exc:
                errors.append(f"Python syntax/encoding error in {path.relative_to(root)}: {exc}")

    # The main documented scripts must exist.
    expected_scripts = [
        "supermimic-asimov/scripts/convert_asimov_walking_reference.py",
        "supermimic-asimov/scripts/validate_asimov_reference.py",
        "supermimic-asimov/scripts/inspect_world_model.py",
        "supermimic-asimov/src/supermimic/scripts/train.py",
        "supermimic-asimov/src/supermimic/scripts/play.py",
        "supermimic-asimov/src/supermimic/scripts/benchmark.py",
    ]
    for rel in expected_scripts:
        if not (root / rel).exists():
            errors.append(f"Documented script is missing: {rel}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="Path to ~/supermimic_ws")
    parser.add_argument(
        "--skip-lock",
        action="store_true",
        help="Patch and statically validate without regenerating uv.lock files.",
    )
    args = parser.parse_args()
    root = args.workspace.expanduser().resolve()
    require_workspace(root)

    planned = plan_changes(root)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = root / ".audit_fix_backups" / stamp
    backup_root.mkdir(parents=True, exist_ok=True)

    for path in planned:
        backup_path(root, backup_root, path)
    for path, content in planned.items():
        path.write_text(content, encoding="utf-8")

    removed = remove_legacy_paths(root, backup_root)

    print("[INFO] Documentation, metadata, and MuJoCo dependency source updated.")
    print(f"[INFO] Updated files: {len(planned)}")
    print("[INFO] Removed Asimov-only PPO remnants:")
    if removed:
        for item in removed:
            print(f"  - {item}")
    else:
        print("  - none found (already absent)")

    if not args.skip_lock:
        uv = shutil.which("uv")
        if not uv:
            fail("uv was not found. Install uv or rerun with --skip-lock, then regenerate locks manually.")
        for lock in (root / "asimov-mjlab/uv.lock", root / "supermimic-asimov/uv.lock"):
            if lock.exists():
                backup_path(root, backup_root, lock)
        for project in (root / "asimov-mjlab", root / "supermimic-asimov"):
            print(f"[INFO] Regenerating lock: {project.relative_to(root)}/uv.lock")
            try:
                subprocess.run(
                    [uv, "lock", "--default-index", PYPI_INDEX],
                    cwd=project,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                fail(
                    f"uv lock failed in {project.relative_to(root)} (exit {exc.returncode}). "
                    f"Review the error above; original files are backed up at {backup_root}."
                )
        for lock in (root / "asimov-mjlab/uv.lock", root / "supermimic-asimov/uv.lock"):
            if lock.exists() and "py.mujoco.org" in lock.read_text(encoding="utf-8", errors="replace"):
                fail(f"The stale py.mujoco.org source remains in {lock.relative_to(root)}")

    forbidden_hits = scan_python_sources(root)
    errors = static_checks(root)
    if forbidden_hits:
        errors.extend(["Asimov PPO registration reference remains: " + item for item in forbidden_hits])

    # Ensure the exact legacy paths are absent after cleanup.
    for rel in (
        "supermimic-asimov/src/supermimic/scripts/train_ppo.py",
        "supermimic-asimov/src/supermimic/runners/ppo_runner.py",
        "asimov-mjlab/src/mjlab/tasks/velocity/config/asimov",
        "asimov-mjlab/asimov_ppo_rollout.npz",
    ):
        if (root / rel).exists():
            errors.append(f"Legacy Asimov PPO path still exists: {rel}")

    if errors:
        print("[FAIL] Validation found problems:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(f"[INFO] Original files are backed up under: {backup_root}", file=sys.stderr)
        raise SystemExit(2)

    # The backup directory also records the exact rollback location.
    report = root / "REPOSITORY_AUDIT_2026-11-20.md"
    report_text = f"""# SuperMimic-Asimov repository audit and repair

Applied on: {dt.datetime.now().isoformat(timespec='seconds')}

## Changes

- Removed the special `py.mujoco.org` source override for the `mujoco` package.
- Pinned `mujoco==3.4.0`, the version exercised with the checked-in MuJoCo-Warp revision in this workspace.
{('- Regenerated both project lockfiles from the configured PyPI default index.' if not args.skip_lock else '- Lockfile regeneration was skipped by request; regenerate both lockfiles before relying on `uv sync`.')}
- Rewrote the workspace READMEs to describe the current algorithm and cleanup state without referring to a historical cleanup installer/patch as if it were still required.
- Removed any remaining Asimov-only PPO runner, entry point, task registration directory, and specifically named PPO rollout artifact at their known paths.

## Algorithm boundary checked

- `train.py` imports and constructs `SuperMimicRunner`.
- The runner contains the world-model learner and policy learner update path.
- `ActorCriticPolicyLearner` inherits from `BpttPolicyLearner`; actor updates use imagined rollouts through the learned model and GAE is used for critic fitting.
- Generic MJLab/RSL-RL APIs and `rsl_rl.models.MLPModel` were not removed.

## Validation performed

- Python AST syntax parsing across `supermimic-asimov/src` and `asimov-mjlab/src`.
- Static checks for the SuperMimic/BPTT/GAE algorithm path.
- Static checks for the absence of known Asimov PPO registrations and named legacy artifacts.
- Existence checks for the scripts referenced by the adaptation README.
- Lockfile check for stale `py.mujoco.org` references.

These are repository/static checks, not a simulator runtime test, a world-model held-out prediction test, or a successful imitation evaluation.

## Rollback

The pre-edit files are backed up at:

`{backup_root.relative_to(root)}`

## Next engineering validation

Run the documented import smoke test and a short SuperMimic run. Confirm the first world-model and policy updates actually execute after replay-buffer warm-up. Then run held-out one-step and multi-step world-model prediction tests and a rendered actor evaluation before claiming locomotion success.
"""
    report.write_text(report_text, encoding="utf-8")

    print("[PASS] Static repository audit completed.")
    print(f"[INFO] Backup: {backup_root}")
    print(f"[INFO] Audit report: {report}")
    print("[INFO] No virtual environment was synced, deleted, or recreated.")


if __name__ == "__main__":
    main()
