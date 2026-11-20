#!/usr/bin/env python3
"""Apply the Asimov overlay to a separate SuperMimic working copy."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
  text = path.read_text()
  if new in text:
    print(f"[skip] {label}: already patched")
    return
  count = text.count(old)
  if count != 1:
    raise RuntimeError(
      f"Cannot safely patch {label}: expected one anchor, found {count} in {path}"
    )
  path.write_text(text.replace(old, new, 1))
  print(f"[patch] {label}")


def copy_overlay(source: Path, target: Path) -> None:
  for src in source.rglob("*"):
    if not src.is_file():
      continue
    rel = src.relative_to(source)
    if rel.name.endswith(".addendum"):
      continue
    dst = target / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.read_bytes() != src.read_bytes():
      raise FileExistsError(f"Refusing to overwrite existing file: {dst}")
    if not dst.exists():
      shutil.copy2(src, dst)
      print(f"[add] {dst.relative_to(target)}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--project", type=Path, required=True,
                      help="Path to the separate supermimic-asimov checkout")
  parser.add_argument("--asimov-mjlab", type=Path, default=Path("../asimov-mjlab"),
                      help="Path to the sibling asimov-mjlab checkout, relative to project")
  args = parser.parse_args()
  project = args.project.expanduser().resolve()
  if not (project / "pyproject.toml").is_file():
    raise SystemExit(f"Not a SuperMimic project directory: {project}")

  bundle_root = Path(__file__).resolve().parent
  copy_overlay(bundle_root / "overlay", project)

  # Asimov-MJLab re-exports CommandTermCfg only from mjlab.managers. The two
  # observation config classes live in manager_term_config in that fork.
  env_cfg = project / "src/supermimic/tasks/env_cfgs.py"
  replace_once(
    env_cfg,
    "from mjlab.managers import ObservationGroupCfg, ObservationTermCfg",
    "from mjlab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg",
    "manager-term config imports",
  )

  tasks_init = project / "src/supermimic/tasks/__init__.py"
  addendum = (bundle_root / "overlay/src/supermimic/tasks/__init__.py.addendum").read_text()
  tasks_text = tasks_init.read_text()
  if "SuperMimic-GenCoord-Asimov-12DOF" not in tasks_text:
    tasks_init.write_text(tasks_text.rstrip() + "\n" + addendum)
    print("[patch] register SuperMimic-GenCoord-Asimov-12DOF")
  else:
    print("[skip] Asimov task already registered")

  pyproject = project / "pyproject.toml"
  replace_once(
    pyproject,
    'mjlab = { path = "../mjlab_research", editable = true }',
    f'mjlab = {{ path = "{args.asimov_mjlab.as_posix()}", editable = true }}',
    "MJLab dependency path",
  )

  print("\nOverlay applied to the separate integration copy.")
  print("Original SuperMimic, mjlab_research and asimov-mjlab checkouts were not modified.")
  print("Next: run `uv lock` and sync into a separate environment; see README_ASIMOV.md.")


if __name__ == "__main__":
  main()
