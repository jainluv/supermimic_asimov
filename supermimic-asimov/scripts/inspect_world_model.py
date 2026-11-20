#!/usr/bin/env python3
"""Inspect a saved SuperMimic world_model.pt parameter bundle.

This verifies the checkpoint structure and tensor finiteness. It does not
measure one-step or multi-step predictive accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions predating weights_only.
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    bundle = load_checkpoint(args.checkpoint)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected a dictionary checkpoint; got {type(bundle).__name__}")

    missing = {"dynamics", "sim_context"}.difference(bundle)
    if missing:
        raise ValueError(
            f"Not a SuperMimic world_model.pt bundle; missing keys: {sorted(missing)}. "
            f"Available keys: {sorted(bundle)}"
        )

    total_tensors = 0
    total_parameters = 0
    for group_name in ("dynamics", "sim_context"):
        state = bundle[group_name]
        if not isinstance(state, dict) or not state:
            raise ValueError(f"Checkpoint group '{group_name}' is empty or not a state dict")
        group_tensors = 0
        group_parameters = 0
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            group_tensors += 1
            group_parameters += value.numel()
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite tensor in {group_name}.{key}")
        if group_tensors == 0:
            raise ValueError(f"No tensors found in '{group_name}'")
        total_tensors += group_tensors
        total_parameters += group_parameters
        print(f"{group_name}: {group_tensors} tensors, {group_parameters:,} values")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"TOTAL: {total_tensors} tensors, {total_parameters:,} values")
    print("Checkpoint structure and tensor finiteness: PASS")
    print("Predictive accuracy: NOT TESTED (requires held-out MJLab transitions).")


if __name__ == "__main__":
    main()
