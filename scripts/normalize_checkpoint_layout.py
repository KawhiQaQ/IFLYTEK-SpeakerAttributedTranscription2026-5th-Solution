#!/usr/bin/env python3
"""Normalize the legacy Baidu checkpoint archive to the release layout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.project_root.resolve()
    current = root / "models/data_experts/contextual_speaker_transformer"
    candidates = sorted(
        path.parent.parent
        for path in (root / "models/data_experts").glob("*/fold_0/contextual_metric.pt")
        if path.parent.parent != current
    )
    legacy = candidates[0] if len(candidates) == 1 else None
    for subset in ("fold_0", "full"):
        destination = current / subset / "contextual_metric.pt"
        if destination.is_file():
            continue
        if legacy is None:
            raise FileNotFoundError(
                "could not identify one legacy contextual-checkpoint directory"
            )
        source = legacy / subset / "contextual_metric.pt"
        if not source.is_file():
            raise FileNotFoundError(f"missing checkpoint: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"copied {source.relative_to(root)} -> {destination.relative_to(root)}")


if __name__ == "__main__":
    main()
