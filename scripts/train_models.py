#!/usr/bin/env python3
"""Orchestrate training of the learned components used by the final system."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def stages(python: str) -> list[tuple[str, list[str], bool]]:
    p = python
    return [
        (
            "development speaker-count features",
            [p, "scripts/extract_speaker_count_features.py", ".", "--split", "dev"],
            True,
        ),
        (
            "speaker-count classifier",
            [p, "scripts/train_speaker_count.py", ".", "--fold", "full"],
            False,
        ),
        (
            "CAMPPlus labelled development features",
            [p, "scripts/prepare_speaker_graph_fold.py", ".", "--config", "configs/experiments/v25_multiscale_speaker_graph.yaml", "--fold", "0"],
            True,
        ),
        (
            "ERes2NetV2 labelled development features",
            [p, "scripts/prepare_speaker_graph_fold.py", ".", "--config", "configs/experiments/v134_eres2netv2_multiscale_metric_features.yaml", "--fold", "0"],
            True,
        ),
        (
            "fixed public VoxConverse ERes2NetV2 features",
            [p, "scripts/extract_campplus_multiscale_external.py", ".", "--config", "configs/experiments/v134_eres2netv2_multiscale_metric_features.yaml", "--audio-root", "data/external/voxconverse_fixed_48/wav", "--output-root", "data/external/voxconverse_fixed_48/features/eres2netv2"],
            True,
        ),
        (
            "fixed public VoxConverse CAMPPlus features",
            [p, "scripts/extract_campplus_multiscale_external.py", ".", "--config", "configs/experiments/v25_multiscale_speaker_graph.yaml", "--audio-root", "data/external/voxconverse_fixed_48/wav", "--output-root", "data/external/voxconverse_fixed_48/features/campplus"],
            True,
        ),
        (
            "external-data contextual initialization",
            [p, "scripts/train_data_specialized_v174_heads.py", ".", "--config", "configs/experiments/contextual_speaker_transformer_external_init.yaml", "--fold", "0"],
            True,
        ),
        (
            "CAMPPlus boundary metric",
            [p, "scripts/train_speaker_metric_full.py", ".", "--config", "configs/experiments/v92_boundary_metric_full.yaml"],
            True,
        ),
        (
            "ERes2NetV2 boundary metric",
            [p, "scripts/train_speaker_metric_full.py", ".", "--config", "configs/deployment/v156_eres2netv2_boundary_metric_full.yaml"],
            True,
        ),
        (
            "speaker-purity head",
            [p, "scripts/train_speaker_purity_full.py", ".", "--config", "configs/deployment/v156_speaker_purity_full.yaml"],
            True,
        ),
        (
            "context-aware speaker Transformer",
            [p, "scripts/train_contextual_speaker_metric_full.py", ".", "--config", "configs/deployment/v534_voxconverse_context_full.yaml"],
            True,
        ),
        (
            "deterministic Sortformer mixtures",
            [p, "scripts/build_sortformer_synthetic.py", ".", "--config", "configs/experiments/v14_sortformer_synthetic.yaml", "--fold", "full"],
            True,
        ),
        (
            "streaming Sortformer adaptation",
            [p, "scripts/train_sortformer_fold.py", ".", "--config", "configs/experiments/v14_sortformer_v2_1_synthetic_finetune.yaml", "--fold", "full"],
            False,
        ),
        (
            "WeSpeaker development features",
            [p, "scripts/extract_wespeaker_multiscale_features.py", ".", "--config", "configs/experiments/v171_wespeaker_resnet34_multiscale_features.yaml", "--scope", "dev"],
            True,
        ),
        (
            "two-fold OOF cache for novel-speaker supervision",
            [p, "scripts/build_oof_training_cache.py", "."],
            True,
        ),
        (
            "novel-speaker energy",
            [p, "scripts/train_novel_speaker_energy_full.py", ".", "--config", "configs/deployment/v174_wespeaker_novel_existence_energy_full.yaml"],
            True,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", nargs="?", type=Path, default=Path("."))
    parser.add_argument("--from-stage", type=int, default=1)
    parser.add_argument("--to-stage", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-novel-energy",
        action="store_true",
        help="Keep the supplied energy checkpoint instead of rebuilding it from OOF caches.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    pipeline = stages(sys.executable)
    upper = len(pipeline) if args.to_stage is None else args.to_stage
    if not 1 <= args.from_stage <= upper <= len(pipeline):
        raise ValueError(f"stage range must be within 1..{len(pipeline)}")

    for index, (name, command, accepts_overwrite) in enumerate(pipeline, start=1):
        if index < args.from_stage or index > upper:
            continue
        if args.skip_novel_energy and index == len(pipeline):
            print(f"[{index:02d}/{len(pipeline):02d}] novel-speaker energy: using supplied checkpoint")
            continue
        current = list(command)
        if args.overwrite and accepts_overwrite:
            current.append("--overwrite")
        print(f"[{index:02d}/{len(pipeline):02d}] {name}: {' '.join(current)}", flush=True)
        if not args.dry_run:
            subprocess.run(current, cwd=root, check=True)


if __name__ == "__main__":
    main()
