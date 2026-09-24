#!/usr/bin/env python3
"""Run the released end-to-end test pipeline with frozen custom checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_FINAL_SHA256 = "033b699bfcfbcb9da3a6ae9aa0188d3d31b2de1bc8e82fca8a422d20415d6630"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_stages(python: str, moss_python: str) -> list[tuple[str, list[str], bool]]:
    p = python
    return [
        (
            "speaker-count features",
            [p, "scripts/extract_speaker_count_features.py", ".", "--split", "test"],
            True,
        ),
        (
            "Qwen3-ASR transcription",
            [p, "scripts/run_qwen3_test_asr.py", ".", "--config", "configs/experiments/v4_qwen3_asr.yaml"],
            True,
        ),
        (
            "FireRedASR2-AED transcription",
            [p, "scripts/run_fireredasr2_test.py", ".", "--config", "configs/experiments/v28_fireredasr2_aed.yaml"],
            True,
        ),
        (
            "FireRedASR2-LLM transcription",
            [p, "scripts/run_fireredasr2_test.py", ".", "--config", "configs/experiments/v32_fireredasr2_llm.yaml"],
            True,
        ),
        (
            "MOSS joint transcription and diarization",
            [moss_python, "scripts/run_moss_test_inference.py", ".", "--config", "configs/experiments/v74_moss_transcribe_diarize.yaml"],
            True,
        ),
        (
            "offline Sortformer candidate",
            [p, "scripts/run_sortformer_test_candidate.py", ".", "--config", "configs/experiments/submission_v16.yaml"],
            True,
        ),
        (
            "adapted streaming Sortformer candidate",
            [p, "scripts/run_sortformer_test_candidate.py", ".", "--config", "configs/experiments/submission_v19.yaml"],
            True,
        ),
        (
            "label-free diarization router",
            [p, "scripts/run_v20_test_router.py", ".", "--config", "configs/experiments/submission_v20.yaml"],
            True,
        ),
        (
            "CAMPPlus multiscale test features",
            [p, "scripts/prepare_speaker_graph_test.py", ".", "--config", "configs/experiments/v25_multiscale_speaker_graph.yaml"],
            True,
        ),
        (
            "ERes2NetV2 multiscale test features",
            [p, "scripts/prepare_speaker_graph_test.py", ".", "--config", "configs/experiments/v134_eres2netv2_multiscale_metric_features.yaml"],
            True,
        ),
        (
            "WeSpeaker multiscale test features",
            [p, "scripts/extract_wespeaker_multiscale_features.py", ".", "--config", "configs/experiments/v171_wespeaker_resnet34_multiscale_features.yaml", "--scope", "test"],
            True,
        ),
        (
            "boundary-aware acoustic graph",
            [p, "scripts/run_metric_graph_test.py", ".", "--config", "configs/experiments/v93_boundary_metric_graph_full_test.yaml"],
            True,
        ),
        (
            "multi-ASR consensus candidate",
            [p, "scripts/build_v84_test_candidate.py", "."],
            True,
        ),
        (
            "acoustic-graph fusion candidate",
            [p, "scripts/build_v95_test_submission.py", "."],
            False,
        ),
        (
            "context-aware speaker Transformer",
            [p, "scripts/run_consensus_prototype_arbitration.py", ".", "--config", "configs/deployment/v537_voxconverse_context_v156_test.yaml", "--final-test"],
            True,
        ),
        (
            "novel-speaker verification",
            [p, "scripts/build_v174_test_submission.py", ".", "--config", "configs/deployment/v538_voxconverse_context_v174_test.yaml"],
            True,
        ),
        (
            "overlap consistency",
            [
                p,
                "scripts/postprocess_self_overlap_source_restore.py",
                "--refined",
                "submissions/v538_voxconverse_context_v174.seglst.json",
                "--source",
                "submissions/v95_moss_acoustic_graph_fusion.seglst.json",
                "--output",
                "submissions/final_solution.seglst.json",
                "--audit",
                "submissions/final_solution.audit.json",
                "--min-overlap",
                "0.20",
            ],
            False,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", nargs="?", type=Path, default=Path("."))
    parser.add_argument("--from-stage", type=int, default=1)
    parser.add_argument("--to-stage", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--moss-python",
        type=Path,
        help="Python from the isolated MOSS runtime (default: .venv-moss/bin/python).",
    )
    parser.add_argument(
        "--allow-retrained-checkpoints",
        action="store_true",
        help="Accept locally retrained custom weights instead of released hashes.",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    moss_python = args.moss_python
    if moss_python is None:
        configured = os.environ.get("MOSS_PYTHON")
        moss_python = Path(configured) if configured else root / ".venv-moss/bin/python"
    moss_python = moss_python.expanduser().resolve()
    if not moss_python.is_file() and not args.dry_run:
        raise FileNotFoundError(
            f"MOSS runtime not found: {moss_python}. Run scripts/install_runtime.sh first."
        )
    stages = command_stages(sys.executable, str(moss_python))
    upper = len(stages) if args.to_stage is None else args.to_stage
    if not 1 <= args.from_stage <= upper <= len(stages):
        raise ValueError(f"stage range must be within 1..{len(stages)}")

    if not args.skip_preflight:
        preflight = [
            sys.executable,
            "scripts/verify_release.py",
            ".",
            "--require-official-data",
            "--require-public-models",
        ]
        if args.allow_retrained_checkpoints:
            preflight.append("--allow-retrained-checkpoints")
        print("[preflight] " + " ".join(preflight), flush=True)
        if not args.dry_run:
            subprocess.run(preflight, cwd=root, check=True)

    for index, (name, command, accepts_overwrite) in enumerate(stages, start=1):
        if index < args.from_stage or index > upper:
            continue
        current = list(command)
        if args.overwrite and accepts_overwrite:
            current.append("--overwrite")
        print(f"[{index:02d}/{len(stages):02d}] {name}: {' '.join(current)}", flush=True)
        if not args.dry_run:
            subprocess.run(current, cwd=root, check=True)

    final_path = root / "submissions/final_solution.seglst.json"
    if upper == len(stages) and not args.dry_run:
        actual = sha256_file(final_path)
        payload = {
            "submission": str(final_path),
            "sha256": actual,
            "reference_sha256": EXPECTED_FINAL_SHA256,
            "byte_identical_to_reference": actual == EXPECTED_FINAL_SHA256,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if actual != EXPECTED_FINAL_SHA256:
            print(
                "warning: output is structurally reproducible but not byte-identical; "
                "check public-model hashes, CUDA determinism, and package versions",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
