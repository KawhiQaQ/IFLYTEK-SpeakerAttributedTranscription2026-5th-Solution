#!/usr/bin/env python3
"""Build the two fold-pure OOF sources required by the novel-speaker trainer."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def fold_commands(python: str, moss_python: str, fold: int) -> list[tuple[str, list[str], bool]]:
    p = python
    suffix = str(fold)
    return [
        ("CAMPPlus fold features", [p, "scripts/prepare_speaker_graph_fold.py", ".", "--config", "configs/experiments/v25_multiscale_speaker_graph.yaml", "--fold", suffix], True),
        ("ERes2NetV2 fold features", [p, "scripts/prepare_speaker_graph_fold.py", ".", "--config", "configs/experiments/v134_eres2netv2_multiscale_metric_features.yaml", "--fold", suffix], True),
        ("speaker-count classifier", [p, "scripts/train_speaker_count.py", ".", "--fold", suffix], False),
        ("Sortformer overlap mixtures", [p, "scripts/build_sortformer_synthetic.py", ".", "--config", "configs/experiments/v14_sortformer_synthetic.yaml", "--fold", suffix], True),
        ("Sortformer fold adaptation", [p, "scripts/train_sortformer_fold.py", ".", "--config", "configs/experiments/v14_sortformer_v2_1_synthetic_finetune.yaml", "--fold", suffix], False),
        ("Qwen3-ASR OOF inference", [p, "scripts/run_qwen3_asr.py", ".", "--config", "configs/experiments/v4_qwen3_asr.yaml", "--fold", suffix], True),
        ("FireRedASR2-AED OOF inference", [p, "scripts/run_fireredasr2_aed.py", ".", "--config", "configs/experiments/v28_fireredasr2_aed.yaml", "--fold", suffix], True),
        ("FireRedASR2-LLM OOF inference", [p, "scripts/run_fireredasr2_llm.py", ".", "--config", "configs/experiments/v32_fireredasr2_llm.yaml", "--fold", suffix], True),
        ("MOSS OOF inference", [moss_python, "scripts/run_moss_transcribe_diarize.py", ".", "--config", "configs/experiments/v74_moss_transcribe_diarize.yaml", "--fold", suffix], True),
        ("3D-Speaker capacity fallback", [p, "scripts/run_3dspeaker_relabel.py", ".", "--config", "configs/experiments/v4_qwen3_3dspeaker_count.yaml", "--fold", suffix], True),
        ("offline Sortformer partition", [p, "scripts/run_sortformer_relabel.py", ".", "--config", "configs/experiments/v16_qwen3_sortformer_offline_public.yaml", "--fold", suffix], True),
        ("streaming Sortformer partition", [p, "scripts/run_sortformer_turn_clustering.py", ".", "--config", "configs/experiments/v19_qwen3_sortformer_consistent_turns.yaml", "--fold", suffix], True),
        ("diarization quality router", [p, "scripts/run_diarization_quality_router.py", ".", "--config", "configs/experiments/v20_diarization_quality_router.yaml", "--fold", suffix], True),
        ("AED text on routed tracks", [p, "scripts/run_cached_asr_on_speaker_tracks.py", ".", "--config", "configs/experiments/v29_fireredasr2_v20_tracks.yaml", "--fold", suffix], True),
        ("boundary metric", [p, "scripts/train_speaker_metric_fold.py", ".", "--config", "configs/experiments/v36_boundary_robust_metric_adapter.yaml", "--fold", suffix], True),
        ("boundary metric graph", [p, "scripts/run_metric_graph_relabel.py", ".", "--config", "configs/experiments/v37_boundary_metric_graph_firered.yaml", "--fold", suffix], True),
        ("novel acoustic partition", [p, "scripts/run_segment_novel_speaker_fusion.py", ".", "--config", "configs/experiments/v38_novel_speaker_hybrid.yaml", "--fold", suffix], True),
        ("MOSS substitution consensus", [p, "scripts/run_segment_text_consensus.py", ".", "--config", "configs/experiments/v82_moss_native_substitution_consensus.yaml", "--fold", suffix], True),
        ("real OOF novel-speaker episodes", [p, "scripts/run_segment_novel_speaker_fusion.py", ".", "--config", "configs/experiments/v90_moss_v38_novel_fusion.yaml", "--fold", suffix], True),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", nargs="?", type=Path, default=Path("."))
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--moss-python",
        type=Path,
        help="Python from the isolated MOSS runtime (default: .venv-moss/bin/python).",
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
    for fold in args.folds:
        if fold not in (0, 1):
            raise ValueError("the released novel-speaker model uses OOF folds 0 and 1")
        commands = fold_commands(sys.executable, str(moss_python), fold)
        for index, (name, command, accepts_overwrite) in enumerate(commands, start=1):
            current = list(command)
            if args.overwrite and accepts_overwrite:
                current.append("--overwrite")
            print(f"[fold {fold} {index:02d}/{len(commands):02d}] {name}: {' '.join(current)}", flush=True)
            if not args.dry_run:
                subprocess.run(current, cwd=root, check=True)


if __name__ == "__main__":
    main()
