#!/usr/bin/env python3
"""Download public ASR, diarization, and speaker checkpoints."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


HF_MODELS = [
    ("Qwen/Qwen3-ASR-1.7B", "models/modelscope/Qwen3-ASR-1.7B"),
    ("Qwen/Qwen3-ForcedAligner-0.6B", "models/modelscope/Qwen3-ForcedAligner-0.6B"),
    ("FireRedTeam/FireRedASR2-AED", "models/modelscope/FireRedASR2-AED"),
    ("FireRedTeam/FireRedASR2-LLM", "models/modelscope/FireRedASR2-LLM"),
    ("OpenMOSS-Team/MOSS-Transcribe-Diarize", "models/huggingface/OpenMOSS-Team/MOSS-Transcribe-Diarize"),
    ("nvidia/diar_sortformer_4spk-v1", "models/huggingface/nvidia/diar_sortformer_4spk-v1"),
    ("nvidia/diar_streaming_sortformer_4spk-v2.1", "models/huggingface/nvidia/diar_streaming_sortformer_4spk-v2.1"),
    ("pyannote/wespeaker-voxceleb-resnet34-LM", "models/huggingface/pyannote/wespeaker-voxceleb-resnet34-LM"),
]
MS_MODELS = [
    (
        "iic/speech_eres2netv2_sv_zh-cn_16k-common",
        "models/modelscope/models/iic--speech_eres2netv2_sv_zh-cn_16k-common/snapshots/master",
    ),
    (
        "iic/speech_campplus_sv_zh_en_16k-common_advanced",
        "models/modelscope/3dspeaker/models/iic--speech_campplus_sv_zh_en_16k-common_advanced/snapshots/v1.0.0",
    ),
    (
        "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "models/modelscope/3dspeaker/models/iic--speech_fsmn_vad_zh-cn-16k-common-pytorch/snapshots/v2.0.4",
    ),
]


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run(command: list[str], root: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def download_hf(repo: str, destination: Path, root: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if command_exists("hf"):
        command = ["hf", "download", repo, "--local-dir", str(destination)]
    elif command_exists("huggingface-cli"):
        command = [
            "huggingface-cli",
            "download",
            repo,
            "--local-dir",
            str(destination),
        ]
    else:
        raise RuntimeError(
            "Install huggingface_hub (`pip install huggingface_hub`) before downloading"
        )
    run(command, root)


def download_modelscope(repo: str, destination: Path, root: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not command_exists("modelscope"):
        raise RuntimeError("Install ModelScope (`pip install modelscope`) before downloading")
    run(
        ["modelscope", "download", "--model", repo, "--local_dir", str(destination)],
        root,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--only", choices=("all", "hf", "modelscope"), default="all")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
    if args.only in ("all", "hf"):
        for repo, relative in HF_MODELS:
            destination = root / relative
            if args.skip_existing and any(destination.glob("*")):
                print(f"skip existing {destination}")
                continue
            download_hf(repo, destination, root)
    if args.only in ("all", "modelscope"):
        for repo, relative in MS_MODELS:
            destination = root / relative
            if args.skip_existing and any(destination.glob("*")):
                print(f"skip existing {destination}")
                continue
            download_modelscope(repo, destination, root)

    print("Downloads complete. Compare exact files with configs/public_models.yaml.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
