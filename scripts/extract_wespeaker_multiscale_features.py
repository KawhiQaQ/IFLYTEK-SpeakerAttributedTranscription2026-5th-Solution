#!/usr/bin/env python3
"""Extract label-free multiscale WeSpeaker ResNet34 embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from torch.nn import functional as F


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_window(
    waveform: torch.Tensor, center: float, duration: float, sample_rate: int
) -> torch.Tensor:
    samples = int(round(duration * sample_rate))
    midpoint = int(round(center * sample_rate))
    start = midpoint - samples // 2
    end = start + samples
    source_start = max(0, start)
    source_end = min(waveform.shape[-1], end)
    output = waveform.new_zeros(samples)
    destination_start = source_start - start
    output[destination_start : destination_start + source_end - source_start] = waveform[
        source_start:source_end
    ]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scope", choices=("cv", "dev", "test"), default="cv")
    parser.add_argument(
        "--audio-root",
        type=Path,
        help="Optional project-relative audio directory for public training data.",
    )
    parser.add_argument(
        "--output-root-override",
        type=Path,
        help="Optional project-relative output directory paired with --audio-root.",
    )
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if bool(args.audio_root) != bool(args.output_root_override):
        raise ValueError("--audio-root and --output-root-override must be provided together")
    if args.audio_root and args.scope != "dev":
        raise ValueError("Custom audio roots are development/training-only")
    if args.audio_root:
        audio_root = root / args.audio_root
        session_ids = sorted(path.stem for path in audio_root.glob("*.wav"))
    elif args.scope == "test":
        audio_root = root / "data/test/wav"
        session_ids = sorted(path.stem for path in audio_root.glob("*.wav"))
    elif args.scope == "dev":
        audio_root = root / "data/dev/wav"
        session_ids = sorted(path.stem for path in audio_root.glob("*.wav"))
    else:
        audio_root = root / "data/dev/wav"
        session_ids = sorted(
            set((root / "data/splits/fold_0/val_sessions.txt").read_text().split())
            | set((root / "data/splits/fold_1/val_sessions.txt").read_text().split())
        )
    if args.max_sessions:
        session_ids = session_ids[: args.max_sessions]
    if not session_ids:
        raise RuntimeError("No feature-extraction sessions selected")

    # DiariZen vendors the matching pyannote.audio implementation.  The
    # dependency namespace contains the rest of the pinned pyannote stack.
    dependency_root = root / "models/runtime/diarizen_pydeps"
    pyannote_audio_root = root / "third_party/DiariZen/pyannote-audio"
    for path in [dependency_root, pyannote_audio_root, root / "third_party/DiariZen"]:
        sys.path.insert(0, str(path))
    import pyannote

    pyannote.__path__.append(str(dependency_root / "pyannote"))
    from run_diarizen_relabel import install_runtime_compatibility

    install_runtime_compatibility()
    from pyannote.audio import Model

    checkpoint_path = root / config["model"]["checkpoint"]
    model = Model.from_pretrained(checkpoint_path, map_location="cpu")
    device = torch.device(config["features"]["device"])
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    hop = float(config["features"]["hop_seconds"])
    scales = [float(value) for value in config["features"]["window_seconds"]]
    batch_size = int(config["features"]["batch_size"])
    sample_rate = 16000
    output_root = (
        root / args.output_root_override
        if args.output_root_override
        else root / f"data/speaker_features_label_free/{config['name']}/{args.scope}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    frame_total = 0
    for position, session_id in enumerate(session_ids, start=1):
        output_path = output_root / f"{session_id}.pt"
        if output_path.exists() and not args.overwrite:
            saved = torch.load(output_path, map_location="cpu", weights_only=True)
            frame_total += len(saved["centers"])
            continue
        wav_path = audio_root / f"{session_id}.wav"
        values, loaded_sample_rate = sf.read(
            wav_path, dtype="float32", always_2d=True
        )
        if loaded_sample_rate != sample_rate or values.shape[1] != 1:
            raise RuntimeError(f"Unexpected audio format: {wav_path}")
        waveform = torch.from_numpy(values[:, 0].copy())
        duration = waveform.shape[-1] / sample_rate
        centers = torch.arange(hop / 2, duration, hop, dtype=torch.float32)
        scale_features = []
        for scale in scales:
            windows = torch.stack(
                [fixed_window(waveform, float(center), scale, sample_rate) for center in centers]
            )
            embeddings = []
            for start in range(0, len(windows), batch_size):
                batch = windows[start : start + batch_size].unsqueeze(1).to(device)
                with torch.inference_mode():
                    output = model(batch)
                embeddings.append(F.normalize(output.float(), dim=-1).cpu())
            scale_features.append(torch.cat(embeddings))
        features = torch.cat(scale_features, dim=-1)
        if features.shape != (len(centers), 256 * len(scales)):
            raise RuntimeError(
                f"Unexpected WeSpeaker feature shape for {session_id}: {features.shape}"
            )
        payload = {
            "session_id": session_id,
            "scope": args.scope,
            "development_only": args.scope != "test",
            "uses_speaker_labels": False,
            "uses_test_data": args.scope == "test",
            "inference_only": args.scope == "test",
            "wav_sha256": sha256_file(wav_path),
            "model_checkpoint_sha256": sha256_file(checkpoint_path),
            "features": features,
            "centers": centers,
            "duration": duration,
            "window_seconds": scales,
        }
        torch.save(payload, output_path)
        frame_total += len(centers)
        print(
            json.dumps(
                {
                    "scope": args.scope,
                    "session": session_id,
                    "position": position,
                    "total": len(session_ids),
                    "frames": len(centers),
                }
            ),
            flush=True,
        )
    metadata = {
        "experiment": config["name"],
        "scope": args.scope,
        "uses_speaker_labels": False,
        "uses_test_data": args.scope == "test",
        "inference_only": args.scope == "test",
        "sessions": session_ids,
        "frames": frame_total,
        "feature_dimension": 256 * len(scales),
        "model_checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
