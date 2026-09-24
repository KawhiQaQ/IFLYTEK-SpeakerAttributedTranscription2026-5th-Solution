#!/usr/bin/env python3
"""Extract label-free multiscale CAM++ embeddings for public training audio."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from run_sortformer_relabel import sha256_file
from run_word_embedding_diarization import centered_window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audio_root = root / args.audio_root
    session_ids = sorted(path.stem for path in audio_root.glob("*.wav"))
    if args.max_sessions:
        session_ids = session_ids[: args.max_sessions]
    if not session_ids:
        raise RuntimeError("No public training sessions selected")

    toolkit = root / "third_party/3D-Speaker"
    sys.path.insert(0, str(toolkit))
    os.environ.setdefault("MODELSCOPE_CACHE", str(root / "models/modelscope"))
    from speakerlab.bin.infer_diarization import Diarization3Dspeaker
    from speakerlab.utils.fileio import load_audio

    feature_config = config["features"]
    embedding_checkpoint_sha256 = None
    if feature_config.get("embedding_override"):
        override = feature_config["embedding_override"]
        if override["architecture"] != "eres2netv2":
            raise ValueError(
                f"Unsupported embedding override: {override['architecture']}"
            )
        from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2
        from speakerlab.process.processor import FBank

        checkpoint = (root / override["checkpoint"]).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        embedding_model = ERes2NetV2(
            feat_dim=80,
            embedding_size=int(feature_config["embedding_dimension"]),
            baseWidth=26,
            scale=2,
            expansion=2,
        )
        embedding_model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        diarizer = Diarization3Dspeaker.__new__(Diarization3Dspeaker)
        diarizer.device = torch.device(feature_config["device"])
        diarizer.feature_extractor = FBank(
            n_mels=80, sample_rate=16000, mean_nor=True
        )
        diarizer.embedding_model = embedding_model.to(diarizer.device).eval()
        diarizer.batchsize = 64
        diarizer.fs = diarizer.feature_extractor.sample_rate
        embedding_checkpoint_sha256 = sha256_file(checkpoint)
    else:
        diarizer = Diarization3Dspeaker(
            device=feature_config["device"],
            include_overlap=False,
            model_cache_dir=str(root / "models/modelscope/3dspeaker"),
        )
    hop = float(feature_config["hop_seconds"])
    scales = [float(value) for value in feature_config["window_seconds"]]
    dimension = int(feature_config["embedding_dimension"])
    output_root = root / args.output_root
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
        wav = load_audio(str(wav_path), None, diarizer.fs)
        duration = wav.shape[-1] / diarizer.fs
        centers = np.arange(hop / 2, duration, hop, dtype=np.float32)
        windows = [
            centered_window(float(center), duration, scale)
            for scale in scales
            for center in centers
        ]
        embeddings = np.asarray(diarizer.do_emb_extraction(windows, wav), dtype=np.float32)
        expected = len(centers) * len(scales)
        if embeddings.shape != (expected, dimension):
            raise RuntimeError(f"Unexpected CAM++ shape for {session_id}: {embeddings.shape}")
        values = embeddings.reshape(len(scales), len(centers), -1)
        values /= np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-6)
        features = np.concatenate(list(values), axis=-1)
        payload = {
            "session_id": session_id,
            "scope": "external_public_train",
            "development_only": True,
            "uses_speaker_labels": False,
            "uses_test_data": False,
            "inference_only": False,
            "wav_sha256": sha256_file(wav_path),
            "features": torch.from_numpy(features),
            "centers": torch.from_numpy(centers),
            "duration": float(duration),
            "window_seconds": scales,
        }
        torch.save(payload, output_path)
        frame_total += len(centers)
        print(json.dumps({"session": session_id, "position": position, "total": len(session_ids), "frames": len(centers)}), flush=True)
    metadata = {
        "experiment": config["name"],
        "scope": "external_public_train",
        "uses_speaker_labels": False,
        "uses_test_data": False,
        "sessions": session_ids,
        "frames": frame_total,
        "feature_dimension": dimension * len(scales),
        "embedding_architecture": feature_config.get("embedding_override", {}).get(
            "architecture", "campplus"
        ),
        "embedding_checkpoint_sha256": embedding_checkpoint_sha256,
        "config_sha256": sha256_file(config_path),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
