#!/usr/bin/env python3
"""Extract fold-audited multiscale CAMPPlus sequences and frame labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from run_sortformer_relabel import sha256_file
from run_word_embedding_diarization import centered_window


def interval_overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fold = int(config["fold"] if args.fold is None else args.fold)
    split_dir = root / "data" / "splits" / f"fold_{fold}"
    train_ids = (split_dir / "train_sessions.txt").read_text().split()
    val_ids = (split_dir / "val_sessions.txt").read_text().split()
    if set(train_ids) & set(val_ids):
        raise RuntimeError("Fold leakage guard failed")
    if args.max_sessions:
        train_ids = train_ids[: args.max_sessions]
        val_ids = val_ids[: args.max_sessions]

    reference = json.loads((root / "data" / "dev" / "ref.seglst.json").read_text())
    by_session: dict[str, list[dict]] = defaultdict(list)
    for row in reference:
        by_session[str(row["session_id"])].append(row)
    if (set(train_ids) | set(val_ids)) - set(by_session):
        raise RuntimeError("Reference coverage is incomplete")

    toolkit = root / "third_party" / "3D-Speaker"
    sys.path.insert(0, str(toolkit))
    os.environ.setdefault("MODELSCOPE_CACHE", str(root / "models" / "modelscope"))
    from speakerlab.bin.infer_diarization import Diarization3Dspeaker
    from speakerlab.utils.fileio import load_audio

    feature_config = config["features"]
    diarizer = Diarization3Dspeaker(
        device=feature_config["device"],
        include_overlap=False,
        model_cache_dir=str(root / "models" / "modelscope" / "3dspeaker"),
    )
    embedding_checkpoint_sha256 = None
    if feature_config.get("embedding_override"):
        override = feature_config["embedding_override"]
        if override["architecture"] != "eres2netv2":
            raise ValueError(f"Unsupported embedding override: {override['architecture']}")
        from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2

        checkpoint = (root / str(override["checkpoint"]).format(fold=fold)).resolve()
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
        diarizer.embedding_model = embedding_model.to(diarizer.device).eval()
        embedding_checkpoint_sha256 = sha256_file(checkpoint)
    hop = float(feature_config["hop_seconds"])
    scales = [float(value) for value in feature_config["window_seconds"]]
    output_root = root / "data" / "speaker_graph" / config["name"] / f"fold_{fold}"
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    subset_audit: dict[str, dict] = {}

    for subset, session_ids in (("train", train_ids), ("val", val_ids)):
        subset_dir = output_root / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        frame_total = 0
        for position, session_id in enumerate(session_ids, start=1):
            output_path = subset_dir / f"{session_id}.pt"
            if output_path.exists() and not args.overwrite:
                saved = torch.load(output_path, map_location="cpu", weights_only=True)
                frame_total += int(saved["features"].shape[0])
                continue
            wav_path = root / "data" / "dev" / "wav" / f"{session_id}.wav"
            wav = load_audio(str(wav_path), None, diarizer.fs)
            duration = wav.shape[-1] / diarizer.fs
            centers = np.arange(hop / 2, duration, hop, dtype=np.float32)
            all_windows = [
                centered_window(float(center), duration, scale)
                for scale in scales
                for center in centers
            ]
            embeddings = np.asarray(
                diarizer.do_emb_extraction(all_windows, wav), dtype=np.float32
            )
            expected = len(centers) * len(scales)
            if embeddings.shape != (expected, int(feature_config["embedding_dimension"])):
                raise RuntimeError(
                    f"Unexpected embedding shape for {session_id}: {embeddings.shape}"
                )
            multiscale = embeddings.reshape(len(scales), len(centers), -1)
            multiscale /= np.maximum(
                np.linalg.norm(multiscale, axis=-1, keepdims=True), 1e-6
            )
            features = np.concatenate(list(multiscale), axis=-1)

            rows = by_session[session_id]
            speakers = sorted(
                {str(row["speaker"]) for row in rows},
                key=lambda speaker: (
                    min(float(row["start_time"]) for row in rows if row["speaker"] == speaker),
                    speaker,
                ),
            )
            if not 2 <= len(speakers) <= int(config["model"]["max_speakers"]):
                raise RuntimeError(f"Speaker capacity mismatch in {session_id}")
            targets = np.zeros((len(centers), len(speakers)), dtype=np.float32)
            for frame, center in enumerate(centers.tolist()):
                cell = (center - hop / 2, center + hop / 2)
                for speaker_index, speaker in enumerate(speakers):
                    occupied = sum(
                        interval_overlap(
                            cell, (float(row["start_time"]), float(row["end_time"]))
                        )
                        for row in rows
                        if str(row["speaker"]) == speaker
                    )
                    targets[frame, speaker_index] = min(1.0, occupied / hop)
            payload = {
                "session_id": session_id,
                "subset": subset,
                "development_only": True,
                "uses_test_data": False,
                "wav_sha256": sha256_file(wav_path),
                "features": torch.from_numpy(features),
                "targets": torch.from_numpy(targets),
                "centers": torch.from_numpy(centers),
                "speaker_ids": speakers,
                "duration": float(duration),
            }
            torch.save(payload, output_path)
            frame_total += len(centers)
            print(
                json.dumps(
                    {
                        "subset": subset,
                        "session": session_id,
                        "position": position,
                        "total": len(session_ids),
                        "frames": len(centers),
                        "speakers": len(speakers),
                    }
                ),
                flush=True,
            )
        subset_audit[subset] = {"sessions": session_ids, "frames": frame_total}

    metadata = {
        "experiment": config["name"],
        "fold": fold,
        "development_only": True,
        "uses_training_labels": True,
        "uses_validation_labels_for_gradient": False,
        "uses_test_data": False,
        "train_validation_disjoint": not bool(set(train_ids) & set(val_ids)),
        "feature_dimension": len(scales) * int(feature_config["embedding_dimension"]),
        "embedding_architecture": feature_config.get("embedding_override", {}).get(
            "architecture", "campplus"
        ),
        "embedding_checkpoint_sha256": embedding_checkpoint_sha256,
        "subsets": subset_audit,
        "config_sha256": sha256_file(config_path),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
