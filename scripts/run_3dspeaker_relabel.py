#!/usr/bin/env python3
"""Relabel cached Paraformer word timestamps with the open 3D-Speaker pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml


SPECIAL_TAG = re.compile(r"<\|[^|]+\|>")
TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+(?:'[a-z0-9]+)?|[\u3400-\u4dbf\u4e00-\u9fff]",
    flags=re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", SPECIAL_TAG.sub(" ", text)).lower()
    return TOKEN_PATTERN.findall(text)


def overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def nearest_distance(interval: tuple[float, float], candidate: tuple[float, float]) -> float:
    midpoint = (interval[0] + interval[1]) / 2
    if candidate[0] <= midpoint <= candidate[1]:
        return 0.0
    return min(abs(midpoint - candidate[0]), abs(midpoint - candidate[1]))


def assign_speaker(
    token_interval: tuple[float, float], diarization: list[list[float]]
) -> int:
    best = max(
        diarization,
        key=lambda segment: (
            overlap(token_interval, (float(segment[0]), float(segment[1]))),
            -nearest_distance(token_interval, (float(segment[0]), float(segment[1]))),
        ),
    )
    return int(best[2])


def relabel(
    raw_result: dict,
    diarization: list[list[float]],
    session_id: str,
    max_gap_seconds: float,
) -> list[dict]:
    if not diarization:
        raise RuntimeError(f"No diarization segments for {session_id}")
    tokens = normalize_tokens(str(raw_result["text"]))
    timestamps = raw_result.get("timestamp")
    if not isinstance(timestamps, list) or len(tokens) != len(timestamps):
        raise RuntimeError(
            f"Token/timestamp mismatch for {session_id}: {len(tokens)} != "
            f"{len(timestamps) if isinstance(timestamps, list) else 'missing'}"
        )

    token_rows = []
    for token, timestamp in zip(tokens, timestamps):
        start = float(timestamp[0]) / 1000.0
        end = float(timestamp[1]) / 1000.0
        if end <= start:
            continue
        token_rows.append(
            {
                "token": token,
                "start": start,
                "end": end,
                "raw_speaker": assign_speaker((start, end), diarization),
            }
        )
    if not token_rows:
        raise RuntimeError(f"No timestamped tokens for {session_id}")

    speaker_map: dict[int, str] = {}
    grouped: list[dict] = []
    for row in token_rows:
        raw_speaker = row["raw_speaker"]
        if raw_speaker not in speaker_map:
            speaker_map[raw_speaker] = f"spk{len(speaker_map) + 1}"
        speaker = speaker_map[raw_speaker]
        new_segment = (
            not grouped
            or grouped[-1]["speaker"] != speaker
            or row["start"] - grouped[-1]["end_time"] > max_gap_seconds
        )
        if new_segment:
            grouped.append(
                {
                    "session_id": session_id,
                    "speaker": speaker,
                    "start_time": row["start"],
                    "end_time": row["end"],
                    "tokens": [row["token"]],
                }
            )
        else:
            grouped[-1]["end_time"] = row["end"]
            grouped[-1]["tokens"].append(row["token"])

    return [
        {
            "session_id": row["session_id"],
            "speaker": row["speaker"],
            "start_time": round(row["start_time"], 2),
            "end_time": round(row["end_time"], 2),
            "words": " ".join(row["tokens"]),
        }
        for row in grouped
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--diagnostic-oracle-speaker-count",
        action="store_true",
        help="Development-only diagnostic; uses reference counts and is never deployable.",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fold = int(config["fold"] if args.fold is None else args.fold)
    dev_dir = (root / "data" / "dev").resolve()
    if dev_dir.name != "dev":
        raise RuntimeError("Development-only guard failed")
    split_dir = root / "data" / "splits" / f"fold_{fold}"
    session_ids = (split_dir / "val_sessions.txt").read_text(encoding="utf-8").split()
    if args.max_sessions is not None:
        session_ids = session_ids[: args.max_sessions]
    wav_paths = [dev_dir / "wav" / f"{session_id}.wav" for session_id in session_ids]
    if not session_ids or not all(
        path.is_file() and path.parent == dev_dir / "wav" for path in wav_paths
    ):
        raise RuntimeError("Validation input guard failed")

    oracle_speaker_counts = None
    if args.diagnostic_oracle_speaker_count:
        reference_rows = json.loads(
            (split_dir / "val_ref.seglst.json").read_text(encoding="utf-8")
        )
        speaker_sets: dict[str, set[str]] = {}
        for row in reference_rows:
            speaker_sets.setdefault(row["session_id"], set()).add(row["speaker"])
        oracle_speaker_counts = {
            session_id: len(speakers) for session_id, speakers in speaker_sets.items()
        }
        if set(session_ids) - set(oracle_speaker_counts):
            raise RuntimeError("Oracle diagnostic speaker counts are incomplete")

    source_dir = (
        root
        / "outputs"
        / config["asr_source_experiment"]
        / f"fold_{fold}"
        / "sessions"
    )
    source_paths = [source_dir / f"{session_id}.json" for session_id in session_ids]
    if not all(path.is_file() for path in source_paths):
        raise FileNotFoundError("Cached development ASR result is incomplete")

    predicted_speaker_counts = None
    count_model_path = None
    if config.get("speaker_count_model"):
        count_model_path = root / str(config["speaker_count_model"]).format(fold=fold)
        feature_path = root / str(config["speaker_count_features"])
        artifact = joblib.load(count_model_path)
        if int(artifact["fold"]) != fold:
            raise RuntimeError("Speaker-count model fold mismatch")
        if set(artifact["train_sessions"]) & set(session_ids):
            raise RuntimeError("Speaker-count model leakage into validation sessions")
        feature_payload = json.loads(feature_path.read_text(encoding="utf-8"))
        if feature_payload.get("uses_speaker_labels") is not False:
            raise RuntimeError("Speaker-count input features are not label-free")
        feature_names = artifact["feature_names"]
        feature_rows = feature_payload["sessions"]
        matrix = np.asarray(
            [
                [float(feature_rows[session_id][name]) for name in feature_names]
                for session_id in session_ids
            ],
            dtype=np.float64,
        )
        predicted = artifact["model"].predict(matrix).astype(int)
        predicted_speaker_counts = dict(zip(session_ids, predicted.tolist()))

    toolkit_dir = root / "third_party" / "3D-Speaker"
    if not toolkit_dir.is_dir():
        raise FileNotFoundError(f"Missing 3D-Speaker toolkit: {toolkit_dir}")
    sys.path.insert(0, str(toolkit_dir))
    os.environ.setdefault("MODELSCOPE_CACHE", str(root / "models" / "modelscope"))
    from speakerlab.bin.infer_diarization import Diarization3Dspeaker

    diarizer = Diarization3Dspeaker(
        device=config["diarization"]["device"],
        include_overlap=bool(config["diarization"]["include_overlap"]),
        model_cache_dir=str(root / "models" / "modelscope" / "3dspeaker"),
    )
    embedding_override_sha256 = None
    if config["diarization"].get("embedding_override"):
        override = config["diarization"]["embedding_override"]
        if override["architecture"] != "eres2netv2":
            raise ValueError(f"Unsupported embedding override: {override['architecture']}")
        from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2

        checkpoint = (root / override["checkpoint"]).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing ERes2NetV2 checkpoint: {checkpoint}")
        embedding_model = ERes2NetV2(
            feat_dim=80, embedding_size=192, baseWidth=26, scale=2, expansion=2
        )
        embedding_model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        diarizer.embedding_model = embedding_model.to(diarizer.device).eval()
        embedding_override_sha256 = sha256_file(checkpoint)

    suffix = "_smoke" if args.max_sessions is not None else ""
    if args.diagnostic_oracle_speaker_count:
        suffix += "_diagnostic_oracle_speaker_count"
    output_dir = root / "outputs" / config["name"] / f"fold_{fold}{suffix}"
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    all_segments = []
    for position, (session_id, wav_path, source_path) in enumerate(
        zip(session_ids, wav_paths, source_paths), start=1
    ):
        session_path = session_dir / f"{session_id}.json"
        if session_path.exists() and not args.overwrite:
            saved = json.loads(session_path.read_text(encoding="utf-8"))
            all_segments.extend(saved["segments"])
            print(json.dumps({"session": session_id, "status": "resumed"}), flush=True)
            continue
        session_started = time.time()
        source = json.loads(source_path.read_text(encoding="utf-8"))
        raw_result = source["raw_result"][0]
        diarization = diarizer(
            str(wav_path),
            speaker_num=(
                oracle_speaker_counts[session_id]
                if oracle_speaker_counts is not None
                else (
                    predicted_speaker_counts[session_id]
                    if predicted_speaker_counts is not None
                    else None
                )
            ),
        )
        diarization = [
            [float(segment[0]), float(segment[1]), int(segment[2])]
            for segment in diarization
        ]
        segments = relabel(
            raw_result,
            diarization,
            session_id,
            float(config["assignment"]["max_gap_seconds"]),
        )
        payload = {
            "session_id": session_id,
            "wav_sha256": sha256_file(wav_path),
            "asr_source_sha256": sha256_file(source_path),
            "diarization": diarization,
            "segments": segments,
            "predicted_speaker_count": (
                predicted_speaker_counts[session_id]
                if predicted_speaker_counts is not None
                else None
            ),
        }
        session_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        all_segments.extend(segments)
        print(
            json.dumps(
                {
                    "session": session_id,
                    "position": position,
                    "total": len(session_ids),
                    "segments": len(segments),
                    "speakers": len({row["speaker"] for row in segments}),
                    "elapsed_seconds": round(time.time() - session_started, 2),
                }
            ),
            flush=True,
        )

    prediction_path = output_dir / "hyp.seglst.json"
    prediction_path.write_text(
        json.dumps(all_segments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "experiment": config["name"],
        "fold": fold,
        "smoke_only": args.max_sessions is not None,
        "development_only": True,
        "diagnostic_oracle_speaker_count": args.diagnostic_oracle_speaker_count,
        "non_deployable": args.diagnostic_oracle_speaker_count,
        "uses_validation_labels": args.diagnostic_oracle_speaker_count,
        "speaker_count_model": str(count_model_path) if count_model_path else None,
        "embedding_override_sha256": embedding_override_sha256,
        "predicted_speaker_counts": predicted_speaker_counts,
        "session_ids": session_ids,
        "config": config,
        "config_sha256": sha256_file(config_path),
        "cv_manifest_sha256": sha256_file(root / "configs" / "cv" / "folds_v1.csv"),
        "prediction_sha256": sha256_file(prediction_path),
        "elapsed_seconds": round(time.time() - started, 2),
        "argv": sys.argv,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
