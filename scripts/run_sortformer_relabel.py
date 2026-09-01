#!/usr/bin/env python3
"""Relabel cached Qwen word timestamps with overlap-aware NVIDIA Sortformer.

Sortformer supports at most four speakers. Sessions whose fold-pure, label-free
speaker-count prediction exceeds that capacity use the already frozen V4 result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import sys
import time
import unicodedata
import wave
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
TRAILING_INTEGER = re.compile(r"(\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def normalize_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", SPECIAL_TAG.sub(" ", text)).lower()
    return TOKEN_PATTERN.findall(text)


def interval_overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def speaker_index(value: object) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    match = TRAILING_INTEGER.search(str(value))
    if match is None:
        raise ValueError(f"Cannot parse Sortformer speaker index: {value!r}")
    return int(match.group(1))


def parse_diarization(raw_segments: list[object]) -> list[list[float]]:
    parsed: list[list[float]] = []
    for item in raw_segments:
        fields = item.split() if isinstance(item, str) else list(item)
        if len(fields) != 3:
            raise ValueError(f"Unexpected Sortformer segment: {item!r}")
        parsed.append([float(fields[0]), float(fields[1]), speaker_index(fields[2])])
    return parsed


def load_fold_pure_counts(
    root: Path, config: dict, fold: int, session_ids: list[str]
) -> tuple[dict[str, int], Path]:
    model_path = root / str(config["speaker_count_model"]).format(fold=fold)
    artifact = joblib.load(model_path)
    if int(artifact["fold"]) != fold:
        raise RuntimeError("Speaker-count model fold mismatch")
    if set(artifact["train_sessions"]) & set(session_ids):
        raise RuntimeError("Speaker-count model leakage into validation sessions")

    feature_payload = json.loads(
        (root / config["speaker_count_features"]).read_text(encoding="utf-8")
    )
    if feature_payload.get("uses_speaker_labels") is not False:
        raise RuntimeError("Speaker-count routing features are not label-free")
    matrix = np.asarray(
        [
            [
                float(feature_payload["sessions"][session_id][name])
                for name in artifact["feature_names"]
            ]
            for session_id in session_ids
        ],
        dtype=np.float64,
    )
    predictions = artifact["model"].predict(matrix).astype(int)
    return dict(zip(session_ids, predictions.tolist())), model_path


def load_reference_counts(
    split_dir: Path, session_ids: list[str]
) -> dict[str, int]:
    """Development-only speaker-count ceiling diagnostic."""
    reference_rows = json.loads(
        (split_dir / "val_ref.seglst.json").read_text(encoding="utf-8")
    )
    speakers: dict[str, set[str]] = {session_id: set() for session_id in session_ids}
    for row in reference_rows:
        if row["session_id"] in speakers:
            speakers[row["session_id"]].add(str(row["speaker"]))
    if any(not speakers[session_id] for session_id in session_ids):
        raise RuntimeError("Oracle diagnostic speaker counts are incomplete")
    return {session_id: len(values) for session_id, values in speakers.items()}


def token_rows(raw_result: dict, session_id: str) -> list[dict]:
    tokens = normalize_tokens(str(raw_result["text"]))
    timestamps = raw_result.get("timestamp")
    if not isinstance(timestamps, list) or len(tokens) != len(timestamps):
        raise RuntimeError(
            f"Token/timestamp mismatch for {session_id}: {len(tokens)} != "
            f"{len(timestamps) if isinstance(timestamps, list) else 'missing'}"
        )
    rows = []
    for token, timestamp in zip(tokens, timestamps):
        start = float(timestamp[0]) / 1000.0
        end = float(timestamp[1]) / 1000.0
        if end > start:
            rows.append({"token": token, "start": start, "end": end})
    if not rows:
        raise RuntimeError(f"No timestamped tokens for {session_id}")
    return rows


def assign_speakers(
    rows: list[dict],
    diarization: list[list[float]],
    probabilities: torch.Tensor,
    duration: float,
    expected_speakers: int,
    use_predicted_arrival_slots: bool,
) -> None:
    if not diarization:
        raise RuntimeError("Sortformer returned no speech segments")
    probs = probabilities.detach().float().cpu().squeeze().numpy()
    if probs.ndim != 2:
        raise RuntimeError(f"Unexpected Sortformer probability shape: {probs.shape}")
    frame_count, speaker_slots = probs.shape
    if frame_count <= 0 or speaker_slots <= 0:
        raise RuntimeError(f"Empty Sortformer probability tensor: {probs.shape}")
    if use_predicted_arrival_slots:
        # Streaming Sortformer slots follow speaker arrival order. The fold-pure
        # count prediction determines how many of those slots are admissible;
        # this avoids silently collapsing a predicted 3/4-speaker session to
        # two speakers because of default segment postprocessing.
        allowed = list(range(min(expected_speakers, speaker_slots)))
    else:
        allowed = sorted({int(segment[2]) for segment in diarization})
    if not allowed or max(allowed) >= speaker_slots:
        raise RuntimeError("Sortformer segments/probabilities disagree on speaker slots")
    frame_seconds = duration / frame_count

    for row in rows:
        token_interval = (float(row["start"]), float(row["end"]))
        active = sorted(
            {
                int(segment[2])
                for segment in diarization
                if interval_overlap(
                    token_interval, (float(segment[0]), float(segment[1]))
                )
                > 0
            }
        )
        candidates = allowed if use_predicted_arrival_slots else (active or allowed)
        first = max(0, min(frame_count - 1, math.floor(row["start"] / frame_seconds)))
        last = max(first + 1, min(frame_count, math.ceil(row["end"] / frame_seconds)))
        scores = probs[first:last].mean(axis=0)
        row["raw_speaker"] = max(candidates, key=lambda index: (scores[index], -index))


def group_rows(
    rows: list[dict],
    session_id: str,
    max_gap: float,
    emit_token_segments: bool = False,
) -> list[dict]:
    speaker_map: dict[int, str] = {}
    grouped: list[dict] = []
    for row in rows:
        raw_speaker = int(row["raw_speaker"])
        if raw_speaker not in speaker_map:
            speaker_map[raw_speaker] = f"spk{len(speaker_map) + 1}"
        speaker = speaker_map[raw_speaker]
        if (
            not grouped
            or emit_token_segments
            or grouped[-1]["speaker"] != speaker
            or row["start"] - grouped[-1]["end_time"] > max_gap
        ):
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

    predicted_counts, count_model_path = load_fold_pure_counts(
        root, config, fold, session_ids
    )
    if args.diagnostic_oracle_speaker_count:
        predicted_counts = load_reference_counts(split_dir, session_ids)
    asr_dir = root / "outputs" / config["asr_source_experiment"] / f"fold_{fold}" / "sessions"
    fallback_fold = f"fold_{fold}"
    if args.diagnostic_oracle_speaker_count:
        fallback_fold += "_diagnostic_oracle_speaker_count"
    fallback_dir = root / "outputs" / config["fallback_experiment"] / fallback_fold / "sessions"
    if not all((asr_dir / f"{sid}.json").is_file() for sid in session_ids):
        raise FileNotFoundError("Cached Qwen development ASR result is incomplete")
    if not all((fallback_dir / f"{sid}.json").is_file() for sid in session_ids):
        raise FileNotFoundError("Frozen V4 fallback is incomplete")

    model_config = config["sortformer"]
    checkpoint = (
        root / str(model_config["checkpoint"]).format(fold=fold)
    ).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    from nemo.collections.asr.models import SortformerEncLabelModel

    device = torch.device(model_config["device"])
    model = SortformerEncLabelModel.restore_from(str(checkpoint), map_location=device)
    model = model.to(device).eval()

    suffix = "_smoke" if args.max_sessions is not None else ""
    if args.diagnostic_oracle_speaker_count:
        suffix += "_diagnostic_oracle_speaker_count"
    output_dir = root / "outputs" / config["name"] / f"fold_{fold}{suffix}"
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    max_speakers = int(model_config["max_speakers"])
    max_gap = float(config["assignment"]["max_gap_seconds"])
    emit_token_segments = bool(config["assignment"].get("emit_token_segments", False))
    started = time.time()
    all_segments: list[dict] = []

    for position, (session_id, wav_path) in enumerate(zip(session_ids, wav_paths), start=1):
        session_path = session_dir / f"{session_id}.json"
        if session_path.exists() and not args.overwrite:
            saved = json.loads(session_path.read_text(encoding="utf-8"))
            all_segments.extend(saved["segments"])
            print(json.dumps({"session": session_id, "status": "resumed"}), flush=True)
            continue
        session_started = time.time()
        asr_path = asr_dir / f"{session_id}.json"
        fallback_path = fallback_dir / f"{session_id}.json"
        used_fallback = predicted_counts[session_id] > max_speakers
        diarization: list[list[float]] | None = None
        probability_shape: list[int] | None = None
        if used_fallback:
            segments = json.loads(fallback_path.read_text(encoding="utf-8"))["segments"]
        else:
            source = json.loads(asr_path.read_text(encoding="utf-8"))
            rows = token_rows(source["raw_result"][0], session_id)
            raw_segments, raw_probabilities = model.diarize(
                audio=[str(wav_path)],
                batch_size=int(model_config["batch_size"]),
                include_tensor_outputs=True,
                num_workers=0,
                verbose=False,
            )
            if len(raw_segments) != 1 or len(raw_probabilities) != 1:
                raise RuntimeError("Unexpected Sortformer batch output")
            diarization = parse_diarization(raw_segments[0])
            probabilities = raw_probabilities[0]
            probability_shape = list(probabilities.shape)
            assign_speakers(
                rows,
                diarization,
                probabilities,
                duration_seconds(wav_path),
                predicted_counts[session_id],
                bool(config["assignment"].get("use_predicted_arrival_slots", False)),
            )
            segments = group_rows(
                rows,
                session_id,
                max_gap,
                emit_token_segments=emit_token_segments,
            )

        payload = {
            "session_id": session_id,
            "wav_sha256": sha256_file(wav_path),
            "asr_source_sha256": sha256_file(asr_path),
            "predicted_speaker_count": predicted_counts[session_id],
            "used_capacity_fallback": used_fallback,
            "fallback_source_sha256": sha256_file(fallback_path) if used_fallback else None,
            "diarization": diarization,
            "probability_shape": probability_shape,
            "segments": segments,
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
                    "predicted_speakers": predicted_counts[session_id],
                    "fallback": used_fallback,
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
        "uses_validation_labels": args.diagnostic_oracle_speaker_count,
        "diagnostic_oracle_speaker_count": args.diagnostic_oracle_speaker_count,
        "non_deployable": args.diagnostic_oracle_speaker_count,
        "session_ids": session_ids,
        "predicted_speaker_counts": predicted_counts,
        "capacity_fallback_sessions": [
            sid for sid in session_ids if predicted_counts[sid] > max_speakers
        ],
        "count_model": str(count_model_path),
        "session_list_sha256": sha256_file(split_dir / "val_sessions.txt"),
        "cv_manifest_sha256": sha256_file(root / "configs" / "cv" / "folds_v1.csv"),
        "config": config,
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("nemo-toolkit", "torch", "transformers")
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "prediction_sha256": sha256_file(prediction_path),
        "argv": sys.argv,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
