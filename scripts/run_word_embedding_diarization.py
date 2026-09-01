#!/usr/bin/env python3
"""Cluster one CAMPPlus speaker embedding per Qwen timestamped token."""

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


def load_fold_pure_counts(
    root: Path, config: dict, fold: int, session_ids: list[str]
) -> tuple[dict[str, int], Path]:
    model_path = root / str(config["speaker_count_model"]).format(fold=fold)
    artifact = joblib.load(model_path)
    if int(artifact["fold"]) != fold:
        raise RuntimeError("Speaker-count model fold mismatch")
    if set(artifact["train_sessions"]) & set(session_ids):
        raise RuntimeError("Speaker-count model leakage into validation sessions")
    features = json.loads(
        (root / config["speaker_count_features"]).read_text(encoding="utf-8")
    )
    if features.get("uses_speaker_labels") is not False:
        raise RuntimeError("Speaker-count input features are not label-free")
    matrix = np.asarray(
        [
            [
                float(features["sessions"][session_id][name])
                for name in artifact["feature_names"]
            ]
            for session_id in session_ids
        ],
        dtype=np.float64,
    )
    predictions = artifact["model"].predict(matrix).astype(int)
    return dict(zip(session_ids, predictions.tolist())), model_path


def timestamped_tokens(raw_result: dict, session_id: str) -> list[dict]:
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


def centered_window(center: float, duration: float, width: float) -> list[float]:
    if duration <= width:
        return [0.0, duration]
    start = max(0.0, min(duration - width, center - width / 2))
    return [start, start + width]


def group_tokens(rows: list[dict], session_id: str, max_gap: float) -> list[dict]:
    speaker_map: dict[int, str] = {}
    grouped: list[dict] = []
    for row in rows:
        cluster = int(row["cluster"])
        if cluster not in speaker_map:
            speaker_map[cluster] = f"spk{len(speaker_map) + 1}"
        speaker = speaker_map[cluster]
        if (
            not grouped
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
    source_dir = (
        root / "outputs" / config["asr_source_experiment"] / f"fold_{fold}" / "sessions"
    )
    if not all((source_dir / f"{sid}.json").is_file() for sid in session_ids):
        raise FileNotFoundError("Cached Qwen development ASR result is incomplete")

    toolkit_dir = root / "third_party" / "3D-Speaker"
    if not toolkit_dir.is_dir():
        raise FileNotFoundError(f"Missing 3D-Speaker toolkit: {toolkit_dir}")
    sys.path.insert(0, str(toolkit_dir))
    os.environ.setdefault("MODELSCOPE_CACHE", str(root / "models" / "modelscope"))
    from speakerlab.bin.infer_diarization import Diarization3Dspeaker
    from speakerlab.utils.fileio import load_audio

    diarizer = Diarization3Dspeaker(
        device=config["embedding"]["device"],
        include_overlap=False,
        model_cache_dir=str(root / "models" / "modelscope" / "3dspeaker"),
    )
    window_seconds = float(config["embedding"]["window_seconds"])
    max_gap = float(config["assignment"]["max_gap_seconds"])

    suffix = "_smoke" if args.max_sessions is not None else ""
    output_dir = root / "outputs" / config["name"] / f"fold_{fold}{suffix}"
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    all_segments: list[dict] = []
    started = time.time()
    for position, (session_id, wav_path) in enumerate(zip(session_ids, wav_paths), start=1):
        session_path = session_dir / f"{session_id}.json"
        if session_path.exists() and not args.overwrite:
            saved = json.loads(session_path.read_text(encoding="utf-8"))
            all_segments.extend(saved["segments"])
            print(json.dumps({"session": session_id, "status": "resumed"}), flush=True)
            continue
        session_started = time.time()
        source_path = source_dir / f"{session_id}.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        rows = timestamped_tokens(source["raw_result"][0], session_id)
        wav_data = load_audio(str(wav_path), None, diarizer.fs)
        duration = wav_data.shape[-1] / diarizer.fs
        windows = [
            centered_window((row["start"] + row["end"]) / 2, duration, window_seconds)
            for row in rows
        ]
        embeddings = diarizer.do_emb_extraction(windows, wav_data)
        labels = diarizer.cluster(
            embeddings, speaker_num=predicted_counts[session_id]
        ).astype(int)
        if len(labels) != len(rows):
            raise RuntimeError("Embedding cluster/token mismatch")
        for row, label in zip(rows, labels.tolist()):
            row["cluster"] = label
        segments = group_tokens(rows, session_id, max_gap)
        payload = {
            "session_id": session_id,
            "wav_sha256": sha256_file(wav_path),
            "asr_source_sha256": sha256_file(source_path),
            "predicted_speaker_count": predicted_counts[session_id],
            "embedding_count": len(embeddings),
            "embedding_dimension": int(embeddings.shape[1]),
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
                    "tokens": len(rows),
                    "predicted_speakers": predicted_counts[session_id],
                    "output_speakers": len({row["speaker"] for row in segments}),
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
        "uses_validation_labels": False,
        "session_ids": session_ids,
        "predicted_speaker_counts": predicted_counts,
        "speaker_count_model": str(count_model_path),
        "session_list_sha256": sha256_file(split_dir / "val_sessions.txt"),
        "cv_manifest_sha256": sha256_file(root / "configs" / "cv" / "folds_v1.csv"),
        "config": config,
        "config_sha256": sha256_file(config_path),
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
