#!/usr/bin/env python3
"""Select streaming or offline diarization using speaker-embedding coherence.

The router is deliberately label-free: for each candidate diarization it
extracts CAMPPlus embeddings from non-overlapped turn regions and measures the
duration-weighted margin between the assigned speaker centroid and the closest
other centroid. The offline candidate is accepted only when its margin exceeds
the streaming candidate by a configured conservative amount.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pure_turns(
    diarization: list[list[float]], min_pure_seconds: float
) -> list[tuple[float, float, int, float]]:
    rows: list[tuple[float, float, int, float]] = []
    for start, end, speaker in diarization:
        pieces = [(float(start), float(end))]
        for other_start, other_end, other_speaker in diarization:
            if int(other_speaker) == int(speaker):
                continue
            remaining: list[tuple[float, float]] = []
            for left, right in pieces:
                if float(other_end) <= left or float(other_start) >= right:
                    remaining.append((left, right))
                    continue
                if left < float(other_start):
                    remaining.append((left, min(right, float(other_start))))
                if float(other_end) < right:
                    remaining.append((max(left, float(other_end)), right))
            pieces = remaining
        if not pieces:
            continue
        left, right = max(pieces, key=lambda item: item[1] - item[0])
        duration = right - left
        if duration >= min_pure_seconds:
            rows.append((left, right, int(speaker), duration))
    return rows


def coherence_score(embeddings: np.ndarray, turns: list[tuple]) -> dict:
    labels = np.asarray([row[2] for row in turns], dtype=np.int64)
    weights = np.asarray([row[3] for row in turns], dtype=np.float64)
    unique = np.unique(labels)
    if len(unique) < 2:
        return {
            "score": -9.0,
            "speaker_count": int(len(unique)),
            "embedding_count": int(len(turns)),
            "pure_duration": float(weights.sum()) if len(weights) else 0.0,
        }
    embeddings = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9
    )
    centroids = []
    for speaker in unique:
        selected = labels == speaker
        centroid = np.average(
            embeddings[selected], axis=0, weights=weights[selected]
        )
        centroids.append(centroid / max(np.linalg.norm(centroid), 1e-9))
    centroids = np.stack(centroids)
    own = np.asarray(
        [int(np.flatnonzero(unique == speaker)[0]) for speaker in labels]
    )
    similarities = embeddings @ centroids.T
    own_similarity = similarities[np.arange(len(embeddings)), own]
    masked = similarities.copy()
    masked[np.arange(len(embeddings)), own] = -99.0
    margin = own_similarity - masked.max(axis=1)
    return {
        "score": float(np.average(margin, weights=weights)),
        "speaker_count": int(len(unique)),
        "embedding_count": int(len(turns)),
        "pure_duration": float(weights.sum()),
    }


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
    wav_dir = dev_dir / "wav"
    if not session_ids or not all((wav_dir / f"{sid}.wav").is_file() for sid in session_ids):
        raise RuntimeError("Validation input guard failed")

    streaming_dir = root / "outputs" / config["streaming_experiment"] / f"fold_{fold}"
    offline_dir = root / "outputs" / config["offline_experiment"] / f"fold_{fold}"
    if not all((streaming_dir / "sessions" / f"{sid}.json").is_file() for sid in session_ids):
        raise FileNotFoundError("Streaming candidate predictions are incomplete")
    if not all((offline_dir / "sessions" / f"{sid}.json").is_file() for sid in session_ids):
        raise FileNotFoundError("Offline candidate predictions are incomplete")

    toolkit_dir = root / "third_party" / "3D-Speaker"
    sys.path.insert(0, str(toolkit_dir))
    os.environ.setdefault("MODELSCOPE_CACHE", str(root / "models" / "modelscope"))
    from speakerlab.bin.infer_diarization import Diarization3Dspeaker
    from speakerlab.utils.fileio import load_audio

    block = config["router"]
    diarizer = Diarization3Dspeaker(
        device=block["device"],
        include_overlap=False,
        model_cache_dir=str(root / block["model_cache_dir"]),
    )
    min_pure_seconds = float(block["min_pure_seconds"])
    min_quality_margin = float(block["min_quality_margin"])

    suffix = "_smoke" if args.max_sessions is not None else ""
    output_dir = root / "outputs" / config["name"] / f"fold_{fold}{suffix}"
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    all_segments: list[dict] = []
    decisions: dict[str, dict] = {}
    started = time.time()
    for position, session_id in enumerate(session_ids, start=1):
        output_path = session_dir / f"{session_id}.json"
        if output_path.exists() and not args.overwrite:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            all_segments.extend(payload["segments"])
            decisions[session_id] = payload["decision"]
            continue
        streaming_path = streaming_dir / "sessions" / f"{session_id}.json"
        offline_path = offline_dir / "sessions" / f"{session_id}.json"
        streaming = json.loads(streaming_path.read_text(encoding="utf-8"))
        offline = json.loads(offline_path.read_text(encoding="utf-8"))
        streaming_diarization = streaming.get("raw_diarization")
        offline_diarization = offline.get("diarization")

        if not streaming_diarization or not offline_diarization:
            selected = "streaming"
            streaming_quality = None
            offline_quality = None
            quality_margin = None
            reason = "capacity_fallback_or_missing_diarization"
        else:
            streaming_turns = pure_turns(streaming_diarization, min_pure_seconds)
            offline_turns = pure_turns(offline_diarization, min_pure_seconds)
            windows = [
                [row[0], row[1]] for row in streaming_turns + offline_turns
            ]
            wav_data = load_audio(
                str(wav_dir / f"{session_id}.wav"), None, diarizer.fs
            )
            embeddings = diarizer.do_emb_extraction(windows, wav_data)
            split = len(streaming_turns)
            streaming_quality = coherence_score(
                embeddings[:split], streaming_turns
            )
            offline_quality = coherence_score(embeddings[split:], offline_turns)
            quality_margin = offline_quality["score"] - streaming_quality["score"]
            selected = (
                "offline" if quality_margin > min_quality_margin else "streaming"
            )
            reason = "quality_margin"

        source = offline if selected == "offline" else streaming
        decision = {
            "selected": selected,
            "reason": reason,
            "streaming_quality": streaming_quality,
            "offline_quality": offline_quality,
            "offline_minus_streaming": quality_margin,
        }
        payload = {
            "session_id": session_id,
            "uses_reference_labels": False,
            "streaming_source_sha256": sha256_file(streaming_path),
            "offline_source_sha256": sha256_file(offline_path),
            "decision": decision,
            "segments": source["segments"],
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        decisions[session_id] = decision
        all_segments.extend(payload["segments"])
        print(
            json.dumps(
                {
                    "session": session_id,
                    "position": position,
                    "total": len(session_ids),
                    "selected": selected,
                    "quality_margin": quality_margin,
                }
            ),
            flush=True,
        )

    prediction_path = output_dir / "hyp.seglst.json"
    prediction_path.write_text(
        json.dumps(all_segments, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "experiment": config["name"],
        "fold": fold,
        "smoke_only": args.max_sessions is not None,
        "development_only": True,
        "uses_validation_labels": False,
        "uses_test_data": False,
        "session_ids": session_ids,
        "decisions": decisions,
        "offline_sessions": [
            sid for sid, row in decisions.items() if row["selected"] == "offline"
        ],
        "cv_manifest_sha256": sha256_file(root / "configs" / "cv" / "folds_v1.csv"),
        "config": config,
        "config_sha256": sha256_file(config_path),
        "streaming_prediction_sha256": sha256_file(streaming_dir / "hyp.seglst.json"),
        "offline_prediction_sha256": sha256_file(offline_dir / "hyp.seglst.json"),
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
