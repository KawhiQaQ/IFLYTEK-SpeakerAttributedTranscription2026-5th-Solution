#!/usr/bin/env python3
"""Refine Sortformer speaker identity across turns with CAMPPlus embeddings.

Sortformer supplies speech activity, overlap handling, and token-level slot
scores.  CAMPPlus only reclusters non-overlapped speech from those turns so a
speaker can retain one identity after a long pause or a change in speaking
style.  No reference labels are read by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

from run_sortformer_relabel import (
    assign_speakers,
    duration_seconds,
    group_rows,
    interval_overlap,
    load_fold_pure_counts,
    load_reference_counts,
    parse_diarization,
    sha256_file,
    token_rows,
)


def longest_pure_interval(
    segment: list[float], diarization: list[list[float]]
) -> tuple[float, float] | None:
    """Return the longest region not overlapped by another Sortformer slot."""
    start, end, speaker = float(segment[0]), float(segment[1]), int(segment[2])
    pieces = [(start, end)]
    for other_start, other_end, other_speaker in diarization:
        if int(other_speaker) == speaker:
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
    return max(pieces, key=lambda item: item[1] - item[0]) if pieces else None


def midpoint_distance(interval: tuple[float, float], segment: list[float]) -> float:
    left_midpoint = (interval[0] + interval[1]) / 2
    right_midpoint = (float(segment[0]) + float(segment[1])) / 2
    return abs(left_midpoint - right_midpoint)


def cluster_turns(
    diarization: list[list[float]],
    wav_data,
    diarizer,
    min_pure_seconds: float,
    require_count_agreement: bool,
) -> tuple[list[list[float]], list[dict]]:
    """Cluster sufficiently long pure turn regions and propagate short turns."""
    active_slots = sorted({int(segment[2]) for segment in diarization})
    if len(active_slots) <= 1:
        return [list(segment) for segment in diarization], []

    embedding_indices: list[int] = []
    embedding_intervals: list[list[float]] = []
    audit_rows: list[dict] = []
    for index, segment in enumerate(diarization):
        pure = longest_pure_interval(segment, diarization)
        pure_duration = 0.0 if pure is None else pure[1] - pure[0]
        audit_rows.append(
            {
                "segment_index": index,
                "raw_speaker": int(segment[2]),
                "pure_interval": list(pure) if pure is not None else None,
                "pure_duration": round(pure_duration, 4),
            }
        )
        if pure is not None and pure_duration >= min_pure_seconds:
            embedding_indices.append(index)
            embedding_intervals.append([pure[0], pure[1]])

    if len(embedding_indices) < len(active_slots):
        for row, segment in zip(audit_rows, diarization):
            row["cluster"] = int(segment[2])
            row["fallback_reason"] = "too_few_pure_turns"
        return [list(segment) for segment in diarization], audit_rows

    embeddings = diarizer.do_emb_extraction(embedding_intervals, wav_data)
    # The 3D-Speaker backend intentionally switches to thresholded AHC when
    # there are fewer than 40 embeddings. In that branch an oracle count is
    # not consumed, so treat the returned cardinality as an independent
    # identity-consistency check rather than pretending it was forced.
    labels = diarizer.cluster(embeddings).astype(int).tolist()
    estimated_count = len(set(labels))
    if require_count_agreement and estimated_count != len(active_slots):
        for row, segment in zip(audit_rows, diarization):
            row["cluster"] = int(segment[2])
            row["estimated_cluster_count"] = estimated_count
            row["fallback_reason"] = "cluster_count_disagrees_with_sortformer"
        return [list(segment) for segment in diarization], audit_rows
    cluster_by_index = dict(zip(embedding_indices, labels))

    # A short or fully overlapped segment inherits the closest embedded turn
    # from its original Sortformer slot. This keeps overlap behavior stable.
    for index, segment in enumerate(diarization):
        if index in cluster_by_index:
            continue
        same_slot = [
            candidate
            for candidate in embedding_indices
            if int(diarization[candidate][2]) == int(segment[2])
        ]
        candidates = same_slot or embedding_indices
        nearest = min(
            candidates,
            key=lambda candidate: midpoint_distance(
                (float(segment[0]), float(segment[1])), diarization[candidate]
            ),
        )
        cluster_by_index[index] = int(cluster_by_index[nearest])

    refined = []
    for index, segment in enumerate(diarization):
        cluster = int(cluster_by_index[index])
        audit_rows[index]["cluster"] = cluster
        refined.append([float(segment[0]), float(segment[1]), cluster])
    return refined, audit_rows


def assign_refined_clusters(
    rows: list[dict],
    raw_diarization: list[list[float]],
    refined_diarization: list[list[float]],
) -> None:
    for row in rows:
        raw_speaker = int(row["raw_speaker"])
        candidates = [
            index
            for index, segment in enumerate(raw_diarization)
            if int(segment[2]) == raw_speaker
        ]
        if not candidates:
            raise RuntimeError("Token speaker has no matching Sortformer turn")
        token_interval = (float(row["start"]), float(row["end"]))
        index = max(
            candidates,
            key=lambda candidate: (
                interval_overlap(
                    token_interval,
                    (
                        float(raw_diarization[candidate][0]),
                        float(raw_diarization[candidate][1]),
                    ),
                ),
                -midpoint_distance(token_interval, raw_diarization[candidate]),
            ),
        )
        row["raw_speaker"] = int(refined_diarization[index][2])


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
    if not session_ids or not all(path.is_file() for path in wav_paths):
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

    from nemo.collections.asr.models import SortformerEncLabelModel

    model_config = config["sortformer"]
    checkpoint = (root / str(model_config["checkpoint"]).format(fold=fold)).resolve()
    sortformer = SortformerEncLabelModel.restore_from(
        str(checkpoint), map_location=torch.device(model_config["device"])
    ).to(model_config["device"]).eval()

    toolkit_dir = root / "third_party" / "3D-Speaker"
    if not toolkit_dir.is_dir():
        raise FileNotFoundError(toolkit_dir)
    sys.path.insert(0, str(toolkit_dir))
    os.environ.setdefault("MODELSCOPE_CACHE", str(root / "models" / "modelscope"))
    from speakerlab.bin.infer_diarization import Diarization3Dspeaker
    from speakerlab.utils.fileio import load_audio

    embedding_config = config["embedding"]
    diarizer = Diarization3Dspeaker(
        device=embedding_config["device"],
        include_overlap=False,
        model_cache_dir=str(root / embedding_config["model_cache_dir"]),
    )
    min_pure_seconds = float(embedding_config["min_pure_seconds"])
    require_count_agreement = bool(
        embedding_config.get("require_count_agreement", False)
    )
    max_speakers = int(model_config["max_speakers"])
    max_gap = float(config["assignment"]["max_gap_seconds"])

    suffix = "_smoke" if args.max_sessions is not None else ""
    if args.diagnostic_oracle_speaker_count:
        suffix += "_diagnostic_oracle_speaker_count"
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
        asr_path = asr_dir / f"{session_id}.json"
        fallback_path = fallback_dir / f"{session_id}.json"
        used_fallback = predicted_counts[session_id] > max_speakers
        raw_diarization = None
        refined_diarization = None
        turn_audit = None
        if used_fallback:
            segments = json.loads(fallback_path.read_text(encoding="utf-8"))["segments"]
        else:
            source = json.loads(asr_path.read_text(encoding="utf-8"))
            rows = token_rows(source["raw_result"][0], session_id)
            raw_segments, raw_probabilities = sortformer.diarize(
                audio=[str(wav_path)],
                batch_size=int(model_config["batch_size"]),
                include_tensor_outputs=True,
                num_workers=0,
                verbose=False,
            )
            raw_diarization = parse_diarization(raw_segments[0])
            assign_speakers(
                rows,
                raw_diarization,
                raw_probabilities[0],
                duration_seconds(wav_path),
                predicted_counts[session_id],
                False,
            )
            wav_data = load_audio(str(wav_path), None, diarizer.fs)
            refined_diarization, turn_audit = cluster_turns(
                raw_diarization,
                wav_data,
                diarizer,
                min_pure_seconds,
                require_count_agreement,
            )
            assign_refined_clusters(rows, raw_diarization, refined_diarization)
            segments = group_rows(rows, session_id, max_gap)

        payload = {
            "session_id": session_id,
            "wav_sha256": sha256_file(wav_path),
            "asr_source_sha256": sha256_file(asr_path),
            "predicted_speaker_count": predicted_counts[session_id],
            "used_capacity_fallback": used_fallback,
            "fallback_source_sha256": sha256_file(fallback_path) if used_fallback else None,
            "raw_diarization": raw_diarization,
            "refined_diarization": refined_diarization,
            "turn_audit": turn_audit,
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
                    "fallback": used_fallback,
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
        "uses_test_data": False,
        "session_ids": session_ids,
        "count_model": str(count_model_path),
        "cv_manifest_sha256": sha256_file(root / "configs" / "cv" / "folds_v1.csv"),
        "config": config,
        "config_sha256": sha256_file(config_path),
        "sortformer_checkpoint_sha256": sha256_file(checkpoint),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("nemo-toolkit", "torch")
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
