#!/usr/bin/env python3
"""Generate the frozen V7 test SegLST submission with capacity-aware routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import wave
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

from run_3dspeaker_relabel import relabel
from run_sortformer_relabel import (
    assign_speakers,
    duration_seconds,
    group_rows,
    parse_diarization,
    sha256_file,
    token_rows,
)


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?|[\u3400-\u4dbf\u4e00-\u9fff]")


def load_test_counts(
    root: Path, config: dict, session_ids: list[str]
) -> tuple[dict[str, int], Path, Path]:
    model_path = root / config["speaker_count_model"]
    feature_path = root / config["speaker_count_features"]
    artifact = joblib.load(model_path)
    if artifact["fold"] != "full" or len(artifact["train_sessions"]) != 106:
        raise RuntimeError("Final speaker-count model provenance failed")
    features = json.loads(feature_path.read_text(encoding="utf-8"))
    if (
        features.get("split") != "test"
        or features.get("uses_speaker_labels") is not False
        or features.get("uses_transcripts") is not False
        or len(features.get("sessions", {})) != 394
    ):
        raise RuntimeError("Label-free test speaker-count feature provenance failed")
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
    return dict(zip(session_ids, predictions.tolist())), model_path, feature_path


def validate_segments(segments: list[dict], wav_paths: list[Path]) -> dict:
    durations = {}
    for path in wav_paths:
        with wave.open(str(path), "rb") as handle:
            durations[path.stem] = handle.getnframes() / handle.getframerate()
    expected_fields = {"session_id", "speaker", "start_time", "end_time", "words"}
    for index, row in enumerate(segments):
        if set(row) != expected_fields:
            raise RuntimeError(f"SegLST schema mismatch at row {index}")
        session_id = row["session_id"]
        if session_id not in durations:
            raise RuntimeError(f"Unknown test session: {session_id}")
        if not isinstance(row["speaker"], str) or not re.fullmatch(r"spk[1-9]\d*", row["speaker"]):
            raise RuntimeError(f"Invalid speaker label at row {index}")
        if not (
            isinstance(row["start_time"], (int, float))
            and isinstance(row["end_time"], (int, float))
            and 0 <= row["start_time"] < row["end_time"] <= durations[session_id] + 0.01
        ):
            raise RuntimeError(f"Invalid timestamp at row {index}")
        if round(float(row["start_time"]), 2) != float(row["start_time"]) or round(
            float(row["end_time"]), 2
        ) != float(row["end_time"]):
            raise RuntimeError(f"Timestamp precision mismatch at row {index}")
        words = row["words"]
        if not isinstance(words, str) or not words or words != " ".join(words.split()):
            raise RuntimeError(f"Invalid words spacing at row {index}")
        tokens = words.split()
        if any(token != token.lower() or TOKEN_PATTERN.fullmatch(token) is None for token in tokens):
            raise RuntimeError(f"Invalid token at row {index}")
    actual_sessions = {row["session_id"] for row in segments}
    if actual_sessions != set(durations):
        raise RuntimeError("Submission session coverage mismatch")
    for session_id in actual_sessions:
        labels = sorted(
            {row["speaker"] for row in segments if row["session_id"] == session_id},
            key=lambda label: int(label[3:]),
        )
        if labels != [f"spk{index}" for index in range(1, len(labels) + 1)]:
            raise RuntimeError(f"Non-contiguous speaker labels for {session_id}")
    return {"sessions": len(actual_sessions), "segments": len(segments)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    test_dir = (root / "data" / "test").resolve()
    if test_dir.name != "test":
        raise RuntimeError("Final-test guard failed")
    wav_paths = sorted((test_dir / "wav").glob("*.wav"))
    if args.max_sessions is not None:
        wav_paths = wav_paths[: args.max_sessions]
    session_ids = [path.stem for path in wav_paths]
    if not session_ids or not all(path.parent == test_dir / "wav" for path in wav_paths):
        raise RuntimeError("Test audio discovery failed")

    source_name = config["asr_source_experiment"]
    if args.max_sessions is not None:
        source_name += "_smoke"
    source_dir = root / "outputs" / source_name / "sessions"
    if not all((source_dir / f"{session_id}.json").is_file() for session_id in session_ids):
        raise FileNotFoundError("Frozen Qwen test ASR cache is incomplete")
    predicted_counts, count_model_path, feature_path = load_test_counts(
        root, config, session_ids
    )

    checkpoint = (root / config["sortformer"]["checkpoint"]).resolve()
    from nemo.collections.asr.models import SortformerEncLabelModel

    device = torch.device(config["sortformer"]["device"])
    sortformer = SortformerEncLabelModel.restore_from(
        str(checkpoint), map_location=device
    ).to(device).eval()

    toolkit_dir = root / "third_party" / "3D-Speaker"
    sys.path.insert(0, str(toolkit_dir))
    os.environ.setdefault("MODELSCOPE_CACHE", str(root / "models" / "modelscope"))
    from speakerlab.bin.infer_diarization import Diarization3Dspeaker

    fallback = Diarization3Dspeaker(
        device=config["fallback_diarization"]["device"],
        include_overlap=False,
        model_cache_dir=str(root / "models" / "modelscope" / "3dspeaker"),
    )

    suffix = "_smoke" if args.max_sessions is not None else ""
    output_dir = root / "outputs" / config["name"] / f"test{suffix}"
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    max_speakers = int(config["sortformer"]["max_speakers"])
    max_gap = float(config["assignment"]["max_gap_seconds"])
    all_segments: list[dict] = []
    route_counts = {"sortformer": 0, "3dspeaker_fallback": 0}
    started = time.time()
    for position, (session_id, wav_path) in enumerate(zip(session_ids, wav_paths), start=1):
        session_path = session_dir / f"{session_id}.json"
        if session_path.exists() and not args.overwrite:
            payload = json.loads(session_path.read_text(encoding="utf-8"))
            all_segments.extend(payload["segments"])
            route_counts[payload["route"]] += 1
            print(json.dumps({"session": session_id, "status": "resumed"}), flush=True)
            continue
        session_started = time.time()
        source_path = source_dir / f"{session_id}.json"
        raw_result = json.loads(source_path.read_text(encoding="utf-8"))["raw_result"][0]
        predicted_count = predicted_counts[session_id]
        if predicted_count <= max_speakers:
            route = "sortformer"
            rows = token_rows(raw_result, session_id)
            raw_segments, raw_probabilities = sortformer.diarize(
                audio=[str(wav_path)],
                batch_size=int(config["sortformer"]["batch_size"]),
                include_tensor_outputs=True,
                num_workers=0,
                verbose=False,
            )
            if len(raw_segments) != 1 or len(raw_probabilities) != 1:
                raise RuntimeError("Unexpected Sortformer batch output")
            diarization = parse_diarization(raw_segments[0])
            assign_speakers(
                rows,
                diarization,
                raw_probabilities[0],
                duration_seconds(wav_path),
                predicted_count,
                bool(config["assignment"]["use_predicted_arrival_slots"]),
            )
            segments = group_rows(rows, session_id, max_gap)
        else:
            route = "3dspeaker_fallback"
            diarization = [
                [float(segment[0]), float(segment[1]), int(segment[2])]
                for segment in fallback(str(wav_path), speaker_num=predicted_count)
            ]
            segments = relabel(raw_result, diarization, session_id, max_gap)
        route_counts[route] += 1
        payload = {
            "session_id": session_id,
            "wav_sha256": sha256_file(wav_path),
            "asr_source_sha256": sha256_file(source_path),
            "predicted_speaker_count": predicted_count,
            "route": route,
            "diarization": diarization,
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
                    "predicted_count": predicted_count,
                    "route": route,
                    "segments": len(segments),
                    "speakers": len({row["speaker"] for row in segments}),
                    "elapsed_seconds": round(time.time() - session_started, 2),
                }
            ),
            flush=True,
        )

    audit = validate_segments(all_segments, wav_paths)
    prediction_path = output_dir / "hyp.seglst.json"
    prediction_path.write_text(
        json.dumps(all_segments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    submission_path = root / "submissions" / "v7_second_submission.seglst.json"
    if args.max_sessions is None:
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        submission_path.write_bytes(prediction_path.read_bytes())
    metadata = {
        "final_test_inference": True,
        "uses_test_for_training": False,
        "uses_test_for_model_selection": False,
        "smoke_only": args.max_sessions is not None,
        "session_ids": session_ids,
        "predicted_speaker_counts": predicted_counts,
        "route_counts": route_counts,
        "audit": audit,
        "config": config,
        "config_sha256": sha256_file(config_path),
        "speaker_count_model_sha256": sha256_file(count_model_path),
        "speaker_count_features_sha256": sha256_file(feature_path),
        "sortformer_checkpoint_sha256": sha256_file(checkpoint),
        "prediction_sha256": sha256_file(prediction_path),
        "elapsed_seconds": round(time.time() - started, 2),
        "argv": sys.argv,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "prediction": str(prediction_path),
                "submission": str(submission_path) if args.max_sessions is None else None,
                "sha256": metadata["prediction_sha256"],
                **audit,
                "route_counts": route_counts,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
