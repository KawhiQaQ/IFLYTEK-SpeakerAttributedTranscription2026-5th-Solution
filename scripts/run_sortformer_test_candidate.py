#!/usr/bin/env python3
"""Generate one frozen Sortformer test candidate for the V20 router."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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
from run_sortformer_turn_clustering import (
    assign_refined_clusters,
    cluster_turns,
)
from run_v7_test_submission import load_test_counts, validate_segments


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
    if not all((source_dir / f"{sid}.json").is_file() for sid in session_ids):
        raise FileNotFoundError("Frozen Qwen test ASR cache is incomplete")
    predicted_counts, count_model_path, feature_path = load_test_counts(
        root, config, session_ids
    )

    from nemo.collections.asr.models import SortformerEncLabelModel

    checkpoint = (root / config["sortformer"]["checkpoint"]).resolve()
    checkpoint_sha256 = sha256_file(checkpoint)
    expected_checkpoint_sha256 = config["sortformer"].get("checkpoint_sha256")
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_sha256 != expected_checkpoint_sha256
    ):
        raise RuntimeError("Sortformer checkpoint SHA-256 mismatch")
    device = torch.device(config["sortformer"]["device"])
    sortformer = SortformerEncLabelModel.restore_from(
        str(checkpoint), map_location=device
    ).to(device).eval()

    toolkit_dir = root / "third_party" / "3D-Speaker"
    sys.path.insert(0, str(toolkit_dir))
    os.environ.setdefault("MODELSCOPE_CACHE", str(root / "models" / "modelscope"))
    from speakerlab.bin.infer_diarization import Diarization3Dspeaker
    from speakerlab.utils.fileio import load_audio

    fallback = Diarization3Dspeaker(
        device=config["fallback_diarization"]["device"],
        include_overlap=False,
        model_cache_dir=str(root / "models" / "modelscope" / "3dspeaker"),
    )
    refine = "embedding" in config
    embedding_config = config.get("embedding", {})
    min_pure_seconds = float(embedding_config.get("min_pure_seconds", 0.5))
    require_count_agreement = bool(
        embedding_config.get("require_count_agreement", False)
    )

    suffix = "_smoke" if args.max_sessions is not None else ""
    output_dir = root / "outputs" / config["name"] / f"test{suffix}"
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    max_speakers = int(config["sortformer"]["max_speakers"])
    max_gap = float(config["assignment"]["max_gap_seconds"])
    all_segments: list[dict] = []
    route_counts = {"sortformer": 0, "sortformer_refined": 0, "3dspeaker_fallback": 0}
    started = time.time()
    for position, (session_id, wav_path) in enumerate(zip(session_ids, wav_paths), start=1):
        session_path = session_dir / f"{session_id}.json"
        if session_path.exists() and not args.overwrite:
            payload = json.loads(session_path.read_text(encoding="utf-8"))
            all_segments.extend(payload["segments"])
            route_counts[payload["route"]] += 1
            continue
        source_path = source_dir / f"{session_id}.json"
        raw_result = json.loads(source_path.read_text(encoding="utf-8"))["raw_result"][0]
        predicted_count = predicted_counts[session_id]
        raw_diarization = None
        refined_diarization = None
        turn_audit = None
        if predicted_count <= max_speakers:
            route = "sortformer_refined" if refine else "sortformer"
            rows = token_rows(raw_result, session_id)
            raw_segments, raw_probabilities = sortformer.diarize(
                audio=[str(wav_path)],
                batch_size=int(config["sortformer"]["batch_size"]),
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
                predicted_count,
                bool(config["assignment"]["use_predicted_arrival_slots"]),
            )
            if refine:
                wav_data = load_audio(str(wav_path), None, fallback.fs)
                refined_diarization, turn_audit = cluster_turns(
                    raw_diarization,
                    wav_data,
                    fallback,
                    min_pure_seconds,
                    require_count_agreement,
                )
                assign_refined_clusters(rows, raw_diarization, refined_diarization)
            segments = group_rows(rows, session_id, max_gap)
        else:
            route = "3dspeaker_fallback"
            raw_diarization = [
                [float(segment[0]), float(segment[1]), int(segment[2])]
                for segment in fallback(str(wav_path), speaker_num=predicted_count)
            ]
            segments = relabel(raw_result, raw_diarization, session_id, max_gap)
        route_counts[route] += 1
        payload = {
            "session_id": session_id,
            "development_only": False,
            "final_test_inference": True,
            "inference_only": True,
            "uses_validation_labels": False,
            "uses_test_data": True,
            "uses_test_for_training_or_selection": False,
            "wav_sha256": sha256_file(wav_path),
            "asr_source_sha256": sha256_file(source_path),
            "predicted_speaker_count": predicted_count,
            "route": route,
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
                    "predicted_count": predicted_count,
                    "route": route,
                    "speakers": len({row["speaker"] for row in segments}),
                }
            ),
            flush=True,
        )

    audit = validate_segments(all_segments, wav_paths)
    prediction_path = output_dir / "hyp.seglst.json"
    prediction_path.write_text(
        json.dumps(all_segments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "final_test_inference": True,
        "candidate_only": True,
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
        "sortformer_checkpoint_sha256": checkpoint_sha256,
        "prediction_sha256": sha256_file(prediction_path),
        "elapsed_seconds": round(time.time() - started, 2),
        "argv": sys.argv,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"prediction": str(prediction_path), **audit, "route_counts": route_counts}), flush=True)


if __name__ == "__main__":
    main()
