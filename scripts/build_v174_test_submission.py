#!/usr/bin/env python3
"""Build final-test V174 with the frozen V173 novel-speaker energy."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import yaml

from build_v85_test_submission import canonicalize_speakers, substitutions_on_moss
from novel_speaker_energy_features import FEATURE_NAMES, energy_features, multiscale_embeddings
from run_diarizen_relabel import assign_tracks
from run_partition_quality_gate import frame_labels
from run_segment_novel_speaker_fusion import fuse_session
from run_sortformer_relabel import group_rows, sha256_file, token_rows
from run_v7_test_submission import validate_segments


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sigmoid(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    test_dir = (root / "data/test").resolve()
    if test_dir.name != "test":
        raise RuntimeError("Final-test guard failed")
    wav_paths = sorted((test_dir / "wav").glob("*.wav"))
    session_ids = [path.stem for path in wav_paths]
    if len(session_ids) != 394:
        raise RuntimeError(f"Unexpected final-test coverage: {len(session_ids)}")

    source_roots = {
        name: root / relative for name, relative in config["sources"].items()
    }
    feature_root = root / config["feature_root"]
    checkpoint_path = root / config["checkpoint_path"]
    feature_metadata_path = feature_root / "metadata.json"
    required = [
        directory / f"{session_id}.json"
        for directory in source_roots.values()
        for session_id in session_ids
    ] + [feature_root / f"{session_id}.pt" for session_id in session_ids]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Final-test inference cache incomplete: {missing[:5]}")

    checkpoint = load_json(checkpoint_path)
    feature_metadata = load_json(feature_metadata_path)
    if (
        checkpoint.get("trained_from_cv_architecture") != config["cv_architecture"]
        or checkpoint.get("uses_test_data") is not False
        or checkpoint.get("uses_test_for_training_or_selection") is not False
        or len(checkpoint.get("train_sessions", [])) != 106
        or checkpoint.get("feature_names") != FEATURE_NAMES
    ):
        raise RuntimeError("Frozen full-development energy checkpoint audit failed")
    if (
        feature_metadata.get("scope") != "test"
        or feature_metadata.get("uses_speaker_labels") is not False
        or feature_metadata.get("uses_test_data") is not True
        or feature_metadata.get("inference_only") is not True
        or feature_metadata.get("sessions") != session_ids
    ):
        raise RuntimeError("Label-free final-test feature audit failed")
    if checkpoint_path.stat().st_mtime > feature_metadata_path.stat().st_mtime:
        raise RuntimeError("Checkpoint was not frozen before test feature extraction")

    mean = torch.tensor(checkpoint["scaler_mean"], dtype=torch.float32)
    scale = torch.tensor(checkpoint["scaler_scale"], dtype=torch.float32)
    coefficient = torch.tensor(checkpoint["coefficient"], dtype=torch.float32)
    intercept = float(checkpoint["intercept"])
    boundary = float(config["probability_boundary"])
    if boundary != 0.5:
        raise RuntimeError("V173 fixed probability boundary must remain 0.5")

    output_root = root / config["output_root"]
    session_root = output_root / "sessions"
    session_root.mkdir(parents=True, exist_ok=True)
    submission_path = root / config["submission_path"]
    audit_path = root / config["audit_path"]
    if (submission_path.exists() or audit_path.exists()) and not args.overwrite:
        raise FileExistsError("V174 submission already exists; pass --overwrite")

    output_rows: list[dict] = []
    decisions: dict[str, dict] = {}
    started = time.time()
    for position, session_id in enumerate(session_ids, start=1):
        paths = {
            name: directory / f"{session_id}.json"
            for name, directory in source_roots.items()
        }
        payloads = {name: load_json(path) for name, path in paths.items()}
        refined = canonicalize_speakers(payloads["refined"]["segments"])
        native, native_text_audit = substitutions_on_moss(
            payloads["moss"],
            [payloads["aed"], payloads["qwen"], payloads["llm"]],
        )
        native = canonicalize_speakers(native)
        # Reproduce the exact V38 complementary partition consumed by V95.
        rows = token_rows(payloads["aed"]["raw_result"][0], session_id)
        turns = [
            [
                float(row["start_time"]),
                float(row["end_time"]),
                str(row["speaker"]),
            ]
            for row in payloads["tracks"]["segments"]
        ]
        assign_tracks(rows, turns)
        modular_segments = group_rows(rows, session_id, 0.8)
        v38_segments, _ = fuse_session(
            session_id, modular_segments, payloads["graph"]["segments"]
        )
        reconstructed_injected, reconstructed_audit = fuse_session(
            session_id, native, v38_segments
        )
        reconstructed_injected = canonicalize_speakers(reconstructed_injected)
        if (
            reconstructed_injected != payloads["injected"]["segments"]
            or reconstructed_audit
            != payloads["injected"]["fusion_audit"]["moss_v38_fusion"]
        ):
            raise RuntimeError(f"V95 reconstruction mismatch: {session_id}")
        novel_speakers = payloads["injected"]["fusion_audit"][
            "moss_v38_fusion"
        ]["novel_complementary_speakers"]
        candidate_decisions: list[dict] = []
        feature_path = feature_root / f"{session_id}.pt"
        if novel_speakers:
            acoustic = torch.load(feature_path, map_location="cpu", weights_only=True)
            if (
                acoustic.get("uses_speaker_labels") is not False
                or acoustic.get("uses_test_data") is not True
                or acoustic.get("inference_only") is not True
            ):
                raise RuntimeError(f"Test energy feature provenance failed: {session_id}")
            embeddings = multiscale_embeddings(acoustic["features"])
            complementary_labels = frame_labels(
                acoustic["centers"], v38_segments
            )
            base_labels = frame_labels(acoustic["centers"], native)
            for novel_speaker in novel_speakers:
                candidate = torch.tensor(
                    [speaker == novel_speaker for speaker in complementary_labels]
                )
                base_masks = []
                for base_speaker in sorted(set(base_labels)):
                    mask = torch.tensor(
                        [speaker == base_speaker for speaker in base_labels]
                    ) & ~candidate
                    if int(mask.sum()):
                        base_masks.append(mask)
                if not int(candidate.sum()) or not base_masks:
                    candidate_decisions.append(
                        {
                            "speaker": novel_speaker,
                            "logit": None,
                            "probability": 0.0,
                            "accepted": False,
                            "candidate_frames": int(candidate.sum()),
                            "reason": "insufficient_acoustic_support",
                            "energy_features": None,
                        }
                    )
                    continue
                values = energy_features(embeddings, candidate, base_masks)
                standardized = (values - mean) / scale
                logit = float(standardized @ coefficient + intercept)
                probability = sigmoid(logit)
                candidate_decisions.append(
                    {
                        "speaker": novel_speaker,
                        "logit": logit,
                        "probability": probability,
                        "accepted": probability >= boundary,
                        "candidate_frames": int(candidate.sum()),
                        "reason": "fixed_v173_energy_boundary",
                        "energy_features": {
                            name: float(value)
                            for name, value in zip(FEATURE_NAMES, values)
                        },
                    }
                )

        reject_novel = bool(novel_speakers) and not all(
            row["accepted"] for row in candidate_decisions
        )
        selected = native if reject_novel else refined
        decision = {
            "selected": (
                "native_without_unverified_novel"
                if reject_novel
                else "v156_refined_with_verified_novel"
            ),
            "novel_candidates": candidate_decisions,
            "native_text_audit": native_text_audit if reject_novel else None,
        }
        session_payload = {
            "session_id": session_id,
            "final_test_inference": True,
            "uses_test_for_training": False,
            "uses_test_for_model_selection": False,
            "source_sha256": {name: sha256_file(path) for name, path in paths.items()},
            "feature_source_sha256": (
                sha256_file(feature_path) if novel_speakers else None
            ),
            "energy_checkpoint_sha256": sha256_file(checkpoint_path),
            "decision": decision,
            "segments": selected,
        }
        (session_root / f"{session_id}.json").write_text(
            json.dumps(session_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_rows.extend(selected)
        decisions[session_id] = decision
        if novel_speakers:
            print(
                json.dumps(
                    {
                        "session": session_id,
                        "position": position,
                        "novel_candidates": candidate_decisions,
                        "selected": decision["selected"],
                    }
                ),
                flush=True,
            )

    validation = validate_segments(output_rows, wav_paths)
    validation["tokens"] = sum(len(row["words"].split()) for row in output_rows)
    prediction_path = output_root / "hyp.seglst.json"
    prediction_path.write_text(
        json.dumps(output_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_bytes(prediction_path.read_bytes())
    fallback_sessions = [
        session_id
        for session_id, decision in decisions.items()
        if decision["selected"] == "native_without_unverified_novel"
    ]
    metadata = {
        "experiment": config["name"],
        "final_test_inference": True,
        "uses_test_for_training": False,
        "uses_test_for_model_selection": False,
        "architecture_frozen_from_cv": config["cv_architecture"],
        "full_development_fit": config["full_fit"],
        "fixed_probability_boundary": boundary,
        "checkpoint_precedes_test_features": True,
        "session_ids": session_ids,
        "decisions": decisions,
        "novel_candidate_sessions": [
            session_id
            for session_id, decision in decisions.items()
            if decision["novel_candidates"]
        ],
        "native_fallback_sessions": fallback_sessions,
        "validation": validation,
        "config_sha256": sha256_file(config_path),
        "energy_checkpoint_sha256": sha256_file(checkpoint_path),
        "feature_metadata_sha256": sha256_file(feature_metadata_path),
        "prediction_sha256": sha256_file(prediction_path),
        "submission_sha256": sha256_file(submission_path),
        "elapsed_seconds": round(time.time() - started, 3),
        "argv": sys.argv,
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "submission": str(submission_path),
                "audit": str(audit_path),
                "sha256": metadata["submission_sha256"],
                "novel_candidate_sessions": len(metadata["novel_candidate_sessions"]),
                "native_fallback_sessions": len(fallback_sessions),
                **validation,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
