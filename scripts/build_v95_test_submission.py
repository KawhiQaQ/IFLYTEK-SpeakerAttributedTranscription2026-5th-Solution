#!/usr/bin/env python3
"""Build the deployable MOSS + full-development acoustic-graph fusion."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from build_v85_test_submission import (
    canonicalize_speakers,
    substitutions_on_moss,
)
from run_diarizen_relabel import assign_tracks
from run_segment_novel_speaker_fusion import fuse_session
from run_sortformer_relabel import group_rows, sha256_file, token_rows
from run_v7_test_submission import validate_segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("submissions/v95_moss_acoustic_graph_fusion.seglst.json"),
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    test_dir = (root / "data/test").resolve()
    if test_dir.name != "test":
        raise RuntimeError("Final-test guard failed")
    wav_paths = sorted((test_dir / "wav").glob("*.wav"))
    session_ids = [path.stem for path in wav_paths]
    sources = {
        "qwen": root / "outputs/test_asr_qwen3_1.7b/sessions",
        "aed": root / "outputs/test_fireredasr2_aed/sessions",
        "llm": root / "outputs/test_fireredasr2_llm/sessions",
        "moss": root / "outputs/test_moss_transcribe_diarize/sessions",
        "tracks": root / "outputs/v20_diarization_quality_router_submission/test/sessions",
        "graph": root / "outputs/v93_boundary_metric_graph_full_test/test/sessions",
    }
    missing = [
        str(directory / f"{session_id}.json")
        for directory in sources.values()
        for session_id in session_ids
        if not (directory / f"{session_id}.json").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Final-test cache incomplete ({len(missing)} missing): {missing[:5]}"
        )

    output_dir = root / "outputs/v95_moss_acoustic_graph_fusion/test"
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    output_segments: list[dict] = []
    audits: dict[str, dict] = {}
    started = time.time()
    for position, session_id in enumerate(session_ids, start=1):
        paths = {
            name: directory / f"{session_id}.json"
            for name, directory in sources.items()
        }
        payloads = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in paths.items()
        }
        # Reproduce the V29 base partition: FireRed AED tokens on deployable V20 tracks.
        rows = token_rows(payloads["aed"]["raw_result"][0], session_id)
        turns = [
            [
                float(row["start_time"]),
                float(row["end_time"]),
                str(row["speaker"]),
            ]
            for row in payloads["tracks"]["segments"]
        ]
        assignment_audit = assign_tracks(rows, turns)
        modular_segments = group_rows(rows, session_id, 0.8)
        # Reproduce V38 with a full-development acoustic metric checkpoint.
        v38_segments, v38_audit = fuse_session(
            session_id, modular_segments, payloads["graph"]["segments"]
        )
        # Preserve MOSS native detections/timing and accept majority substitutions only.
        moss_segments, text_audit = substitutions_on_moss(
            payloads["moss"],
            [payloads["aed"], payloads["qwen"], payloads["llm"]],
        )
        fused_segments, fusion_audit = fuse_session(
            session_id, moss_segments, v38_segments
        )
        fused_segments = canonicalize_speakers(fused_segments)
        audit = {
            "v20_assignment": assignment_audit,
            "v38_fusion": v38_audit,
            "moss_text": text_audit,
            "moss_v38_fusion": fusion_audit,
        }
        session_payload = {
            "session_id": session_id,
            "final_test_inference": True,
            "uses_test_for_training": False,
            "uses_test_for_model_selection": False,
            "source_sha256": {
                name: sha256_file(path) for name, path in paths.items()
            },
            "fusion_audit": audit,
            "segments": fused_segments,
        }
        (session_dir / f"{session_id}.json").write_text(
            json.dumps(session_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_segments.extend(fused_segments)
        audits[session_id] = audit
        print(
            json.dumps(
                {
                    "session": session_id,
                    "position": position,
                    "total": len(session_ids),
                    "moss_speakers": fusion_audit["base_speaker_count"],
                    "v38_speakers": fusion_audit["complementary_speaker_count"],
                    "novel_speakers": fusion_audit["novel_speaker_count"],
                    "changed_tokens": fusion_audit["changed_tokens"],
                }
            ),
            flush=True,
        )

    validation = validate_segments(output_segments, wav_paths)
    prediction_path = output_dir / "hyp.seglst.json"
    prediction_path.write_text(
        json.dumps(output_segments, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    submission_path = (root / args.submission).resolve()
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_bytes(prediction_path.read_bytes())
    metadata = {
        "final_test_inference": True,
        "uses_test_for_training": False,
        "uses_test_for_model_selection": False,
        "cv_architecture_frozen_from": "v90_moss_v38_novel_fusion",
        "full_training_epoch_selection": "median_of_fold_pure_best_epochs_21_and_7",
        "session_ids": session_ids,
        "fusion_audits": audits,
        "audit": validation,
        "prediction_sha256": sha256_file(prediction_path),
        "submission_sha256": sha256_file(submission_path),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "submission": str(submission_path),
                "sha256": metadata["submission_sha256"],
                "novel_sessions": sum(
                    row["moss_v38_fusion"]["novel_speaker_count"] > 0
                    for row in audits.values()
                ),
                **validation,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
