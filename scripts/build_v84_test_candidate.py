#!/usr/bin/env python3
"""Build the frozen V84 modular candidate on final-test inference caches."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from build_v85_test_submission import canonicalize_speakers, majority_on_aed
from run_diarizen_relabel import assign_tracks
from run_sortformer_relabel import group_rows, sha256_file, token_rows
from run_v7_test_submission import validate_segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    test_dir = (root / "data/test").resolve()
    if test_dir.name != "test":
        raise RuntimeError("Final-test path guard failed")
    wav_paths = sorted((test_dir / "wav").glob("*.wav"))
    session_ids = [path.stem for path in wav_paths]
    sources = {
        "qwen": root / "outputs/test_asr_qwen3_1.7b/sessions",
        "aed": root / "outputs/test_fireredasr2_aed/sessions",
        "llm": root / "outputs/test_fireredasr2_llm/sessions",
        "moss": root / "outputs/test_moss_transcribe_diarize/sessions",
        "tracks": root
        / "outputs/v20_diarization_quality_router_submission/test/sessions",
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

    output_dir = root / "outputs/v84_moss_consensus_v20_tracks/test"
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    output_segments = []
    audits = {}
    started = time.time()
    for position, session_id in enumerate(session_ids, start=1):
        output_path = session_dir / f"{session_id}.json"
        if output_path.exists() and not args.overwrite:
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            output_segments.extend(saved["segments"])
            audits[session_id] = saved["audit"]
            continue
        paths = {
            name: directory / f"{session_id}.json"
            for name, directory in sources.items()
        }
        payloads = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in paths.items()
        }
        consensus_result, consensus_audit = majority_on_aed(
            payloads["aed"],
            [payloads["qwen"], payloads["llm"], payloads["moss"]],
            session_id,
        )
        rows = token_rows(consensus_result, session_id)
        turns = [
            [
                float(row["start_time"]),
                float(row["end_time"]),
                str(row["speaker"]),
            ]
            for row in payloads["tracks"]["segments"]
        ]
        assignment_audit = assign_tracks(rows, turns)
        segments = canonicalize_speakers(group_rows(rows, session_id, 0.8))
        audit = {
            "consensus": consensus_audit,
            "assignment": assignment_audit,
        }
        payload = {
            "session_id": session_id,
            "final_test_inference": True,
            "uses_test_for_training": False,
            "uses_test_for_model_selection": False,
            "source_sha256": {
                name: sha256_file(path) for name, path in paths.items()
            },
            "audit": audit,
            "segments": segments,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_segments.extend(segments)
        audits[session_id] = audit
        print(
            json.dumps(
                {
                    "session": session_id,
                    "position": position,
                    "total": len(session_ids),
                    "segments": len(segments),
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
    metadata = {
        "experiment": "v84_moss_consensus_v20_tracks",
        "final_test_inference": True,
        "uses_test_for_training": False,
        "uses_test_for_model_selection": False,
        "session_ids": session_ids,
        "audits": audits,
        "audit": validation,
        "prediction_sha256": sha256_file(prediction_path),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate": str(prediction_path),
                "sha256": metadata["prediction_sha256"],
                **validation,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
