#!/usr/bin/env python3
"""Assign an independent cached ASR stream to frozen label-free speaker tracks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from run_diarizen_relabel import assign_tracks
from run_sortformer_relabel import group_rows, sha256_file, token_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--session-id")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fold = int(config["fold"] if args.fold is None else args.fold)
    validation_ids = (
        root / "data" / "splits" / f"fold_{fold}" / "val_sessions.txt"
    ).read_text().split()
    session_ids = validation_ids
    suffix = ""
    if args.session_id:
        if args.session_id not in validation_ids:
            raise RuntimeError("Session is outside the frozen validation fold")
        session_ids = [args.session_id]
        suffix = "_diagnostic"

    asr_dir = (
        root
        / "outputs"
        / config["asr_source_experiment"]
        / f"fold_{fold}{suffix}"
        / "sessions"
    )
    # Prefer a frozen full-fold track run.  A genuinely independent candidate
    # may also be cold-tested on one validation session before the expensive
    # full fold, in which case its explicitly diagnostic track cache is used.
    track_base = root / "outputs" / config["diarization_source_experiment"]
    track_dir = track_base / f"fold_{fold}" / "sessions"
    if args.session_id and not (track_dir / f"{args.session_id}.json").is_file():
        track_dir = track_base / f"fold_{fold}_diagnostic" / "sessions"
    output_dir = root / "outputs" / config["name"] / f"fold_{fold}{suffix}"
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    all_segments: list[dict] = []

    for session_id in session_ids:
        output_path = session_dir / f"{session_id}.json"
        if output_path.exists() and not args.overwrite:
            all_segments.extend(json.loads(output_path.read_text())["segments"])
            continue
        asr_path = asr_dir / f"{session_id}.json"
        track_path = track_dir / f"{session_id}.json"
        asr = json.loads(asr_path.read_text())
        track_payload = json.loads(track_path.read_text())
        rows = token_rows(asr["raw_result"][0], session_id)
        tracks = [
            [float(row["start_time"]), float(row["end_time"]), str(row["speaker"])]
            for row in track_payload["segments"]
        ]
        audit = assign_tracks(rows, tracks)
        segments = group_rows(
            rows, session_id, float(config["assignment"]["max_gap_seconds"])
        )
        payload = {
            "session_id": session_id,
            "development_only": True,
            "uses_validation_labels": False,
            "uses_test_data": False,
            "asr_source_sha256": sha256_file(asr_path),
            "speaker_track_source_sha256": sha256_file(track_path),
            "assignment_audit": audit,
            "segments": segments,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        all_segments.extend(segments)
        print(
            json.dumps(
                {
                    "session": session_id,
                    "segments": len(segments),
                    **audit,
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
        "development_only": True,
        "uses_validation_labels": False,
        "uses_test_data": False,
        "session_ids": session_ids,
        "prediction_sha256": sha256_file(prediction_path),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
