#!/usr/bin/env python3
"""Create the final V20 submission from two frozen test candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

from run_diarization_quality_router import (
    coherence_score,
    pure_turns,
    sha256_file,
)
from run_v7_test_submission import validate_segments


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
    suffix = "_smoke" if args.max_sessions is not None else ""
    streaming_dir = root / "outputs" / config["streaming_experiment"] / f"test{suffix}"
    offline_dir = root / "outputs" / config["offline_experiment"] / f"test{suffix}"
    if not all((streaming_dir / "sessions" / f"{sid}.json").is_file() for sid in session_ids):
        raise FileNotFoundError("Frozen streaming test candidate is incomplete")
    if not all((offline_dir / "sessions" / f"{sid}.json").is_file() for sid in session_ids):
        raise FileNotFoundError("Frozen offline test candidate is incomplete")

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
    output_dir = root / "outputs" / config["name"] / f"test{suffix}"
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    all_segments: list[dict] = []
    decisions: dict[str, dict] = {}
    started = time.time()
    for position, (session_id, wav_path) in enumerate(zip(session_ids, wav_paths), start=1):
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
        offline_diarization = offline.get("raw_diarization")
        if (
            streaming["route"] == "3dspeaker_fallback"
            or offline["route"] == "3dspeaker_fallback"
            or not streaming_diarization
            or not offline_diarization
        ):
            selected = "streaming"
            streaming_quality = offline_quality = quality_margin = None
            reason = "capacity_fallback_or_missing_diarization"
        else:
            streaming_turns = pure_turns(streaming_diarization, min_pure_seconds)
            offline_turns = pure_turns(offline_diarization, min_pure_seconds)
            windows = [[row[0], row[1]] for row in streaming_turns + offline_turns]
            wav_data = load_audio(str(wav_path), None, diarizer.fs)
            embeddings = diarizer.do_emb_extraction(windows, wav_data)
            split = len(streaming_turns)
            streaming_quality = coherence_score(embeddings[:split], streaming_turns)
            offline_quality = coherence_score(embeddings[split:], offline_turns)
            quality_margin = offline_quality["score"] - streaming_quality["score"]
            selected = "offline" if quality_margin > min_quality_margin else "streaming"
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
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        decisions[session_id] = decision
        all_segments.extend(payload["segments"])
        print(json.dumps({"session": session_id, "position": position, "total": len(session_ids), "selected": selected, "quality_margin": quality_margin}), flush=True)

    audit = validate_segments(all_segments, wav_paths)
    prediction_path = output_dir / "hyp.seglst.json"
    prediction_path.write_text(json.dumps(all_segments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    submission_path = root / config["submission_path"]
    if args.max_sessions is None:
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        submission_path.write_bytes(prediction_path.read_bytes())
    metadata = {
        "final_test_inference": True,
        "uses_test_for_training": False,
        "uses_test_for_model_selection": False,
        "router_frozen_from_cv": True,
        "smoke_only": args.max_sessions is not None,
        "session_ids": session_ids,
        "decisions": decisions,
        "offline_sessions": [sid for sid, row in decisions.items() if row["selected"] == "offline"],
        "audit": audit,
        "config": config,
        "config_sha256": sha256_file(config_path),
        "streaming_prediction_sha256": sha256_file(streaming_dir / "hyp.seglst.json"),
        "offline_prediction_sha256": sha256_file(offline_dir / "hyp.seglst.json"),
        "prediction_sha256": sha256_file(prediction_path),
        "elapsed_seconds": round(time.time() - started, 2),
        "argv": sys.argv,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prediction": str(prediction_path), "submission": str(submission_path) if args.max_sessions is None else None, "sha256": metadata["prediction_sha256"], **audit, "offline_sessions": len(metadata["offline_sessions"])}), flush=True)


if __name__ == "__main__":
    main()
