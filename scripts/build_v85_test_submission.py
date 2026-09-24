#!/usr/bin/env python3
"""Build the frozen V85 ensemble submission from final-test inference caches."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from run_asr_majority_consensus import EPSILON, aligned_to_anchor, payload_tokens
from run_diarizen_relabel import assign_tracks
from run_sortformer_relabel import group_rows, sha256_file, token_rows
from run_v7_test_submission import validate_segments


def majority_on_aed(anchor_payload: dict, voters: list[dict], session_id: str) -> tuple[dict, dict]:
    anchor_result = anchor_payload["raw_result"][0]
    anchor_tokens = payload_tokens(anchor_payload)
    timestamps = anchor_result["timestamp"]
    if len(anchor_tokens) != len(timestamps):
        raise RuntimeError(f"AED token/timestamp mismatch for {session_id}")
    alignments = [aligned_to_anchor(anchor_tokens, payload_tokens(payload)) for payload in voters]
    consensus_tokens, consensus_timestamps = [], []
    substitutions = deletions = 0
    for index, anchor_token in enumerate(anchor_tokens):
        votes = [alignment[0][index] for alignment in alignments]
        counts = Counter(votes)
        top = max(counts.values())
        winners = [token for token, count in counts.items() if count == top]
        selected = anchor_token
        if top >= 2 and len(winners) == 1 and winners[0] != anchor_token:
            selected = winners[0]
            if selected == EPSILON:
                deletions += 1
            else:
                substitutions += 1
        if selected != EPSILON:
            consensus_tokens.append(selected)
            consensus_timestamps.append(timestamps[index])
    return {
        "text": " ".join(consensus_tokens),
        "timestamp": consensus_timestamps,
    }, {
        "anchor_tokens": len(anchor_tokens),
        "consensus_tokens": len(consensus_tokens),
        "substitutions": substitutions,
        "deletions": deletions,
        "voter_edit_distances": [row[1] for row in alignments],
    }


def substitutions_on_moss(moss_payload: dict, voters: list[dict]) -> tuple[list[dict], dict]:
    segments = sorted(
        moss_payload["segments"],
        key=lambda row: (float(row["start_time"]), float(row["end_time"]), str(row["speaker"])),
    )
    anchor_tokens, token_segments = [], []
    for segment_index, segment in enumerate(segments):
        tokens = str(segment["words"]).split()
        anchor_tokens.extend(tokens)
        token_segments.extend([segment_index] * len(tokens))
    alignments = [aligned_to_anchor(anchor_tokens, payload_tokens(payload)) for payload in voters]
    rebuilt = [[] for _ in segments]
    substitutions = 0
    for index, anchor_token in enumerate(anchor_tokens):
        votes = [alignment[0][index] for alignment in alignments]
        counts = Counter(votes)
        top = max(counts.values())
        winners = [token for token, count in counts.items() if count == top]
        selected = anchor_token
        if top >= 2 and len(winners) == 1 and winners[0] not in {anchor_token, EPSILON}:
            selected = winners[0]
            substitutions += 1
        rebuilt[token_segments[index]].append(selected)
    output = []
    for segment, tokens in zip(segments, rebuilt):
        row = dict(segment)
        row["words"] = " ".join(tokens)
        output.append(row)
    return output, {
        "anchor_tokens": len(anchor_tokens),
        "substitutions": substitutions,
        "voter_edit_distances": [row[1] for row in alignments],
    }


def canonicalize_speakers(segments: list[dict]) -> list[dict]:
    """Map arbitrary per-session speaker IDs to the required spk1, spk2, ... form."""
    speaker_map: dict[str, str] = {}
    output = []
    for segment in segments:
        row = dict(segment)
        raw_speaker = str(row["speaker"])
        if raw_speaker not in speaker_map:
            speaker_map[raw_speaker] = f"spk{len(speaker_map) + 1}"
        row["speaker"] = speaker_map[raw_speaker]
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("submissions/v85_moss_ensemble_submission.seglst.json"),
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    test_dir = (root / "data" / "test").resolve()
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
    }
    missing = [
        str(directory / f"{session_id}.json")
        for directory in sources.values()
        for session_id in session_ids
        if not (directory / f"{session_id}.json").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Final-test cache incomplete ({len(missing)} missing): {missing[:5]}")

    output_dir = root / "outputs/v85_deployable_joint_router/test"
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    output_segments = []
    decisions = {}
    started = time.time()
    for position, session_id in enumerate(session_ids, start=1):
        paths = {name: directory / f"{session_id}.json" for name, directory in sources.items()}
        payloads = {
            name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()
        }
        consensus_result, modular_text_audit = majority_on_aed(
            payloads["aed"], [payloads["qwen"], payloads["llm"], payloads["moss"]], session_id
        )
        rows = token_rows(consensus_result, session_id)
        turns = [
            [float(row["start_time"]), float(row["end_time"]), str(row["speaker"])]
            for row in payloads["tracks"]["segments"]
        ]
        assignment_audit = assign_tracks(rows, turns)
        modular_segments = group_rows(rows, session_id, 0.8)
        joint_segments, joint_text_audit = substitutions_on_moss(
            payloads["moss"], [payloads["aed"], payloads["qwen"], payloads["llm"]]
        )
        joint_speakers = len({row["speaker"] for row in joint_segments})
        modular_speakers = len({row["speaker"] for row in modular_segments})
        joint_tokens = sum(len(row["words"].split()) for row in joint_segments)
        modular_tokens = sum(len(row["words"].split()) for row in modular_segments)
        more_speakers = joint_speakers > modular_speakers
        simple_content_recovery = (
            joint_speakers == modular_speakers
            and joint_speakers <= 3
            and joint_tokens > modular_tokens
        )
        use_joint = more_speakers or simple_content_recovery
        selected_segments = canonicalize_speakers(
            joint_segments if use_joint else modular_segments
        )
        decision = {
            "chosen": "joint" if use_joint else "modular",
            "joint_speakers": joint_speakers,
            "modular_speakers": modular_speakers,
            "joint_tokens": joint_tokens,
            "modular_tokens": modular_tokens,
            "more_speakers": more_speakers,
            "simple_content_recovery": simple_content_recovery,
            "joint_text_audit": joint_text_audit,
            "modular_text_audit": modular_text_audit,
            "assignment_audit": assignment_audit,
        }
        session_payload = {
            "session_id": session_id,
            "final_test_inference": True,
            "uses_test_for_training": False,
            "uses_test_for_model_selection": False,
            "source_sha256": {name: sha256_file(path) for name, path in paths.items()},
            "decision": decision,
            "segments": selected_segments,
        }
        session_path = session_dir / f"{session_id}.json"
        session_path.write_text(
            json.dumps(session_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        output_segments.extend(selected_segments)
        decisions[session_id] = decision
        print(
            json.dumps(
                {
                    "session": session_id,
                    "position": position,
                    "total": len(session_ids),
                    "chosen": decision["chosen"],
                    "joint_speakers": joint_speakers,
                    "modular_speakers": modular_speakers,
                }
            ),
            flush=True,
        )

    audit = validate_segments(output_segments, wav_paths)
    prediction_path = output_dir / "hyp.seglst.json"
    prediction_path.write_text(
        json.dumps(output_segments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    submission_path = (root / args.submission).resolve()
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_bytes(prediction_path.read_bytes())
    metadata = {
        "final_test_inference": True,
        "uses_test_for_training": False,
        "uses_test_for_model_selection": False,
        "router_frozen_from_cv": True,
        "session_ids": session_ids,
        "decisions": decisions,
        "joint_sessions": [sid for sid, row in decisions.items() if row["chosen"] == "joint"],
        "audit": audit,
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
                "joint_sessions": len(metadata["joint_sessions"]),
                **audit,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
