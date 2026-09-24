#!/usr/bin/env python3
"""Conservatively correct joint-model segment text without changing its tracks."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import yaml

from run_asr_majority_consensus import EPSILON, aligned_to_anchor, payload_tokens
from run_sortformer_relabel import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fold = int(config["fold"] if args.fold is None else args.fold)
    session_ids = (
        root / "data" / "splits" / f"fold_{fold}" / "val_sessions.txt"
    ).read_text().split()
    anchor_dir = (
        root / "outputs" / config["segment_anchor_experiment"] / f"fold_{fold}" / "sessions"
    )
    voter_dirs = [
        root / "outputs" / name / f"fold_{fold}" / "sessions"
        for name in config["voter_experiments"]
    ]
    min_agreement = int(config["min_voter_agreement"])
    allow_deletions = bool(config.get("allow_deletions", False))
    if not 2 <= min_agreement <= len(voter_dirs):
        raise RuntimeError("Invalid voter agreement threshold")

    output_dir = root / "outputs" / config["name"] / f"fold_{fold}"
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    all_segments: list[dict] = []
    audits = {}
    started = time.time()
    for session_id in session_ids:
        output_path = session_dir / f"{session_id}.json"
        if output_path.exists() and not args.overwrite:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            all_segments.extend(payload["segments"])
            audits[session_id] = payload["consensus_audit"]
            continue
        anchor_path = anchor_dir / f"{session_id}.json"
        voter_paths = [directory / f"{session_id}.json" for directory in voter_dirs]
        anchor_payload = json.loads(anchor_path.read_text(encoding="utf-8"))
        segments = sorted(
            anchor_payload["segments"],
            key=lambda row: (
                float(row["start_time"]),
                float(row["end_time"]),
                str(row["speaker"]),
            ),
        )
        anchor_tokens: list[str] = []
        token_segments: list[int] = []
        for segment_index, segment in enumerate(segments):
            tokens = str(segment["words"]).split()
            anchor_tokens.extend(tokens)
            token_segments.extend([segment_index] * len(tokens))
        voters = [payload_tokens(json.loads(path.read_text())) for path in voter_paths]
        alignments = [aligned_to_anchor(anchor_tokens, tokens) for tokens in voters]
        rebuilt = [[] for _ in segments]
        substitutions = deletions = agreements = ties = 0
        for index, anchor_token in enumerate(anchor_tokens):
            votes = [alignment[0][index] for alignment in alignments]
            counts = Counter(votes)
            top_count = max(counts.values())
            winners = [token for token, count in counts.items() if count == top_count]
            selected = anchor_token
            if top_count >= min_agreement and len(winners) == 1:
                agreements += 1
                if winners[0] != anchor_token:
                    if winners[0] == EPSILON:
                        if allow_deletions:
                            selected = EPSILON
                            deletions += 1
                    else:
                        selected = winners[0]
                        substitutions += 1
            elif top_count >= min_agreement:
                ties += 1
            if selected != EPSILON:
                rebuilt[token_segments[index]].append(selected)
        output_segments = []
        for segment, tokens in zip(segments, rebuilt):
            if not tokens:
                continue
            row = dict(segment)
            row["words"] = " ".join(tokens)
            output_segments.append(row)
        audit = {
            "anchor_tokens": len(anchor_tokens),
            "consensus_tokens": sum(len(tokens) for tokens in rebuilt),
            "voter_count": len(voter_dirs),
            "min_voter_agreement": min_agreement,
            "allow_deletions": allow_deletions,
            "voter_agreements": agreements,
            "voter_ties": ties,
            "consensus_substitutions": substitutions,
            "consensus_deletions": deletions,
            "voter_edit_distances": [row[1] for row in alignments],
            "ignored_voter_insertions": [row[2] for row in alignments],
        }
        payload = {
            "session_id": session_id,
            "development_only": True,
            "uses_validation_labels": False,
            "uses_test_data": False,
            "segment_anchor_sha256": sha256_file(anchor_path),
            "voter_source_sha256": [sha256_file(path) for path in voter_paths],
            "consensus_audit": audit,
            "segments": output_segments,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        all_segments.extend(output_segments)
        audits[session_id] = audit
        print(json.dumps({"session": session_id, **audit}), flush=True)

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
        "audits": audits,
        "config_sha256": sha256_file(config_path),
        "prediction_sha256": sha256_file(prediction_path),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
