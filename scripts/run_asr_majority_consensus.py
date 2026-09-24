#!/usr/bin/env python3
"""Build a conservative token consensus on an independently timestamped anchor."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

from run_fireredasr2_llm import edit_alignment
from run_sortformer_relabel import normalize_tokens


EPSILON = "<eps>"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aligned_to_anchor(anchor: list[str], candidate: list[str]) -> tuple[list[str], int, int]:
    mapping, distance = edit_alignment(anchor, candidate)
    aligned = [EPSILON] * len(anchor)
    ignored_insertions = 0
    for token, anchor_index in zip(candidate, mapping):
        if anchor_index is None:
            ignored_insertions += 1
        else:
            aligned[anchor_index] = token
    return aligned, distance, ignored_insertions


def payload_tokens(payload: dict) -> list[str]:
    """Read either a timestamped ASR cache or a joint-model SegLST cache."""
    if payload.get("raw_result"):
        return normalize_tokens(str(payload["raw_result"][0]["text"]))
    segments = payload.get("segments")
    if isinstance(segments, list) and segments:
        ordered = sorted(
            segments,
            key=lambda row: (
                float(row["start_time"]),
                float(row["end_time"]),
                str(row["speaker"]),
            ),
        )
        return [token for row in ordered for token in str(row["words"]).split()]
    raise RuntimeError("Voter cache contains neither raw_result nor speaker segments")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    fold = int(config["fold"] if args.fold is None else args.fold)
    session_ids = (root / "data" / "splits" / f"fold_{fold}" / "val_sessions.txt").read_text().split()
    anchor_dir = root / "outputs" / config["anchor_experiment"] / f"fold_{fold}" / "sessions"
    voter_dirs = [root / "outputs" / name / f"fold_{fold}" / "sessions" for name in config["voter_experiments"]]
    if len(voter_dirs) < 2:
        raise RuntimeError("Conservative consensus requires at least two independent voters")
    min_voter_agreement = int(config.get("min_voter_agreement", len(voter_dirs)))
    if not 2 <= min_voter_agreement <= len(voter_dirs):
        raise RuntimeError(
            "min_voter_agreement must be between two and the number of voters"
        )
    output_dir = root / "outputs" / config["name"] / f"fold_{fold}"
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    all_audits = {}
    started = time.time()
    for session_id in session_ids:
        output_path = session_dir / f"{session_id}.json"
        if output_path.exists() and not args.overwrite:
            all_audits[session_id] = json.loads(output_path.read_text())["consensus_audit"]
            continue
        anchor_path = anchor_dir / f"{session_id}.json"
        voter_paths = [directory / f"{session_id}.json" for directory in voter_dirs]
        anchor_payload = json.loads(anchor_path.read_text())
        anchor_result = anchor_payload["raw_result"][0]
        anchor_tokens = normalize_tokens(str(anchor_result["text"]))
        timestamps = anchor_result.get("timestamp")
        if not isinstance(timestamps, list) or len(anchor_tokens) != len(timestamps):
            raise RuntimeError(f"Invalid anchor timestamps for {session_id}")
        voter_tokens = [payload_tokens(json.loads(path.read_text())) for path in voter_paths]
        alignments = [aligned_to_anchor(anchor_tokens, tokens) for tokens in voter_tokens]
        consensus_tokens, consensus_timestamps = [], []
        substitutions = deletions = agreements = ties = 0
        for index, anchor_token in enumerate(anchor_tokens):
            votes = [alignment[0][index] for alignment in alignments]
            counts = Counter(votes)
            top_count = max(counts.values())
            winners = [token for token, count in counts.items() if count == top_count]
            selected = anchor_token
            if top_count >= min_voter_agreement and len(winners) == 1:
                agreements += 1
                winner = winners[0]
                if winner != anchor_token:
                    selected = winner
                    if selected == EPSILON:
                        deletions += 1
                    else:
                        substitutions += 1
            elif top_count >= min_voter_agreement:
                ties += 1
            if selected != EPSILON:
                consensus_tokens.append(selected)
                consensus_timestamps.append(timestamps[index])
        audit = {
            "anchor_tokens": len(anchor_tokens), "consensus_tokens": len(consensus_tokens),
            "voter_count": len(voter_dirs),
            "min_voter_agreement": min_voter_agreement,
            "voter_agreements": agreements, "voter_ties": ties,
            "consensus_substitutions": substitutions,
            "consensus_deletions": deletions,
            "voter_edit_distances": [row[1] for row in alignments],
            "ignored_voter_insertions": [row[2] for row in alignments],
        }
        payload = {
            "session_id": session_id, "development_only": True,
            "uses_validation_labels": False, "uses_test_data": False,
            "anchor_source_sha256": sha256_file(anchor_path),
            "voter_source_sha256": [sha256_file(path) for path in voter_paths],
            "consensus_audit": audit,
            "raw_result": [{"text": " ".join(consensus_tokens), "timestamp": consensus_timestamps}],
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        all_audits[session_id] = audit
        print(json.dumps({"session": session_id, **audit}), flush=True)
    metadata = {
        "experiment": config["name"], "fold": fold, "development_only": True,
        "uses_validation_labels": False, "uses_test_data": False,
        "session_ids": session_ids, "consensus_audits": all_audits,
        "config_sha256": sha256_file(config_path), "elapsed_seconds": round(time.time() - started, 2),
        "argv": sys.argv,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
