#!/usr/bin/env python3
"""Inject unmatched speakers from a complementary diarizer into frozen segments.

The base hypothesis keeps its words and time support.  A session is changed only
when the complementary partition contains more speakers than the base partition;
Hungarian matching preserves all matched base identities and only the unmatched
complementary identities may create new speakers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    midpoint = (left[0] + left[1]) / 2
    if right[0] <= midpoint <= right[1]:
        return 0.0
    return min(abs(midpoint - right[0]), abs(midpoint - right[1]))


def assign(interval: tuple[float, float], segments: list[dict]) -> str:
    return str(
        max(
            segments,
            key=lambda row: (
                overlap(
                    interval,
                    (float(row["start_time"]), float(row["end_time"])),
                ),
                -distance(
                    interval,
                    (float(row["start_time"]), float(row["end_time"])),
                ),
            ),
        )["speaker"]
    )


def token_rows(segments: list[dict], complementary: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for segment_index, segment in enumerate(segments):
        tokens = str(segment["words"]).split()
        if not tokens:
            continue
        start = float(segment["start_time"])
        end = float(segment["end_time"])
        step = max(0.0, end - start) / len(tokens)
        for token_index, token in enumerate(tokens):
            token_start = start + token_index * step
            token_end = end if token_index + 1 == len(tokens) else start + (token_index + 1) * step
            interval = (token_start, token_end)
            rows.append(
                {
                    "segment_index": segment_index,
                    "token": token,
                    "start": token_start,
                    "end": token_end,
                    "base": str(segment["speaker"]),
                    "complementary": assign(interval, complementary),
                }
            )
    return rows


def fuse_session(
    session_id: str, base_segments: list[dict], complementary: list[dict]
) -> tuple[list[dict], dict]:
    rows = token_rows(base_segments, complementary)
    base_speakers = sorted({str(row["speaker"]) for row in base_segments})
    complementary_speakers = sorted({str(row["speaker"]) for row in complementary})
    contingency = np.zeros(
        (len(complementary_speakers), len(base_speakers)), dtype=np.float64
    )
    for row in rows:
        left = complementary_speakers.index(row["complementary"])
        right = base_speakers.index(row["base"])
        contingency[left, right] += max(0.01, row["end"] - row["start"])
    left_indices, right_indices = linear_sum_assignment(-contingency)
    mapping = {
        complementary_speakers[left]: base_speakers[right]
        for left, right in zip(left_indices.tolist(), right_indices.tolist())
    }
    novel = (
        sorted(set(complementary_speakers) - set(mapping))
        if len(complementary_speakers) > len(base_speakers)
        else []
    )
    novel_map = {
        speaker: f"novel_{index}" for index, speaker in enumerate(novel, start=1)
    }
    if not novel:
        segments = [dict(row) for row in base_segments]
        changed_tokens = 0
    else:
        segments = []
        changed_tokens = 0
        rows_by_segment: dict[int, list[dict]] = {}
        for row in rows:
            final = novel_map.get(row["complementary"], row["base"])
            row["final"] = final
            changed_tokens += int(final != row["base"])
            rows_by_segment.setdefault(int(row["segment_index"]), []).append(row)
        for segment_index, original in enumerate(base_segments):
            segment_rows = rows_by_segment.get(segment_index, [])
            if not segment_rows or all(row["final"] == row["base"] for row in segment_rows):
                segments.append(dict(original))
                continue
            current: dict | None = None
            for row in segment_rows:
                if current is None or current["speaker"] != row["final"]:
                    current = {
                        "session_id": session_id,
                        "speaker": row["final"],
                        "start_time": row["start"],
                        "end_time": row["end"],
                        "tokens": [row["token"]],
                    }
                    segments.append(current)
                else:
                    current["end_time"] = row["end"]
                    current["tokens"].append(row["token"])
        speaker_order: dict[str, str] = {}
        normalized = []
        for row in sorted(segments, key=lambda item: (float(item["start_time"]), float(item["end_time"]))):
            raw_speaker = str(row["speaker"])
            speaker_order.setdefault(raw_speaker, f"spk{len(speaker_order) + 1}")
            words = row.get("words")
            if words is None:
                words = " ".join(row.pop("tokens"))
            normalized.append(
                {
                    "session_id": session_id,
                    "speaker": speaker_order[raw_speaker],
                    "start_time": round(float(row["start_time"]), 2),
                    "end_time": round(float(row["end_time"]), 2),
                    "words": words,
                }
            )
        segments = normalized
    audit = {
        "base_speaker_count": len(base_speakers),
        "complementary_speaker_count": len(complementary_speakers),
        "novel_speaker_count": len(novel),
        "novel_complementary_speakers": novel,
        "changed_tokens": changed_tokens,
        "base_segments": len(base_segments),
        "fused_segments": len(segments),
    }
    return segments, audit


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
    base_dir = root / "outputs" / config["base_experiment"] / f"fold_{fold}" / "sessions"
    complementary_dir = (
        root
        / "outputs"
        / config["complementary_experiment"]
        / f"fold_{fold}"
        / "sessions"
    )
    output_dir = root / "outputs" / config["name"] / f"fold_{fold}"
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    all_segments: list[dict] = []
    audits: dict[str, dict] = {}
    started = time.time()
    for session_id in session_ids:
        output_path = session_dir / f"{session_id}.json"
        if output_path.exists() and not args.overwrite:
            payload = json.loads(output_path.read_text())
            all_segments.extend(payload["segments"])
            audits[session_id] = payload["fusion_audit"]
            continue
        base_path = base_dir / f"{session_id}.json"
        complementary_path = complementary_dir / f"{session_id}.json"
        base_segments = json.loads(base_path.read_text())["segments"]
        complementary = json.loads(complementary_path.read_text())["segments"]
        segments, audit = fuse_session(session_id, base_segments, complementary)
        payload = {
            "session_id": session_id,
            "development_only": True,
            "uses_validation_labels": False,
            "uses_test_data": False,
            "base_source_sha256": sha256_file(base_path),
            "complementary_source_sha256": sha256_file(complementary_path),
            "fusion_audit": audit,
            "segments": segments,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        all_segments.extend(segments)
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
        "fusion_audits": audits,
        "config_sha256": sha256_file(config_path),
        "prediction_sha256": sha256_file(prediction_path),
        "elapsed_seconds": round(time.time() - started, 2),
        "argv": sys.argv,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
