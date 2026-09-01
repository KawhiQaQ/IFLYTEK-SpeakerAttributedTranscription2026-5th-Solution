"""Shared data utilities for residual speaker-affinity training and inference."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from diagnose_partition_consensus_geometry import map_partition, ordered, rows_from_payload
from train_oof_segment_role_corrector import source_segments
from train_sorted_slot_diarizer import feature_path, sequence_features


def load_base_rows(path: Path) -> list[dict]:
    return ordered(source_segments(json.loads(path.read_text(encoding="utf-8"))))


def load_candidate_labels(
    root: Path,
    fold: int,
    session_id: str,
    base: list[dict],
    candidates: dict[str, str],
    base_name: str,
) -> tuple[list[str], torch.Tensor]:
    partitions = []
    for name, experiment in candidates.items():
        path = root / f"outputs/{experiment}/fold_{fold}/sessions/{session_id}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = rows_from_payload(payload)
        else:
            path = root / f"outputs/{experiment}/fold_{fold}/hyp.seglst.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise RuntimeError(f"Expected fold-level SegLST list: {path}")
            rows = [
                row
                for row in payload
                if str(row.get("session_id")) == str(session_id)
            ]
            if not rows:
                raise RuntimeError(
                    f"No candidate rows for session {session_id}: {path}"
                )
        labels = (
            [str(row["speaker"]) for row in base]
            if name == base_name
            else map_partition(base, rows)
        )
        partitions.append(labels)
    base_labels = partitions[list(candidates).index(base_name)]
    relations = torch.stack(
        [
            torch.tensor(
                [[left == right for right in labels] for left in labels],
                dtype=torch.float32,
            )
            for labels in partitions
        ],
        dim=-1,
    )
    return base_labels, relations


def load_encoder_values(
    root: Path,
    feature_templates: dict[str, str],
    fold: int,
    subset: str,
    session_id: str,
    segments: list[dict],
) -> tuple[list[torch.Tensor], torch.Tensor, list[Path]]:
    paths = [
        feature_path(root, template, fold, subset, session_id)
        for template in feature_templates.values()
    ]
    payloads = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    frame_payloads = [
        payload for payload in payloads if payload.get("feature_level") != "segment"
    ]
    if not frame_payloads:
        raise RuntimeError("At least one frame-level encoder is required for timing")
    frame_values, timing = sequence_features(segments, frame_payloads)
    frame_iterator = iter(frame_values)
    values = []
    expected_keys = [
        [float(row["start_time"]), float(row["end_time"]), str(row["words"])]
        for row in segments
    ]
    for payload in payloads:
        if payload.get("feature_level") == "segment":
            if (
                payload.get("uses_speaker_labels") is not False
                or payload.get("uses_test_data") is not False
                or payload.get("segment_keys") != expected_keys
            ):
                raise RuntimeError("Segment-level feature provenance or alignment failed")
            values.append(F.normalize(payload["features"].float(), dim=-1))
        else:
            values.append(next(frame_iterator))
    return values, timing, paths


def domain_descriptor(
    segments: list[dict], encoder_values: list[torch.Tensor], duration: float
) -> torch.Tensor:
    count = len(segments)
    segment_duration = torch.tensor(
        [max(0.0, float(row["end_time"]) - float(row["start_time"])) for row in segments]
    )
    switches = [
        str(left["speaker"]) != str(right["speaker"])
        for left, right in zip(segments, segments[1:])
    ]
    events = []
    for row in segments:
        events.extend(
            [(float(row["start_time"]), 1), (float(row["end_time"]), -1)]
        )
    active = 0
    previous = 0.0
    speech = overlap = 0.0
    for timestamp, delta in sorted(events, key=lambda item: (item[0], item[1])):
        span = max(0.0, timestamp - previous)
        speech += span if active else 0.0
        overlap += span if active >= 2 else 0.0
        active += delta
        previous = timestamp
    result = [
        len({str(row["speaker"]) for row in segments}) / 6.0,
        count / 30.0,
        sum(switches) / max(len(switches), 1),
        float((segment_duration <= 0.8).float().mean()),
        overlap / max(duration, 0.01),
        sum(len(str(row["words"]).split()) for row in segments) / max(speech, 0.01) / 6.0,
    ]
    for values in encoder_values:
        values = F.normalize(values.float(), dim=-1)
        adjacent = (
            float((values[1:] * values[:-1]).sum(-1).mean()) if len(values) > 1 else 1.0
        )
        prototype = F.normalize(values.mean(0), dim=0)
        dispersion = float(1.0 - (values @ prototype).mean())
        result.extend([adjacent, dispersion])
    return torch.tensor(result, dtype=torch.float32)


def aggregate_domain(examples: list[dict]) -> torch.Tensor:
    if not examples:
        raise RuntimeError("No examples for target-domain aggregation")
    return torch.stack([example["domain"] for example in examples]).mean(0)


def base_same_matrix(labels: list[str]) -> torch.Tensor:
    return torch.tensor(
        [[left == right for right in labels] for left in labels], dtype=torch.float32
    )


def canonicalize(labels: list[int]) -> list[int]:
    mapping: dict[int, int] = {}
    return [mapping.setdefault(label, len(mapping)) for label in labels]
