"""Label-free conversation-domain features for novel-speaker verification."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from diagnose_partition_consensus_geometry import rows_from_payload


DOMAIN_NAMES = [
    "predicted_speaker_count",
    "predicted_segment_count",
    "speaker_switch_fraction",
    "short_segment_fraction",
    "overlap_fraction",
    "tokens_per_speech_second",
    "embedding_adjacent_cosine_s0",
    "embedding_dispersion_s0",
    "embedding_adjacent_cosine_s1",
    "embedding_dispersion_s1",
    "embedding_adjacent_cosine_s2",
    "embedding_dispersion_s2",
]


def conversation_domain(prediction_path: Path, feature_path: Path) -> torch.Tensor:
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    segments = sorted(
        rows_from_payload(prediction),
        key=lambda row: (float(row["start_time"]), float(row["end_time"])),
    )
    acoustic = torch.load(feature_path, map_location="cpu", weights_only=True)
    if acoustic.get("uses_speaker_labels") is not False:
        raise RuntimeError(f"Domain feature is not label-free: {feature_path}")
    duration = max(float(acoustic["duration"]), 0.01)
    segment_durations = torch.tensor(
        [max(0.0, float(row["end_time"]) - float(row["start_time"])) for row in segments]
    )
    switches = [
        str(left["speaker"]) != str(right["speaker"])
        for left, right in zip(segments, segments[1:])
    ]
    events = []
    for row in segments:
        events.extend([(float(row["start_time"]), 1), (float(row["end_time"]), -1)])
    active = 0
    previous = speech = overlap = 0.0
    for timestamp, delta in sorted(events, key=lambda item: (item[0], item[1])):
        span = max(0.0, timestamp - previous)
        speech += span if active else 0.0
        overlap += span if active >= 2 else 0.0
        active += delta
        previous = timestamp
    result = [
        len({str(row["speaker"]) for row in segments}) / 6.0,
        len(segments) / 30.0,
        sum(switches) / max(len(switches), 1),
        float((segment_durations <= 0.8).float().mean()),
        overlap / duration,
        sum(len(str(row["words"]).split()) for row in segments) / max(speech, 0.01) / 6.0,
    ]
    features = F.normalize(acoustic["features"].float().reshape(len(acoustic["features"]), 3, -1), dim=-1)
    for scale in range(3):
        values = features[:, scale]
        adjacent = float((values[1:] * values[:-1]).sum(-1).mean()) if len(values) > 1 else 1.0
        prototype = F.normalize(values.mean(0), dim=0)
        dispersion = float(1.0 - (values @ prototype).mean())
        result.extend([adjacent, dispersion])
    return torch.tensor(result, dtype=torch.float32)


def domain_feature_names() -> list[str]:
    return [
        *[f"session_{name}" for name in DOMAIN_NAMES],
        *[f"target_{name}" for name in DOMAIN_NAMES],
        *[f"session_minus_target_{name}" for name in DOMAIN_NAMES],
    ]
