"""Invariant features for learned unmatched-speaker existence energy."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


FEATURE_NAMES = [
    "log_candidate_frames",
    "fragmentation_ratio",
    "temporal_span_ratio",
    "base_speaker_count_scaled",
    "cohesion_1s",
    "cohesion_2s",
    "cohesion_4s",
    "nearest_similarity_1s",
    "nearest_similarity_2s",
    "nearest_similarity_4s",
    "distinctness_1s",
    "distinctness_2s",
    "distinctness_4s",
]


def multiscale_embeddings(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] % 3:
        raise ValueError(f"Unexpected three-scale speaker feature shape: {features.shape}")
    dimension = features.shape[1] // 3
    return F.normalize(
        features.float().reshape(len(features), 3, dimension), dim=-1
    )


def energy_features(
    embeddings: torch.Tensor,
    candidate_mask: torch.Tensor,
    base_masks: list[torch.Tensor],
) -> torch.Tensor:
    indices = torch.where(candidate_mask)[0]
    base_masks = [mask for mask in base_masks if int(mask.sum())]
    if not len(indices) or not base_masks:
        raise ValueError("Existence energy requires candidate and base support frames")
    cohesion = []
    nearest = []
    for scale in range(3):
        candidate_values = embeddings[candidate_mask, scale]
        prototype = F.normalize(candidate_values.mean(dim=0), dim=0)
        cohesion.append(float((candidate_values @ prototype).mean()))
        nearest.append(
            max(
                float(
                    prototype
                    @ F.normalize(embeddings[mask, scale].mean(dim=0), dim=0)
                )
                for mask in base_masks
            )
        )
    fragments = 1 + int((indices[1:] != indices[:-1] + 1).sum())
    span = int(indices[-1] - indices[0] + 1)
    values = [
        math.log1p(len(indices)),
        fragments / len(indices),
        span / len(embeddings),
        len(base_masks) / 8.0,
        *cohesion,
        *nearest,
        *[left - right for left, right in zip(cohesion, nearest)],
    ]
    return torch.tensor(values, dtype=torch.float32)
