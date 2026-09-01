"""Trainable session-level speaker graph and attractor network."""

from __future__ import annotations

import math

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F


def sinusoidal_positions(length: int, dimension: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dimension)
    )
    result = torch.zeros(length, dimension, device=device, dtype=torch.float32)
    result[:, 0::2] = torch.sin(position * divisor)
    result[:, 1::2] = torch.cos(position * divisor)
    return result


class SpeakerGraphAttractor(nn.Module):
    """Contextual frame encoder with session-conditioned speaker attractors."""

    def __init__(
        self,
        input_dimension: int,
        max_speakers: int = 6,
        model_dimension: int = 256,
        attention_heads: int = 8,
        encoder_layers: int = 4,
        decoder_layers: int = 2,
        feedforward_dimension: int = 768,
        dropout: float = 0.15,
        affinity_dimension: int = 128,
    ) -> None:
        super().__init__()
        self.max_speakers = max_speakers
        self.input_norm = nn.LayerNorm(input_dimension)
        self.input_projection = nn.Linear(input_dimension * 2, model_dimension)
        self.local_context = nn.Sequential(
            nn.Conv1d(model_dimension, model_dimension, 5, padding=2, groups=model_dimension),
            nn.GELU(),
            nn.Conv1d(model_dimension, model_dimension, 1),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dimension,
            nhead=attention_heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_layers, norm=nn.LayerNorm(model_dimension)
        )
        self.attractor_queries = nn.Parameter(
            torch.randn(max_speakers, model_dimension) / math.sqrt(model_dimension)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dimension,
            nhead=attention_heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attractor_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=decoder_layers, norm=nn.LayerNorm(model_dimension)
        )
        self.frame_projection = nn.Linear(model_dimension, model_dimension)
        self.slot_projection = nn.Linear(model_dimension, model_dimension)
        self.frame_activity_bias = nn.Linear(model_dimension, 1)
        self.existence_head = nn.Linear(model_dimension, 1)
        self.count_head = nn.Sequential(
            nn.Linear(model_dimension * 2, model_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dimension, max_speakers - 1),
        )
        self.affinity_projection = nn.Linear(model_dimension, affinity_dimension)
        self.affinity_log_scale = nn.Parameter(torch.tensor(math.log(8.0)))
        self.affinity_bias = nn.Parameter(torch.tensor(0.0))
        self.change_head = nn.Linear(model_dimension, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim == 2:
            features = features.unsqueeze(0)
        normalized = self.input_norm(features)
        delta = torch.zeros_like(normalized)
        delta[:, 1:] = normalized[:, 1:] - normalized[:, :-1]
        hidden = self.input_projection(torch.cat([normalized, delta], dim=-1))
        hidden = hidden + sinusoidal_positions(
            hidden.shape[1], hidden.shape[2], hidden.device
        ).to(hidden.dtype).unsqueeze(0)
        hidden = hidden + self.local_context(hidden.transpose(1, 2)).transpose(1, 2)
        encoded = self.encoder(hidden)
        queries = self.attractor_queries.unsqueeze(0).expand(encoded.shape[0], -1, -1)
        attractors = self.attractor_decoder(queries, encoded)
        frames = F.normalize(self.frame_projection(encoded), dim=-1)
        slots = F.normalize(self.slot_projection(attractors), dim=-1)
        activity_logits = (
            torch.einsum("btd,bkd->btk", frames, slots) * math.sqrt(frames.shape[-1])
            + self.frame_activity_bias(encoded)
        )
        pooled = torch.cat([encoded.mean(dim=1), encoded.amax(dim=1)], dim=-1)
        affinity = F.normalize(self.affinity_projection(encoded), dim=-1)
        return {
            "activity_logits": activity_logits,
            "existence_logits": self.existence_head(attractors).squeeze(-1),
            "count_logits": self.count_head(pooled),
            "affinity_embeddings": affinity,
            "change_logits": self.change_head(encoded).squeeze(-1),
            "affinity_scale": self.affinity_log_scale.exp().clamp(max=30.0),
            "affinity_bias": self.affinity_bias,
        }


def pit_assignment(activity_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return a max-speaker target matrix aligned to the predicted slot order."""
    if activity_logits.ndim != 2 or target.ndim != 2:
        raise ValueError("PIT expects [time, speaker] matrices")
    max_speakers = activity_logits.shape[1]
    if target.shape[1] > max_speakers:
        raise ValueError("Reference has more speakers than model capacity")
    padded = target.new_zeros((target.shape[0], max_speakers))
    padded[:, : target.shape[1]] = target
    cost = torch.empty(max_speakers, max_speakers, device=activity_logits.device)
    for predicted in range(max_speakers):
        expanded = activity_logits[:, predicted].unsqueeze(1).expand_as(padded)
        cost[predicted] = F.binary_cross_entropy_with_logits(
            expanded, padded, reduction="none"
        ).mean(dim=0)
    rows, columns = linear_sum_assignment(cost.detach().float().cpu().numpy())
    aligned = target.new_zeros((target.shape[0], max_speakers))
    for predicted, reference in zip(rows.tolist(), columns.tolist()):
        aligned[:, predicted] = padded[:, reference]
    return aligned


def speaker_graph_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    activity_logits = outputs["activity_logits"].squeeze(0)
    aligned = pit_assignment(activity_logits, target)
    existence_target = (aligned.amax(dim=0) > 0.05).float()
    speaker_count = int(target.shape[1])

    activity_loss = F.binary_cross_entropy_with_logits(activity_logits, aligned)
    existence_loss = F.binary_cross_entropy_with_logits(
        outputs["existence_logits"].squeeze(0), existence_target
    )
    count_loss = F.cross_entropy(
        outputs["count_logits"],
        torch.tensor([speaker_count - 2], device=target.device),
    )

    speech = target.sum(dim=1) > 0.05
    affinity_embeddings = outputs["affinity_embeddings"].squeeze(0)
    indices = torch.nonzero(speech, as_tuple=False).squeeze(1)
    if len(indices) > 72:
        positions = torch.linspace(0, len(indices) - 1, 72, device=target.device).long()
        indices = indices[positions]
    if len(indices) >= 2:
        selected_target = target[indices]
        same_speaker = (selected_target @ selected_target.T > 0.05).float()
        selected_embeddings = affinity_embeddings[indices]
        affinity_logits = (
            selected_embeddings @ selected_embeddings.T
        ) * outputs["affinity_scale"] + outputs["affinity_bias"]
        triangle = torch.triu(
            torch.ones_like(same_speaker, dtype=torch.bool), diagonal=1
        )
        affinity_loss = F.binary_cross_entropy_with_logits(
            affinity_logits[triangle], same_speaker[triangle]
        )
    else:
        affinity_loss = activity_loss.new_zeros(())

    dominant = target.argmax(dim=1)
    valid_change = speech.clone()
    valid_change[1:] &= speech[:-1]
    valid_change[0] = False
    change_target = torch.zeros_like(outputs["change_logits"].squeeze(0))
    change_target[1:] = (dominant[1:] != dominant[:-1]).float()
    change_logits = outputs["change_logits"].squeeze(0)
    change_loss = (
        F.binary_cross_entropy_with_logits(
            change_logits[valid_change], change_target[valid_change]
        )
        if valid_change.any()
        else activity_loss.new_zeros(())
    )
    losses = {
        "activity": activity_loss,
        "existence": existence_loss,
        "count": count_loss,
        "affinity": affinity_loss,
        "change": change_loss,
    }
    total = sum(float(weights[name]) * value for name, value in losses.items())
    return total, {name: float(value.detach()) for name, value in losses.items()}, aligned
