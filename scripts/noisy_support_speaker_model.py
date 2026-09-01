"""Swap-equivariant speaker ranking over potentially contaminated support sets."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def fixed_support(values: torch.Tensor, size: int) -> torch.Tensor:
    """Deterministically resize a non-empty support set without averaging it."""
    if values.ndim != 2 or not len(values):
        raise ValueError(f"Expected a non-empty [support, feature] tensor: {values.shape}")
    if len(values) >= size:
        indices = torch.linspace(0, len(values) - 1, size).round().long()
    else:
        indices = torch.arange(size) % len(values)
    return values[indices]


class NoisySupportSpeakerRanker(nn.Module):
    """Score a speaker candidate from a set of noisy anchor embeddings.

    The same candidate scorer is used for both roles. Swapping the current and
    alternative support sets therefore negates the final logit, preventing a
    branch-specific shortcut.
    """

    def __init__(
        self,
        input_dimension: int = 384,
        model_dimension: int = 192,
        attention_heads: int = 4,
        encoder_layers: int = 2,
        feedforward_dimension: int = 512,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dimension)
        self.input_projection = nn.Linear(input_dimension, model_dimension)
        self.set_token = nn.Parameter(torch.zeros(1, 1, model_dimension))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dimension,
            nhead=attention_heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.support_encoder = nn.TransformerEncoder(
            layer, num_layers=encoder_layers, norm=nn.LayerNorm(model_dimension)
        )
        pair_dimension = 4 * model_dimension
        self.pair_norm = nn.LayerNorm(pair_dimension)
        self.residual = nn.Sequential(
            nn.Linear(pair_dimension, model_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dimension, 1),
        )
        self.residual_gate_logit = nn.Parameter(torch.tensor(-1.5))
        self.cosine_scale_log = nn.Parameter(torch.tensor(2.0))
        nn.init.normal_(self.set_token, std=0.02)

    def project(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(
            self.input_projection(self.input_norm(values.float())), dim=-1
        )

    def candidate_score(
        self, query: torch.Tensor, support: torch.Tensor
    ) -> torch.Tensor:
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if support.ndim == 2:
            support = support.unsqueeze(0)
        query = self.project(query)
        support = self.project(support)
        token = self.set_token.expand(len(support), -1, -1)
        encoded = self.support_encoder(torch.cat([token, support], dim=1))
        robust = F.normalize(encoded[:, 0], dim=-1)
        center = F.normalize(support.mean(dim=1), dim=-1)
        pair = torch.cat(
            [
                torch.abs(query - robust),
                query * robust,
                torch.abs(query - center),
                query * center,
            ],
            dim=-1,
        )
        cosine = (query * robust).sum(dim=-1)
        residual = self.residual(self.pair_norm(pair)).squeeze(-1)
        return (
            cosine * self.cosine_scale_log.exp().clamp(max=30.0)
            + torch.sigmoid(self.residual_gate_logit) * residual
        )

    def forward(
        self,
        query: torch.Tensor,
        current_support: torch.Tensor,
        alternative_support: torch.Tensor,
    ) -> torch.Tensor:
        return self.candidate_score(
            query, alternative_support
        ) - self.candidate_score(query, current_support)
