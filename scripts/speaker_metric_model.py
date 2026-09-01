"""Residual metric adapter for multiscale pretrained speaker embeddings."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpeakerMetricAdapter(nn.Module):
    """Adapt CAMPPlus geometry while retaining a pretrained-identity shortcut."""

    def __init__(
        self,
        scales: int = 3,
        embedding_dimension: int = 192,
        hidden_dimension: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.scales = scales
        self.embedding_dimension = embedding_dimension
        input_dimension = scales * embedding_dimension
        self.input_norm = nn.LayerNorm(input_dimension)
        self.residual = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, embedding_dimension),
        )
        self.residual_gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.logit_scale = nn.Parameter(torch.tensor(3.0))
        self.logit_bias = nn.Parameter(torch.tensor(-1.0))

    def embed(self, features: torch.Tensor) -> torch.Tensor:
        shape = features.shape
        if shape[-1] != self.scales * self.embedding_dimension:
            raise ValueError(f"Unexpected feature dimension: {shape}")
        multiscale = features.reshape(*shape[:-1], self.scales, self.embedding_dimension)
        multiscale = F.normalize(multiscale, dim=-1)
        pretrained = F.normalize(multiscale.mean(dim=-2), dim=-1)
        correction = self.residual(self.input_norm(features))
        gate = torch.sigmoid(self.residual_gate_logit)
        return F.normalize(pretrained + gate * correction, dim=-1)

    def pair_logits(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        cosine = (self.embed(left) * self.embed(right)).sum(dim=-1)
        return cosine * self.logit_scale.exp().clamp(max=30.0) + self.logit_bias

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.pair_logits(left, right)
