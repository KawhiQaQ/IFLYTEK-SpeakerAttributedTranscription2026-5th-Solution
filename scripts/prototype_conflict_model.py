"""Swap-equivariant acoustic arbitration between two speaker prototypes."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PrototypeConflictHead(nn.Module):
    """Score candidate prototypes with one shared pair network.

    Swapping the current and alternative prototypes negates the output logit,
    so the model cannot learn branch-specific identity shortcuts.
    """

    def __init__(
        self,
        embedding_dimension: int = 192,
        hidden_dimension: int = 96,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        pair_dimension = 2 * embedding_dimension
        self.pair_norm = nn.LayerNorm(pair_dimension)
        self.residual = nn.Sequential(
            nn.Linear(pair_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )
        self.residual_gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.cosine_scale_log = nn.Parameter(torch.tensor(2.0))
        self.support_scale = nn.Parameter(torch.tensor(0.0))

    def candidate_score(
        self,
        node: torch.Tensor,
        prototype: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        node = F.normalize(node, dim=-1)
        prototype = F.normalize(prototype, dim=-1)
        pair = torch.cat([torch.abs(node - prototype), node * prototype], dim=-1)
        residual = self.residual(self.pair_norm(pair)).squeeze(-1)
        cosine = (node * prototype).sum(dim=-1)
        return (
            cosine * self.cosine_scale_log.exp().clamp(max=30.0)
            + torch.sigmoid(self.residual_gate_logit) * residual
            + self.support_scale * torch.log1p(support.float())
        )

    def forward(
        self,
        node: torch.Tensor,
        current: torch.Tensor,
        alternative: torch.Tensor,
        current_support: torch.Tensor,
        alternative_support: torch.Tensor,
    ) -> torch.Tensor:
        return self.candidate_score(
            node, alternative, alternative_support
        ) - self.candidate_score(node, current, current_support)
