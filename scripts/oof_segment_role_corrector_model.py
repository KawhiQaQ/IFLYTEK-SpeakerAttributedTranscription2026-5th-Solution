"""Speaker-query Transformer for conservative segment role correction."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class OOFSegmentRoleCorrector(nn.Module):
    def __init__(
        self,
        input_size: int = 768,
        hidden_size: int = 192,
        timing_size: int = 5,
        attention_heads: int = 6,
        transformer_layers: int = 2,
        dropout: float = 0.15,
        base_prior: float = 3.0,
    ) -> None:
        super().__init__()
        self.acoustic = nn.Sequential(
            nn.LayerNorm(input_size), nn.Linear(input_size, hidden_size), nn.GELU()
        )
        self.prototype = nn.Sequential(
            nn.LayerNorm(input_size), nn.Linear(input_size, hidden_size), nn.GELU()
        )
        self.timing = nn.Sequential(
            nn.LayerNorm(timing_size), nn.Linear(timing_size, hidden_size), nn.GELU()
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, transformer_layers)
        self.pair = nn.Sequential(
            nn.LayerNorm(4 * hidden_size + 2),
            nn.Linear(4 * hidden_size + 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        # Start exactly as a strong identity prior. Training must earn every edit.
        nn.init.zeros_(self.pair[-1].weight)
        nn.init.zeros_(self.pair[-1].bias)
        self.base_prior = nn.Parameter(torch.tensor(float(base_prior)))

    def forward(
        self,
        segments: torch.Tensor,
        prototypes: torch.Tensor,
        timing: torch.Tensor,
        base_labels: torch.Tensor,
    ) -> torch.Tensor:
        local = self.acoustic(segments)
        hidden = local + self.timing(timing)
        length = hidden.shape[0]
        position = torch.arange(length, device=hidden.device, dtype=hidden.dtype).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, hidden.shape[1], 2, device=hidden.device, dtype=hidden.dtype)
            * (-math.log(10000.0) / hidden.shape[1])
        )
        positional = torch.zeros_like(hidden)
        positional[:, 0::2] = torch.sin(position * divisor)
        positional[:, 1::2] = torch.cos(position * divisor)
        contextual = hidden + self.context((hidden + positional).unsqueeze(0))[0]
        queries = self.prototype(prototypes)

        segment_rows = contextual[:, None, :].expand(-1, len(queries), -1)
        query_rows = queries[None, :, :].expand(len(contextual), -1, -1)
        cosine = F.cosine_similarity(
            segments[:, None, :], prototypes[None, :, :], dim=-1
        ).unsqueeze(-1)
        current = F.one_hot(base_labels, num_classes=len(queries)).float().unsqueeze(-1)
        pair = torch.cat(
            [
                segment_rows,
                query_rows,
                segment_rows * query_rows,
                (segment_rows - query_rows).abs(),
                cosine,
                current,
            ],
            dim=-1,
        )
        residual = self.pair(pair).squeeze(-1)
        prior = self.base_prior.clamp(0.5, 6.0) * current.squeeze(-1)
        return residual + prior


class DualEncoderLOORoleCorrector(nn.Module):
    """Contextual role decoder over complementary, leave-one-out speaker spaces.

    Prototypes have shape ``[segments, speakers, embedding]``.  In particular,
    the prototype of a segment's current cluster excludes that segment, which
    removes the self-inclusion shortcut present in ordinary cluster scoring.
    """

    def __init__(
        self,
        primary_input_size: int = 576,
        secondary_input_size: int = 768,
        hidden_size: int = 192,
        timing_size: int = 5,
        attention_heads: int = 6,
        transformer_layers: int = 2,
        dropout: float = 0.15,
        base_prior: float = 3.0,
    ) -> None:
        super().__init__()
        self.primary_acoustic = nn.Sequential(
            nn.LayerNorm(primary_input_size), nn.Linear(primary_input_size, hidden_size), nn.GELU()
        )
        self.secondary_acoustic = nn.Sequential(
            nn.LayerNorm(secondary_input_size), nn.Linear(secondary_input_size, hidden_size), nn.GELU()
        )
        self.primary_prototype = nn.Sequential(
            nn.LayerNorm(primary_input_size), nn.Linear(primary_input_size, hidden_size), nn.GELU()
        )
        self.secondary_prototype = nn.Sequential(
            nn.LayerNorm(secondary_input_size), nn.Linear(secondary_input_size, hidden_size), nn.GELU()
        )
        self.acoustic_gate = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.Sigmoid())
        self.prototype_gate = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.Sigmoid())
        self.timing = nn.Sequential(
            nn.LayerNorm(timing_size), nn.Linear(timing_size, hidden_size), nn.GELU()
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, transformer_layers)
        self.pair = nn.Sequential(
            nn.LayerNorm(4 * hidden_size + 3),
            nn.Linear(4 * hidden_size + 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        nn.init.zeros_(self.pair[-1].weight)
        nn.init.zeros_(self.pair[-1].bias)
        self.base_prior = nn.Parameter(torch.tensor(float(base_prior)))

    @staticmethod
    def _fuse(primary: torch.Tensor, secondary: torch.Tensor, gate: nn.Module) -> torch.Tensor:
        weight = gate(torch.cat([primary, secondary], dim=-1))
        return weight * primary + (1.0 - weight) * secondary

    def forward(
        self,
        primary_segments: torch.Tensor,
        primary_prototypes: torch.Tensor,
        secondary_segments: torch.Tensor,
        secondary_prototypes: torch.Tensor,
        timing: torch.Tensor,
        base_labels: torch.Tensor,
    ) -> torch.Tensor:
        primary_local = self.primary_acoustic(primary_segments)
        secondary_local = self.secondary_acoustic(secondary_segments)
        local = self._fuse(primary_local, secondary_local, self.acoustic_gate)

        hidden = local + self.timing(timing)
        length = hidden.shape[0]
        position = torch.arange(length, device=hidden.device, dtype=hidden.dtype).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, hidden.shape[1], 2, device=hidden.device, dtype=hidden.dtype)
            * (-math.log(10000.0) / hidden.shape[1])
        )
        positional = torch.zeros_like(hidden)
        positional[:, 0::2] = torch.sin(position * divisor)
        positional[:, 1::2] = torch.cos(position * divisor)
        contextual = hidden + self.context((hidden + positional).unsqueeze(0))[0]

        primary_queries = self.primary_prototype(primary_prototypes)
        secondary_queries = self.secondary_prototype(secondary_prototypes)
        queries = self._fuse(primary_queries, secondary_queries, self.prototype_gate)
        segment_rows = contextual[:, None, :].expand_as(queries)
        primary_cosine = F.cosine_similarity(
            primary_segments[:, None, :], primary_prototypes, dim=-1
        ).unsqueeze(-1)
        secondary_cosine = F.cosine_similarity(
            secondary_segments[:, None, :], secondary_prototypes, dim=-1
        ).unsqueeze(-1)
        current = F.one_hot(base_labels, num_classes=queries.shape[1]).float().unsqueeze(-1)
        pair = torch.cat(
            [
                segment_rows,
                queries,
                segment_rows * queries,
                (segment_rows - queries).abs(),
                primary_cosine,
                secondary_cosine,
                current,
            ],
            dim=-1,
        )
        residual = self.pair(pair).squeeze(-1)
        prior = self.base_prior.clamp(0.5, 6.0) * current.squeeze(-1)
        return residual + prior
