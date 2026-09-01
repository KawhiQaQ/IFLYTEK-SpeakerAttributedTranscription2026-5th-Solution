"""Dual-speaker-encoder temporal classifier for silence/single/overlap."""

from __future__ import annotations

import torch
from torch import nn


class DualEncoderPurityNet(nn.Module):
    def __init__(
        self,
        input_dimension: int = 576,
        projection_dimension: int = 96,
        recurrent_dimension: int = 96,
        recurrent_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.primary = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, projection_dimension),
            nn.GELU(),
        )
        self.secondary = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, projection_dimension),
            nn.GELU(),
        )
        fused_dimension = 2 * projection_dimension
        self.local = nn.Sequential(
            nn.Conv1d(fused_dimension, fused_dimension, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(fused_dimension, fused_dimension, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.recurrent = nn.GRU(
            fused_dimension,
            recurrent_dimension,
            num_layers=recurrent_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if recurrent_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * recurrent_dimension),
            nn.Dropout(dropout),
            nn.Linear(2 * recurrent_dimension, 3),
        )

    def forward(self, primary: torch.Tensor, secondary: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([self.primary(primary), self.secondary(secondary)], dim=-1)
        local = self.local(fused.transpose(1, 2)).transpose(1, 2)
        contextual, _ = self.recurrent(fused + local)
        return self.classifier(contextual)
