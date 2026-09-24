"""Dual-pretrained-encoder contextual speaker metric."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DualEncoderContextMetric(nn.Module):
    """Encode a complete conversation before computing speaker similarity."""

    def __init__(
        self,
        input_dimension: int = 576,
        projection_dimension: int = 128,
        model_dimension: int = 256,
        output_dimension: int = 192,
        attention_heads: int = 4,
        encoder_layers: int = 3,
        feedforward_dimension: int = 768,
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
        self.fusion = nn.Linear(fused_dimension * 2, model_dimension)
        self.local = nn.Sequential(
            nn.Conv1d(
                model_dimension,
                model_dimension,
                kernel_size=5,
                padding=2,
                groups=model_dimension,
            ),
            nn.GELU(),
            nn.Conv1d(model_dimension, model_dimension, kernel_size=1),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dimension,
            nhead=attention_heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            layer,
            num_layers=encoder_layers,
            norm=nn.LayerNorm(model_dimension),
        )
        self.embedding = nn.Linear(model_dimension, output_dimension)
        self.change_head = nn.Linear(model_dimension, 1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(8.0)))
        self.logit_bias = nn.Parameter(torch.tensor(-1.0))

    def encode(
        self, primary: torch.Tensor, secondary: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if primary.ndim == 2:
            primary = primary.unsqueeze(0)
            secondary = secondary.unsqueeze(0)
        fused = torch.cat([self.primary(primary), self.secondary(secondary)], dim=-1)
        delta = torch.zeros_like(fused)
        delta[:, 1:] = fused[:, 1:] - fused[:, :-1]
        hidden = self.fusion(torch.cat([fused, delta], dim=-1))
        hidden = hidden + self.local(hidden.transpose(1, 2)).transpose(1, 2)
        contextual = self.context(hidden)
        return F.normalize(self.embedding(contextual), dim=-1), self.change_head(
            contextual
        ).squeeze(-1)

    def pair_logits(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        cosine = (left * right).sum(dim=-1)
        return cosine * self.logit_scale.exp().clamp(max=30.0) + self.logit_bias


class DualEncoderResidualContextMetric(DualEncoderContextMetric):
    """Retain both pretrained identity geometries as a 384-D shortcut."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.context_residual_gate_logit = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def pretrained_identity(features: torch.Tensor) -> torch.Tensor:
        multiscale = features.reshape(*features.shape[:-1], 3, -1)
        return F.normalize(F.normalize(multiscale, dim=-1).mean(dim=-2), dim=-1)

    def encode(
        self, primary: torch.Tensor, secondary: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if primary.ndim == 2:
            primary = primary.unsqueeze(0)
            secondary = secondary.unsqueeze(0)
        contextual, change = super().encode(primary, secondary)
        pretrained = F.normalize(
            torch.cat(
                [
                    self.pretrained_identity(primary),
                    self.pretrained_identity(secondary),
                ],
                dim=-1,
            ),
            dim=-1,
        )
        if contextual.shape[-1] != pretrained.shape[-1]:
            raise ValueError(
                "Residual contextual output must match concatenated identity dimension"
            )
        gate = torch.sigmoid(self.context_residual_gate_logit)
        return F.normalize(pretrained + gate * contextual, dim=-1), change
