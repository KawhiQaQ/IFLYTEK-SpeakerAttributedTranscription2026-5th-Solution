"""Prototype-conditioned target-speaker activity decoder."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TargetSpeakerActivityNet(nn.Module):
    """TS-VAD style encoder-decoder over two pretrained speaker spaces."""

    def __init__(
        self,
        input_dimension: int = 576,
        projection_dimension: int = 128,
        model_dimension: int = 256,
        attention_heads: int = 4,
        encoder_layers: int = 3,
        decoder_layers: int = 2,
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
        self.frame_projection = nn.Linear(fused_dimension * 2, model_dimension)
        self.prototype_projection = nn.Linear(fused_dimension, model_dimension)
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
            encoder_layer,
            num_layers=encoder_layers,
            norm=nn.LayerNorm(model_dimension),
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
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(model_dimension),
        )
        self.frame_activity_bias = nn.Linear(model_dimension, 1)
        self.pretrained_logit_scale = nn.Parameter(torch.tensor(math.log(8.0)))
        self.context_logit_scale = nn.Parameter(torch.tensor(math.log(4.0)))
        self.context_residual_gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.logit_bias = nn.Parameter(torch.tensor(-1.0))

    def _fuse(self, primary: torch.Tensor, secondary: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.primary(primary), self.secondary(secondary)], dim=-1)

    def forward(
        self,
        primary_frames: torch.Tensor,
        secondary_frames: torch.Tensor,
        primary_prototypes: torch.Tensor,
        secondary_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        if primary_frames.ndim == 2:
            primary_frames = primary_frames.unsqueeze(0)
            secondary_frames = secondary_frames.unsqueeze(0)
            primary_prototypes = primary_prototypes.unsqueeze(0)
            secondary_prototypes = secondary_prototypes.unsqueeze(0)
        fused = self._fuse(primary_frames, secondary_frames)
        delta = torch.zeros_like(fused)
        delta[:, 1:] = fused[:, 1:] - fused[:, :-1]
        hidden = self.frame_projection(torch.cat([fused, delta], dim=-1))
        hidden = hidden + self.local(hidden.transpose(1, 2)).transpose(1, 2)
        encoded = self.encoder(hidden)
        queries = self.prototype_projection(
            self._fuse(primary_prototypes, secondary_prototypes)
        )
        speakers = self.decoder(queries, encoded)
        frames = F.normalize(encoded, dim=-1)
        speakers = F.normalize(speakers, dim=-1)
        primary_frame_identity = F.normalize(
            F.normalize(
                primary_frames.reshape(*primary_frames.shape[:-1], 3, -1), dim=-1
            ).mean(dim=-2),
            dim=-1,
        )
        secondary_frame_identity = F.normalize(
            F.normalize(
                secondary_frames.reshape(*secondary_frames.shape[:-1], 3, -1), dim=-1
            ).mean(dim=-2),
            dim=-1,
        )
        primary_prototype_identity = F.normalize(
            F.normalize(
                primary_prototypes.reshape(*primary_prototypes.shape[:-1], 3, -1), dim=-1
            ).mean(dim=-2),
            dim=-1,
        )
        secondary_prototype_identity = F.normalize(
            F.normalize(
                secondary_prototypes.reshape(*secondary_prototypes.shape[:-1], 3, -1), dim=-1
            ).mean(dim=-2),
            dim=-1,
        )
        pretrained_similarity = 0.5 * (
            torch.einsum(
                "btd,bkd->btk", primary_frame_identity, primary_prototype_identity
            )
            + torch.einsum(
                "btd,bkd->btk",
                secondary_frame_identity,
                secondary_prototype_identity,
            )
        )
        context_similarity = torch.einsum("btd,bkd->btk", frames, speakers)
        context_residual = (
            context_similarity * self.context_logit_scale.exp().clamp(max=30.0)
            + self.frame_activity_bias(encoded)
        )
        return (
            pretrained_similarity
            * self.pretrained_logit_scale.exp().clamp(max=30.0)
            + torch.sigmoid(self.context_residual_gate_logit) * context_residual
            + self.logit_bias
        )
