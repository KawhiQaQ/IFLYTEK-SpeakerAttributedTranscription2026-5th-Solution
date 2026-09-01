"""Speaker-slot sequence decoder over independent acoustic encoders."""

from __future__ import annotations

import math

import torch
from torch import nn


class SortedSpeakerSlotDecoder(nn.Module):
    """Decode conversation-local speakers ordered by their first appearance."""

    def __init__(
        self,
        input_sizes: list[int],
        maximum_speakers: int = 6,
        hidden_size: int = 256,
        timing_size: int = 5,
        attention_heads: int = 8,
        encoder_layers: int = 4,
        decoder_layers: int = 3,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_sizes = list(input_sizes)
        self.maximum_speakers = int(maximum_speakers)
        self.projections = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(size), nn.Linear(size, hidden_size), nn.GELU())
            for size in input_sizes
        )
        self.encoder_gate = nn.Sequential(
            nn.LayerNorm(hidden_size * len(input_sizes)),
            nn.Linear(hidden_size * len(input_sizes), hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, len(input_sizes)),
        )
        self.timing = nn.Sequential(
            nn.LayerNorm(timing_size), nn.Linear(timing_size, hidden_size), nn.GELU()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
        self.slot_queries = nn.Parameter(torch.empty(maximum_speakers, hidden_size))
        nn.init.normal_(self.slot_queries, std=0.02)
        pair_size = 4 * hidden_size
        self.assignment = nn.Sequential(
            nn.LayerNorm(pair_size),
            nn.Linear(pair_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.presence = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1)
        )

    @staticmethod
    def positional_encoding(values: torch.Tensor) -> torch.Tensor:
        length, dimension = values.shape
        position = torch.arange(
            length, device=values.device, dtype=values.dtype
        ).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimension, 2, device=values.device, dtype=values.dtype)
            * (-math.log(10000.0) / dimension)
        )
        result = torch.zeros_like(values)
        result[:, 0::2] = torch.sin(position * divisor)
        result[:, 1::2] = torch.cos(position * divisor)
        return result

    def forward(
        self, encoder_segments: list[torch.Tensor], timing: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(encoder_segments) != len(self.projections):
            raise ValueError("Sorted-slot encoder count mismatch")
        projected = [
            projection(values)
            for projection, values in zip(self.projections, encoder_segments)
        ]
        weights = self.encoder_gate(torch.cat(projected, dim=-1)).softmax(-1)
        fused = sum(
            weights[:, index : index + 1] * values
            for index, values in enumerate(projected)
        )
        hidden = fused + self.timing(timing)
        hidden = hidden + self.sequence_encoder(
            (hidden + self.positional_encoding(hidden)).unsqueeze(0)
        )[0]
        queries = self.slot_decoder(
            self.slot_queries.unsqueeze(0), hidden.unsqueeze(0)
        )[0]
        segment_rows = hidden[:, None, :].expand(-1, self.maximum_speakers, -1)
        query_rows = queries[None, :, :].expand(len(hidden), -1, -1)
        pair = torch.cat(
            [
                segment_rows,
                query_rows,
                segment_rows * query_rows,
                (segment_rows - query_rows).abs(),
            ],
            dim=-1,
        )
        presence = self.presence(queries).squeeze(-1)
        assignment = self.assignment(pair).squeeze(-1) + presence.unsqueeze(0)
        return assignment, presence
