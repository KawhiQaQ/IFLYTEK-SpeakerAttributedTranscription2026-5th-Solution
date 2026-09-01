#!/usr/bin/env python3
"""Small, explicit LoRA implementation for the public FireRedASR2-AED model."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Frozen linear layer with a float32 low-rank residual branch."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        # Retain float32 optimizer state and low-rank accumulation even when the
        # 1.18B frozen backbone runs in bfloat16.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            hidden = F.linear(self.dropout(inputs.float()), self.lora_a)
            residual = F.linear(hidden, self.lora_b) * self.scale
        return base_output + residual.to(base_output.dtype)


def _parent_and_leaf(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    target_pattern: str,
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    """Freeze ``model`` and replace every full-regex-matched linear module."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    pattern = re.compile(target_pattern)
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and pattern.fullmatch(name)
    ]
    if not targets:
        raise RuntimeError(f"LoRA pattern matched no linear modules: {target_pattern}")
    for name in targets:
        parent, leaf = _parent_and_leaf(model, name)
        base = getattr(parent, leaf)
        adapter = LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout)
        adapter.to(device=base.weight.device)
        setattr(parent, leaf, adapter)
    return targets


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.endswith(".lora_a") or name.endswith(".lora_b")
    }


def save_adapter(
    path: Path,
    model: nn.Module,
    *,
    target_pattern: str,
    rank: int,
    alpha: float,
    dropout: float,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "firered_aed_explicit_lora_v1",
            "target_pattern": target_pattern,
            "rank": int(rank),
            "alpha": float(alpha),
            "dropout": float(dropout),
            "adapter_state_dict": adapter_state_dict(model),
            "metadata": metadata,
        },
        path,
    )


def load_adapter(model: nn.Module, path: Path) -> tuple[list[str], dict]:
    package = torch.load(path, map_location="cpu", weights_only=False)
    if package.get("format") != "firered_aed_explicit_lora_v1":
        raise RuntimeError(f"Unsupported FireRed adapter format: {package.get('format')}")
    targets = inject_lora(
        model,
        target_pattern=package["target_pattern"],
        rank=int(package["rank"]),
        alpha=float(package["alpha"]),
        dropout=float(package["dropout"]),
    )
    missing, unexpected = model.load_state_dict(package["adapter_state_dict"], strict=False)
    unexpected = [name for name in unexpected if ".lora_" in name]
    missing_lora = [name for name in missing if ".lora_" in name]
    if unexpected or missing_lora:
        raise RuntimeError(
            f"Adapter state mismatch: missing={missing_lora[:8]}, unexpected={unexpected[:8]}"
        )
    return targets, dict(package.get("metadata", {}))
