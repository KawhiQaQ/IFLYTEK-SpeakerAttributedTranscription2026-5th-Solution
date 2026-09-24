#!/usr/bin/env python3
"""Train a fold-pure temporal silence/single/overlap classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from speaker_purity_model import DualEncoderPurityNet


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_classes(
    targets: torch.Tensor, overlap_dilation_frames: int = 0
) -> torch.Tensor:
    active = (targets > 0.5).sum(dim=1)
    overlap = active >= 2
    if overlap_dilation_frames > 0:
        kernel = 2 * overlap_dilation_frames + 1
        overlap = (
            F.max_pool1d(
                overlap.float().view(1, 1, -1), kernel_size=kernel,
                stride=1, padding=overlap_dilation_frames,
            ).view(-1)
            > 0.5
        )
    return torch.where(overlap, 2, torch.where(active == 0, 0, 1)).long()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    fold = int(config["fold"] if args.fold is None else args.fold)
    split_dir = root / "data/splits" / f"fold_{fold}"
    train_sessions = (split_dir / "train_sessions.txt").read_text().split()
    validation_sessions = (split_dir / "val_sessions.txt").read_text().split()
    if set(train_sessions) & set(validation_sessions):
        raise RuntimeError("Purity-head fold isolation failed")
    if args.smoke:
        train_sessions = train_sessions[:2]

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    primary_dir = root / "data/speaker_graph" / config["primary_feature_experiment"] / f"fold_{fold}" / "train"
    secondary_dir = root / "data/speaker_graph" / config["secondary_feature_experiment"] / f"fold_{fold}" / "train"
    dataset = []
    counts = torch.zeros(3, dtype=torch.long)
    for session_id in train_sessions:
        primary = torch.load(primary_dir / f"{session_id}.pt", map_location="cpu", weights_only=False)
        secondary = torch.load(secondary_dir / f"{session_id}.pt", map_location="cpu", weights_only=False)
        if primary.get("uses_test_data") is not False or secondary.get("uses_test_data") is not False:
            raise RuntimeError("Purity training feature audit failed")
        if not torch.allclose(primary["centers"], secondary["centers"], atol=1e-5):
            raise RuntimeError(f"Feature center mismatch: {session_id}")
        target = target_classes(
            primary["targets"].float(),
            int(config["training"].get("overlap_dilation_frames", 0)),
        )
        counts += torch.bincount(target, minlength=3)
        dataset.append((session_id, primary["features"].float(), secondary["features"].float(), target))
    if not torch.all(counts > 0):
        raise RuntimeError(f"Purity classes are degenerate: {counts.tolist()}")

    device = torch.device(config["training"].get("device", "cpu"))
    model = DualEncoderPurityNet(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    weights = (counts.sum().float() / (3.0 * counts.float())).to(device)
    epochs = 1 if args.smoke else int(config["training"]["fixed_epochs"])
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(dataset)).tolist()
        losses = []
        correct = total = 0
        for index in order:
            _, primary, secondary, target = dataset[index]
            primary = primary.to(device)
            secondary = secondary.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            if not args.smoke:
                primary = primary + torch.randn_like(primary) * float(config["training"]["feature_noise_std"])
                secondary = secondary + torch.randn_like(secondary) * float(config["training"]["feature_noise_std"])
            logits = model(primary.unsqueeze(0), secondary.unsqueeze(0)).squeeze(0)
            loss = F.cross_entropy(logits, target, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()) * len(target))
            correct += int((logits.detach().argmax(dim=1) == target).sum())
            total += len(target)
        print(json.dumps({"event": "epoch", "epoch": epoch, "loss": round(sum(losses) / total, 6), "frame_accuracy": round(correct / total, 6)}), flush=True)

    checkpoint_path = root / str(config["checkpoint_path"]).format(fold=fold)
    if checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "model_config": config["model"],
        "fold": fold,
        "train_sessions": train_sessions,
        "excluded_validation_sessions": validation_sessions,
        "class_counts": counts.tolist(),
        "class_names": ["silence", "single", "overlap"],
        "primary_feature_experiment": config["primary_feature_experiment"],
        "secondary_feature_experiment": config["secondary_feature_experiment"],
        "uses_validation_labels": False,
        "uses_test_data": False,
        "config_sha256": sha256_file(config_path),
    }
    torch.save(artifact, checkpoint_path)
    print(json.dumps({"checkpoint": str(checkpoint_path), "train_sessions": len(train_sessions), "class_counts": counts.tolist(), "uses_validation_labels": False, "uses_test_data": False, "elapsed_seconds": round(time.time() - started, 2)}), flush=True)


if __name__ == "__main__":
    main()
