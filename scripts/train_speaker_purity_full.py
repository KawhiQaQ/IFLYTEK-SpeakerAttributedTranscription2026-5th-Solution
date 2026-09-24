#!/usr/bin/env python3
"""Train the V156 dual-encoder purity head on all official development data."""

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
from train_speaker_purity_head import target_classes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def indexed_features(root: Path) -> dict[str, Path]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("uses_test_data") is not False:
        raise RuntimeError(f"Development feature audit failed: {root}")
    paths: dict[str, Path] = {}
    for subset in ("train", "val"):
        for session_id in metadata["subsets"][subset]["sessions"]:
            if session_id in paths:
                raise RuntimeError(f"Duplicate development session: {session_id}")
            paths[session_id] = root / subset / f"{session_id}.pt"
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    feature_fold = int(config["feature_fold"])
    primary_paths = indexed_features(
        root
        / "data/speaker_graph"
        / config["primary_feature_experiment"]
        / f"fold_{feature_fold}"
    )
    secondary_paths = indexed_features(
        root
        / "data/speaker_graph"
        / config["secondary_feature_experiment"]
        / f"fold_{feature_fold}"
    )
    reference = json.loads((root / "data/dev/ref.seglst.json").read_text())
    reference_ids = {str(row["session_id"]) for row in reference}
    if set(primary_paths) != reference_ids or set(secondary_paths) != reference_ids:
        raise RuntimeError("Full-development feature coverage mismatch")
    session_ids = sorted(reference_ids)
    if args.smoke:
        session_ids = session_ids[:2]

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset = []
    counts = torch.zeros(3, dtype=torch.long)
    for session_id in session_ids:
        primary = torch.load(
            primary_paths[session_id], map_location="cpu", weights_only=True
        )
        secondary = torch.load(
            secondary_paths[session_id], map_location="cpu", weights_only=True
        )
        if not torch.allclose(primary["centers"], secondary["centers"], atol=1e-5):
            raise RuntimeError(f"Feature center mismatch: {session_id}")
        target = target_classes(
            primary["targets"].float(),
            int(config["training"].get("overlap_dilation_frames", 0)),
        )
        counts += torch.bincount(target, minlength=3)
        dataset.append(
            (
                session_id,
                primary["features"].float(),
                secondary["features"].float(),
                target,
            )
        )
    if not torch.all(counts > 0):
        raise RuntimeError(f"Purity classes are degenerate: {counts.tolist()}")

    training = config["training"]
    device = torch.device(training.get("device", "cuda:0"))
    model = DualEncoderPurityNet(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    weights = (counts.sum().float() / (3.0 * counts.float())).to(device)
    epochs = 1 if args.smoke else int(training["fixed_epochs"])
    checkpoint_path = (
        root / "outputs/diagnostics/v156_speaker_purity_full_smoke/purity_head.pt"
        if args.smoke
        else root / config["checkpoint_path"]
    )
    if checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_path.parent / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()
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
                noise = float(training["feature_noise_std"])
                primary = primary + torch.randn_like(primary) * noise
                secondary = secondary + torch.randn_like(secondary) * noise
            logits = model(primary.unsqueeze(0), secondary.unsqueeze(0)).squeeze(0)
            loss = F.cross_entropy(logits, target, weight=weights)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite purity loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Non-finite purity gradient")
            optimizer.step()
            losses.append(float(loss.detach()) * len(target))
            correct += int((logits.detach().argmax(dim=1) == target).sum())
            total += len(target)
        row = {
            "event": "epoch",
            "epoch": epoch,
            "loss": round(sum(losses) / total, 6),
            "frame_accuracy": round(correct / total, 6),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    artifact = {
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "model_config": config["model"],
        "train_sessions": sorted(reference_ids),
        "excluded_validation_sessions": [],
        "training_scope": "all_official_development",
        "fixed_training_epochs": epochs,
        "class_counts": counts.tolist(),
        "class_names": ["silence", "single", "overlap"],
        "uses_validation_labels": False,
        "uses_test_data": False,
        "uses_test_for_model_selection": False,
        "config_sha256": sha256_file(config_path),
    }
    torch.save(artifact, checkpoint_path)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_scope": "all_official_development",
        "sessions": len(session_ids),
        "fixed_training_epochs": epochs,
        "class_counts": counts.tolist(),
        "uses_test_data": False,
        "uses_test_for_model_selection": False,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (checkpoint_path.parent / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
