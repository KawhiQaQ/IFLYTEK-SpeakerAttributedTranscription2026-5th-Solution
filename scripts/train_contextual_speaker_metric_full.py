#!/usr/bin/env python3
"""Train the V156 contextual speaker metric on all official development data."""

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

from contextual_speaker_metric_model import DualEncoderContextMetric
from train_speaker_metric_fold import pure_frame_labels, sampled_pairs


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
    training = config["training"]
    device = torch.device(training.get("device", "cuda:0"))
    model = DualEncoderContextMetric(**config["model"]).to(device)
    initial_checkpoint_path = None
    if config.get("initial_checkpoint_path"):
        initial_checkpoint_path = root / config["initial_checkpoint_path"]
        initial = torch.load(
            initial_checkpoint_path, map_location="cpu", weights_only=True
        )
        if (
            initial.get("uses_test_data") is not False
            or initial.get("model_config") != config["model"]
        ):
            raise RuntimeError("Initial contextual checkpoint audit failed")
        model.load_state_dict(initial["state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = 1 if args.smoke else int(training["fixed_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_bfloat16 = training.get("precision") == "bfloat16"
    checkpoint_path = (
        root
        / "outputs/diagnostics/v156_contextual_speaker_metric_full_smoke/contextual_metric.pt"
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

    def load(session_id: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        primary = torch.load(
            primary_paths[session_id], map_location="cpu", weights_only=True
        )
        secondary = torch.load(
            secondary_paths[session_id], map_location="cpu", weights_only=True
        )
        if not torch.allclose(primary["centers"], secondary["centers"], atol=1e-5):
            raise RuntimeError(f"Feature center mismatch: {session_id}")
        return (
            primary["features"].float(),
            secondary["features"].float(),
            primary["targets"].float(),
        )

    for epoch in range(1, epochs + 1):
        model.train()
        order = session_ids.copy()
        random.Random(seed + epoch).shuffle(order)
        losses = []
        pair_rows = 0
        for position, session_id in enumerate(order):
            primary, secondary, targets = load(session_id)
            indices, labels = pure_frame_labels(
                targets,
                float(training["pure_activity_min"]),
                float(training["other_activity_max"]),
            )
            generator = torch.Generator().manual_seed(
                seed + epoch * 1009 + position
            )
            left, right, pair_target = sampled_pairs(
                labels,
                int(training["pairs_per_class_per_session"]),
                generator,
            )
            if not len(pair_target):
                continue
            primary = primary.to(device)
            secondary = secondary.to(device)
            if not args.smoke:
                noise = float(training["feature_noise_std"])
                primary = primary + torch.randn_like(primary) * noise
                secondary = secondary + torch.randn_like(secondary) * noise
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16
            ):
                embeddings, change_logits = model.encode(primary, secondary)
                embeddings = embeddings.squeeze(0)[indices.to(device)]
                pair_logits = model.pair_logits(
                    embeddings[left.to(device)], embeddings[right.to(device)]
                )
                pair_loss = F.binary_cross_entropy_with_logits(
                    pair_logits, pair_target.to(device)
                )
                dominant = targets.argmax(dim=1).to(device)
                speech = (targets.sum(dim=1) > 0.5).to(device)
                change_target = torch.zeros(len(targets), device=device)
                change_target[1:] = (dominant[1:] != dominant[:-1]).float()
                valid_change = speech.clone()
                valid_change[1:] &= speech[:-1]
                valid_change[0] = False
                boundary_loss = (
                    F.binary_cross_entropy_with_logits(
                        change_logits.squeeze(0)[valid_change],
                        change_target[valid_change],
                    )
                    if valid_change.any()
                    else pair_loss.new_zeros(())
                )
                loss = pair_loss + float(training["boundary_loss_weight"]) * boundary_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite contextual loss in {session_id}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Non-finite contextual gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            pair_rows += len(pair_target)
        scheduler.step()
        row = {
            "event": "epoch",
            "epoch": epoch,
            "train_loss": round(float(np.mean(losses)), 6),
            "train_pairs": pair_rows,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "max_cuda_memory_gb": round(
                torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    artifact = {
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "model_config": config["model"],
        "model_class": "context_only",
        "train_sessions": sorted(reference_ids),
        "excluded_validation_sessions": [],
        "training_scope": "all_official_development",
        "fixed_training_epochs": epochs,
        "uses_validation_labels": False,
        "uses_test_data": False,
        "uses_test_for_model_selection": False,
        "initial_checkpoint_path": (
            str(initial_checkpoint_path) if initial_checkpoint_path else None
        ),
        "initial_checkpoint_sha256": (
            sha256_file(initial_checkpoint_path) if initial_checkpoint_path else None
        ),
        "config_sha256": sha256_file(config_path),
    }
    torch.save(artifact, checkpoint_path)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_scope": "all_official_development",
        "sessions": len(session_ids),
        "fixed_training_epochs": epochs,
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
