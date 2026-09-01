#!/usr/bin/env python3
"""Train a fold-pure same-speaker metric without tcpWER threshold tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from torch.nn import functional as F

from speaker_metric_model import SpeakerMetricAdapter


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pure_frame_labels(
    targets: torch.Tensor, activity_min: float, other_max: float
) -> tuple[torch.Tensor, torch.Tensor]:
    dominant_value, dominant = targets.max(dim=1)
    other = targets.sum(dim=1) - dominant_value
    keep = (dominant_value >= activity_min) & (other <= other_max)
    return torch.nonzero(keep, as_tuple=False).squeeze(1), dominant[keep]


def metric_examples(
    features: torch.Tensor, targets: torch.Tensor, training: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create train/inference-matched acoustic nodes and dominant identities."""
    window = int(training.get("node_window_frames", 1))
    stride = int(training.get("node_stride_frames", 1))
    if window == 1:
        indices, labels = pure_frame_labels(
            targets,
            float(training["pure_activity_min"]),
            float(training["other_activity_max"]),
        )
        return features[indices], labels
    if window < 1 or stride < 1:
        raise ValueError("Node window and stride must be positive")
    node_features, node_activity = [], []
    for start in range(0, max(1, len(features) - window + 1), stride):
        end = min(len(features), start + window)
        node_features.append(features[start:end].mean(dim=0))
        node_activity.append(targets[start:end].mean(dim=0))
    features_out = torch.stack(node_features)
    activity = torch.stack(node_activity)
    dominant_value, labels = activity.max(dim=1)
    dominance = dominant_value / activity.sum(dim=1).clamp_min(1e-6)
    keep = (
        (dominant_value >= float(training["dominant_activity_min"]))
        & (dominance >= float(training["dominance_ratio_min"]))
    )
    return features_out[keep], labels[keep]


def sampled_pairs(
    labels: torch.Tensor, per_class: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positives: list[tuple[int, int]] = []
    negatives: list[tuple[int, int]] = []
    unique = labels.unique().tolist()
    for speaker in unique:
        indices = torch.nonzero(labels == speaker, as_tuple=False).squeeze(1).tolist()
        if len(indices) >= 2:
            positives.extend((indices[i], indices[j]) for i in range(len(indices)) for j in range(i + 1, len(indices)))
    for left_speaker_index, left_speaker in enumerate(unique):
        left = torch.nonzero(labels == left_speaker, as_tuple=False).squeeze(1).tolist()
        for right_speaker in unique[left_speaker_index + 1 :]:
            right = torch.nonzero(labels == right_speaker, as_tuple=False).squeeze(1).tolist()
            negatives.extend((i, j) for i in left for j in right)
    if not positives or not negatives:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, empty.float()

    def choose(rows: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(rows) <= per_class:
            return rows
        order = torch.randperm(len(rows), generator=generator)[:per_class].tolist()
        return [rows[index] for index in order]

    positive_rows = choose(positives)
    negative_rows = choose(negatives)
    rows = positive_rows + negative_rows
    left = torch.tensor([row[0] for row in rows], dtype=torch.long)
    right = torch.tensor([row[1] for row in rows], dtype=torch.long)
    target = torch.cat([torch.ones(len(positive_rows)), torch.zeros(len(negative_rows))])
    return left, right, target


def equal_error_rate(target: np.ndarray, score: np.ndarray) -> float:
    false_positive, true_positive, _ = roc_curve(target, score)
    false_negative = 1.0 - true_positive
    index = int(np.argmin(np.abs(false_positive - false_negative)))
    return float((false_positive[index] + false_negative[index]) / 2)


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
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fold = int(config["fold"] if args.fold is None else args.fold)
    feature_root = root / "data" / "speaker_graph" / config["feature_experiment"] / f"fold_{fold}"
    feature_metadata = json.loads((feature_root / "metadata.json").read_text())
    train_ids = feature_metadata["subsets"]["train"]["sessions"]
    val_ids = feature_metadata["subsets"]["val"]["sessions"]
    if set(train_ids) & set(val_ids) or feature_metadata.get("uses_test_data") is not False:
        raise RuntimeError("Fold-purity audit failed")
    if args.smoke:
        train_ids, val_ids = train_ids[:2], val_ids[:2]

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    model = SpeakerMetricAdapter(**config["model"]).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    fixed_epoch_mode = "fixed_epochs" in training and not args.smoke
    max_epochs = 1 if args.smoke else int(
        training["fixed_epochs"] if fixed_epoch_mode else training["max_epochs"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    use_bfloat16 = training["precision"] == "bfloat16"

    configured_output = root / str(config["output_dir"]).format(fold=fold)
    output_dir = root / "outputs" / "diagnostics" / f"{config['name']}_smoke" if args.smoke else configured_output
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    log_path = output_dir / "train_log.jsonl"
    checkpoint_path = output_dir / "metric_adapter.pt"

    def log(row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    log({
        "event": "start", "fold": fold, "development_only": True,
        "uses_validation_labels_for_gradient": False, "uses_test_data": False,
        "train_sessions": len(train_ids), "val_sessions": len(val_ids),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    })
    best_auc, best_epoch = -math.inf, 0
    patience_left = int(training["patience"])
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = train_ids.copy()
        random.Random(seed + epoch).shuffle(order)
        losses = []
        train_pairs = 0
        for session_position, session_id in enumerate(order):
            payload = torch.load(feature_root / "train" / f"{session_id}.pt", map_location="cpu", weights_only=True)
            example_features, labels = metric_examples(
                payload["features"], payload["targets"], training
            )
            generator = torch.Generator().manual_seed(seed + epoch * 1009 + session_position)
            left, right, target = sampled_pairs(labels, int(training["pairs_per_class_per_session"]), generator)
            if not len(target):
                continue
            features = example_features.to(device)
            pair_features = torch.stack([features[left], features[right]])
            if float(training["scale_mask_probability"]) > 0:
                mask = torch.rand(pair_features.shape[:-1] + (int(config["model"]["scales"]), 1), device=device) < float(training["scale_mask_probability"])
                reshaped = pair_features.reshape(*pair_features.shape[:-1], int(config["model"]["scales"]), int(config["model"]["embedding_dimension"]))
                pair_features = reshaped.masked_fill(mask, 0.0).reshape_as(pair_features)
            pair_features = pair_features + torch.randn_like(pair_features) * float(training["feature_noise_std"])
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                logits = model(pair_features[0], pair_features[1])
                loss = F.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in {session_id}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Non-finite gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            train_pairs += len(target)

        model.eval()
        target_rows: list[np.ndarray] = []
        learned_rows: list[np.ndarray] = []
        baseline_rows: list[np.ndarray] = []
        with torch.inference_mode():
            for session_position, session_id in enumerate(val_ids):
                payload = torch.load(feature_root / "val" / f"{session_id}.pt", map_location="cpu", weights_only=True)
                example_features, labels = metric_examples(
                    payload["features"], payload["targets"], training
                )
                generator = torch.Generator().manual_seed(seed + 100000 + session_position)
                left, right, target = sampled_pairs(labels, int(training["pairs_per_class_per_session"]), generator)
                if not len(target):
                    continue
                features = example_features.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                    learned = model.embed(features)
                multiscale = F.normalize(features.reshape(len(features), int(config["model"]["scales"]), -1), dim=-1)
                baseline = F.normalize(multiscale.mean(dim=1), dim=-1)
                target_rows.append(target.numpy())
                learned_rows.append((learned[left] * learned[right]).sum(-1).float().cpu().numpy())
                baseline_rows.append((baseline[left] * baseline[right]).sum(-1).float().cpu().numpy())
        target_array = np.concatenate(target_rows)
        learned_array = np.concatenate(learned_rows)
        baseline_array = np.concatenate(baseline_rows)
        learned_auc = float(roc_auc_score(target_array, learned_array))
        baseline_auc = float(roc_auc_score(target_array, baseline_array))
        learned_eer = equal_error_rate(target_array, learned_array)
        scheduler.step()
        log({
            "event": "epoch", "epoch": epoch,
            "train_loss": round(float(np.mean(losses)), 6), "train_pairs": train_pairs,
            "validation_pair_auc": round(learned_auc, 6),
            "validation_pair_eer": round(learned_eer, 6),
            "baseline_pair_auc": round(baseline_auc, 6),
            "residual_gate": round(float(torch.sigmoid(model.residual_gate_logit).detach()), 6),
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        if fixed_epoch_mode:
            # Monitor the held-out pair metric, but never use it for checkpoint
            # selection.  The final fixed-budget state is saved after the loop.
            best_auc, best_epoch = learned_auc, epoch
        elif learned_auc > best_auc:
            best_auc, best_epoch = learned_auc, epoch
            patience_left = int(training["patience"])
            torch.save({
                "state_dict": model.state_dict(), "fold": fold, "model_config": config["model"],
                "train_sessions": feature_metadata["subsets"]["train"]["sessions"],
                "val_sessions": feature_metadata["subsets"]["val"]["sessions"],
                "uses_test_data": False, "best_epoch": epoch,
                "best_validation_pair_auc": learned_auc, "baseline_validation_pair_auc": baseline_auc,
            }, checkpoint_path)
        else:
            patience_left -= 1
            if not args.smoke and patience_left <= 0:
                log({"event": "early_stop", "epoch": epoch})
                break

    if fixed_epoch_mode:
        torch.save({
            "state_dict": model.state_dict(), "fold": fold, "model_config": config["model"],
            "train_sessions": feature_metadata["subsets"]["train"]["sessions"],
            "val_sessions": feature_metadata["subsets"]["val"]["sessions"],
            "uses_test_data": False, "best_epoch": max_epochs,
            "fixed_training_epochs": max_epochs,
            "uses_validation_pair_auc_for_checkpoint_selection": False,
            "best_validation_pair_auc": learned_auc,
            "baseline_validation_pair_auc": baseline_auc,
        }, checkpoint_path)

    metadata = {
        "experiment": config["name"], "fold": fold, "smoke": args.smoke,
        "development_only": True, "uses_validation_labels_for_gradient": False,
        "uses_test_data": False, "best_epoch": best_epoch,
        "fixed_epoch_mode": fixed_epoch_mode,
        "uses_validation_pair_auc_for_checkpoint_selection": not fixed_epoch_mode,
        "best_validation_pair_auc": best_auc,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
