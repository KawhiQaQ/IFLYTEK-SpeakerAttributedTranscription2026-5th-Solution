#!/usr/bin/env python3
"""Train the boundary-robust speaker metric on all official development data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from speaker_metric_model import SpeakerMetricAdapter
from train_speaker_metric_fold import metric_examples, sampled_pairs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    feature_root = (
        root
        / "data"
        / "speaker_graph"
        / config["feature_experiment"]
        / f"fold_{feature_fold}"
    )
    feature_metadata = json.loads((feature_root / "metadata.json").read_text())
    if feature_metadata.get("uses_test_data") is not False:
        raise RuntimeError("Development feature audit failed")
    source_paths: dict[str, Path] = {}
    for subset in ("train", "val"):
        for session_id in feature_metadata["subsets"][subset]["sessions"]:
            if session_id in source_paths:
                raise RuntimeError(f"Duplicate development session: {session_id}")
            source_paths[session_id] = feature_root / subset / f"{session_id}.pt"
    reference = json.loads((root / "data/dev/ref.seglst.json").read_text())
    reference_ids = {str(row["session_id"]) for row in reference}
    if set(source_paths) != reference_ids:
        raise RuntimeError("Full-development feature coverage mismatch")
    session_ids = sorted(source_paths)
    if args.smoke:
        session_ids = session_ids[:2]

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
    epochs = 1 if args.smoke else int(training["fixed_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_bfloat16 = training["precision"] == "bfloat16"

    configured_output = root / config["output_dir"]
    output_dir = (
        root / "outputs/diagnostics/v92_boundary_metric_full_smoke"
        if args.smoke
        else configured_output
    )
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

    started = time.time()
    log(
        {
            "event": "start",
            "training_scope": "all_official_development",
            "sessions": len(session_ids),
            "fixed_epochs": epochs,
            "epoch_selection_source": training["epoch_selection_source"],
            "uses_test_data": False,
            "uses_test_for_model_selection": False,
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
        }
    )
    for epoch in range(1, epochs + 1):
        model.train()
        order = session_ids.copy()
        random.Random(seed + epoch).shuffle(order)
        losses: list[float] = []
        train_pairs = 0
        grad_norms: list[float] = []
        for position, session_id in enumerate(order):
            payload = torch.load(
                source_paths[session_id], map_location="cpu", weights_only=True
            )
            example_features, labels = metric_examples(
                payload["features"], payload["targets"], training
            )
            generator = torch.Generator().manual_seed(
                seed + epoch * 1009 + position
            )
            left, right, target = sampled_pairs(
                labels, int(training["pairs_per_class_per_session"]), generator
            )
            if not len(target):
                continue
            features = example_features.to(device)
            pair_features = torch.stack([features[left], features[right]])
            mask_probability = float(training["scale_mask_probability"])
            if mask_probability > 0:
                mask = (
                    torch.rand(
                        pair_features.shape[:-1]
                        + (int(config["model"]["scales"]), 1),
                        device=device,
                    )
                    < mask_probability
                )
                reshaped = pair_features.reshape(
                    *pair_features.shape[:-1],
                    int(config["model"]["scales"]),
                    int(config["model"]["embedding_dimension"]),
                )
                pair_features = reshaped.masked_fill(mask, 0.0).reshape_as(
                    pair_features
                )
            pair_features = pair_features + torch.randn_like(pair_features) * float(
                training["feature_noise_std"]
            )
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=use_bfloat16
            ):
                logits = model(pair_features[0], pair_features[1])
                loss = F.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in {session_id}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Non-finite gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            grad_norms.append(float(grad_norm.detach()))
            train_pairs += len(target)
        scheduler.step()
        log(
            {
                "event": "epoch",
                "epoch": epoch,
                "train_loss": round(float(np.mean(losses)), 6),
                "mean_grad_norm": round(float(np.mean(grad_norms)), 6),
                "train_pairs": train_pairs,
                "residual_gate": round(
                    float(torch.sigmoid(model.residual_gate_logit).detach()), 6
                ),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": config["model"],
            "train_sessions": sorted(source_paths),
            "val_sessions": [],
            "training_scope": "all_official_development",
            "fixed_epochs": epochs,
            "epoch_selection_source": training["epoch_selection_source"],
            "uses_test_data": False,
            "uses_test_for_model_selection": False,
        },
        checkpoint_path,
    )
    metadata = {
        "experiment": config["name"],
        "smoke": args.smoke,
        "training_scope": "all_official_development",
        "sessions": len(session_ids),
        "fixed_epochs": epochs,
        "epoch_selection_source": training["epoch_selection_source"],
        "uses_test_data": False,
        "uses_test_for_model_selection": False,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
