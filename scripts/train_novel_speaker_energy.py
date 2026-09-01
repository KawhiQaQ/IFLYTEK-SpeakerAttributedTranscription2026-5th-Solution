#!/usr/bin/env python3
"""Train a regularized fold-pure energy model for novel-speaker existence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from novel_speaker_energy_features import FEATURE_NAMES, energy_features, multiscale_embeddings
from run_partition_quality_gate import frame_labels
from target_domain_novel_energy import conversation_domain, domain_feature_names
from train_speaker_metric_fold import pure_frame_labels


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def speaker_masks(
    frame_count: int, indices: torch.Tensor, labels: torch.Tensor
) -> dict[int, torch.Tensor]:
    masks = {}
    for speaker in torch.unique(labels).tolist():
        mask = torch.zeros(frame_count, dtype=torch.bool)
        mask[indices[labels == speaker]] = True
        masks[int(speaker)] = mask
    return masks


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
    split_root = root / f"data/splits/fold_{fold}"
    fold_train_sessions = (split_root / "train_sessions.txt").read_text().split()
    validation_sessions = (split_root / "val_sessions.txt").read_text().split()
    if set(fold_train_sessions) & set(validation_sessions):
        raise RuntimeError("Novel-speaker energy fold isolation failed")
    train_sessions = fold_train_sessions[:3] if args.smoke else fold_train_sessions
    feature_root = root / (
        f"data/speaker_features_label_free/{config['feature_experiment']}/dev"
    )
    label_root = root / (
        f"data/speaker_graph/{config['label_source_experiment']}/fold_{fold}/train"
    )
    training = config["training"]
    domain_config = config.get("target_domain_conditioning")
    domain_by_session: dict[str, torch.Tensor] = {}
    domain_mean = domain_scale = target_domain = None
    domain_weights: dict[str, float] = {}
    feature_names = list(FEATURE_NAMES)
    if domain_config:
        prediction_root = root / str(domain_config["dev_prediction_root"])
        descriptor_ids = sorted(set(fold_train_sessions) | set(validation_sessions))
        domain_by_session = {
            session_id: conversation_domain(
                prediction_root / f"{session_id}.json",
                feature_root / f"{session_id}.pt",
            )
            for session_id in descriptor_ids
        }
        source_values = torch.stack([domain_by_session[session_id] for session_id in fold_train_sessions])
        domain_mean = source_values.mean(0)
        domain_scale = source_values.std(0).clamp_min(0.05)
        target_domain = (
            torch.stack([domain_by_session[session_id] for session_id in validation_sessions]).mean(0)
            - domain_mean
        ) / domain_scale
        bandwidth = float(domain_config["weight_bandwidth"])
        minimum = float(domain_config["minimum_weight"])
        raw_weights = {}
        for session_id in fold_train_sessions:
            standardized = (domain_by_session[session_id] - domain_mean) / domain_scale
            distance = float(((standardized - target_domain) ** 2).mean().sqrt())
            raw_weights[session_id] = max(minimum, float(np.exp(-0.5 * (distance / bandwidth) ** 2)))
        normalizer = sum(raw_weights.values()) / len(raw_weights)
        domain_weights = {key: value / normalizer for key, value in raw_weights.items()}
        feature_names.extend(domain_feature_names())

    def model_features(values: torch.Tensor, session_id: str) -> list[float]:
        result = values.float()
        if domain_config:
            assert domain_mean is not None and domain_scale is not None and target_domain is not None
            session_domain = (domain_by_session[session_id] - domain_mean) / domain_scale
            result = torch.cat([result, session_domain, target_domain, session_domain - target_domain])
        return result.tolist()

    rows: list[list[float]] = []
    targets: list[int] = []
    weights: list[float] = []
    synthetic_count = 0
    for session_id in train_sessions:
        acoustic = torch.load(
            feature_root / f"{session_id}.pt", map_location="cpu", weights_only=True
        )
        labelled = torch.load(
            label_root / f"{session_id}.pt", map_location="cpu", weights_only=True
        )
        if (
            acoustic.get("uses_speaker_labels") is not False
            or acoustic.get("uses_test_data") is not False
            or labelled.get("uses_test_data") is not False
            or not torch.allclose(acoustic["centers"], labelled["centers"], atol=1e-5)
        ):
            raise RuntimeError(f"Energy training feature audit failed: {session_id}")
        embeddings = multiscale_embeddings(acoustic["features"])
        indices, labels = pure_frame_labels(
            labelled["targets"].float(),
            float(training["pure_activity_min"]),
            float(training["other_activity_max"]),
        )
        masks = speaker_masks(len(embeddings), indices, labels)
        for speaker, candidate in masks.items():
            candidate_indices = torch.where(candidate)[0]
            if len(candidate_indices) < 2:
                continue
            base = [mask for other, mask in masks.items() if other != speaker]
            if not base:
                continue
            rows.append(model_features(energy_features(embeddings, candidate, base), session_id))
            targets.append(1)
            weights.append(domain_weights.get(session_id, 1.0))
            synthetic_count += 1
            if len(candidate_indices) < 4:
                continue
            subset_size = max(2, min(len(candidate_indices) // 2, 12))
            selections = [
                candidate_indices[:subset_size],
                candidate_indices[
                    torch.linspace(0, len(candidate_indices) - 1, subset_size)
                    .round()
                    .long()
                ],
            ]
            for selection in selections:
                split_candidate = torch.zeros(len(embeddings), dtype=torch.bool)
                split_candidate[selection] = True
                same_identity_remainder = candidate & ~split_candidate
                split_base = [same_identity_remainder, *base]
                rows.append(model_features(
                    energy_features(embeddings, split_candidate, split_base), session_id
                ))
                targets.append(0)
                weights.append(domain_weights.get(session_id, 1.0))
                synthetic_count += 1

    # The original two-fold experiments used the other completed OOF fold.
    # For later folds, consume every already-frozen OOF source fold that lies
    # wholly inside this fold's training set.  Validation identities are still
    # excluded and the inference rule/model family are unchanged.
    configured_by_target = config.get("real_oof_source_folds_by_target", {})
    configured_source_folds = configured_by_target.get(
        fold, configured_by_target.get(str(fold))
    )
    source_folds = (
        [int(value) for value in configured_source_folds]
        if configured_source_folds is not None
        else [1 - fold]
    )
    real_sessions = []
    real_count = 0
    for source_fold in source_folds:
        source_metadata_path = root / (
            f"outputs/{config['real_oof_sources']['injected']}/fold_{source_fold}/run_metadata.json"
        )
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        for session_id, audit in source_metadata["fusion_audits"].items():
            if not audit["novel_speaker_count"]:
                continue
            if session_id not in fold_train_sessions or session_id in validation_sessions:
                raise RuntimeError(f"Real energy episode isolation failed: {session_id}")
            acoustic = torch.load(
                feature_root / f"{session_id}.pt", map_location="cpu", weights_only=True
            )
            labelled = torch.load(
                label_root / f"{session_id}.pt", map_location="cpu", weights_only=True
            )
            embeddings = multiscale_embeddings(acoustic["features"])
            centers = acoustic["centers"]
            complementary = json.loads(
                (
                    root
                    / f"outputs/{config['real_oof_sources']['complementary']}/fold_{source_fold}/sessions/{session_id}.json"
                ).read_text(encoding="utf-8")
            )["segments"]
            native = json.loads(
                (
                    root
                    / f"outputs/{config['real_oof_sources']['native']}/fold_{source_fold}/sessions/{session_id}.json"
                ).read_text(encoding="utf-8")
            )["segments"]
            complementary_labels = frame_labels(centers, complementary)
            base_labels = frame_labels(centers, native)
            strength, identity = labelled["targets"].float().max(dim=1)
            for novel_speaker in audit["novel_complementary_speakers"]:
                candidate = torch.tensor(
                    [speaker == novel_speaker for speaker in complementary_labels]
                )
                if not int(candidate.sum()):
                    continue
                base_masks = []
                base_identities = []
                for base_speaker in sorted(set(base_labels)):
                    mask = torch.tensor(
                        [speaker == base_speaker for speaker in base_labels]
                    ) & ~candidate
                    if not int(mask.sum()):
                        continue
                    base_masks.append(mask)
                    labelled_mask = mask & (strength >= 0.5)
                    if int(labelled_mask.sum()):
                        base_identities.append(
                            int(
                                torch.bincount(
                                    identity[labelled_mask],
                                    minlength=labelled["targets"].shape[1],
                                ).argmax()
                            )
                        )
                labelled_candidate = candidate & (strength >= 0.5)
                if not int(labelled_candidate.sum()) or not base_masks:
                    continue
                candidate_identity = int(
                    torch.bincount(
                        identity[labelled_candidate],
                        minlength=labelled["targets"].shape[1],
                    ).argmax()
                )
                rows.append(model_features(energy_features(embeddings, candidate, base_masks), session_id))
                targets.append(int(candidate_identity not in base_identities))
                weights.append(
                    float(training["real_oof_episode_weight"])
                    * domain_weights.get(session_id, 1.0)
                )
                real_sessions.append(session_id)
                real_count += 1
    if len(set(targets)) != 2 or not real_count:
        raise RuntimeError("Novel-speaker energy examples are incomplete")

    features = np.asarray(rows, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(features)
    classifier = LogisticRegression(
        C=float(training["inverse_regularization"]),
        class_weight="balanced",
        max_iter=int(training["max_iterations"]),
        random_state=int(config["seed"]),
        solver="lbfgs",
    )
    classifier.fit(standardized, labels, sample_weight=sample_weights)
    probabilities = classifier.predict_proba(standardized)[:, 1]
    accuracy = float(np.mean((probabilities >= 0.5) == labels))

    checkpoint_path = (
        root / f"outputs/diagnostics/{config['name']}_fold_{fold}_smoke/energy.json"
        if args.smoke
        else root / str(config["checkpoint_path"]).format(fold=fold)
    )
    if checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "experiment": config["name"],
        "fold": fold,
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficient": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "train_sessions": train_sessions,
        "validation_sessions": validation_sessions,
        "real_oof_source_folds": source_folds,
        "real_oof_sessions": sorted(set(real_sessions)),
        "synthetic_examples": synthetic_count,
        "real_oof_examples": real_count,
        "training_accuracy": accuracy,
        "uses_validation_labels": False,
        "uses_test_data": False,
        "uses_target_distribution_aggregate": bool(domain_config),
        "target_domain_uses_labels": False,
        "domain_names": domain_feature_names() if domain_config else [],
        "domain_mean": domain_mean.tolist() if domain_mean is not None else None,
        "domain_scale": domain_scale.tolist() if domain_scale is not None else None,
        "target_domain": target_domain.tolist() if target_domain is not None else None,
        "domain_session_weights": domain_weights,
        "training_config": training,
        "feature_metadata_sha256": sha256_file(feature_root / "metadata.json"),
        "config_sha256": sha256_file(config_path),
    }
    checkpoint_path.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "examples": len(rows),
                "synthetic": synthetic_count,
                "real_oof": real_count,
                "training_accuracy": accuracy,
                "sha256": sha256_file(checkpoint_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
