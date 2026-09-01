#!/usr/bin/env python3
"""Train the frozen V173 novel-speaker energy on all official development data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from novel_speaker_energy_features import FEATURE_NAMES, energy_features, multiscale_embeddings
from run_partition_quality_gate import frame_labels
from train_novel_speaker_energy import sha256_file, speaker_masks
from train_speaker_metric_fold import pure_frame_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dev_sessions = sorted(path.stem for path in (root / "data/dev/wav").glob("*.wav"))
    if len(dev_sessions) != 106:
        raise RuntimeError(f"Unexpected full-development coverage: {len(dev_sessions)}")
    feature_root = root / (
        f"data/speaker_features_label_free/{config['feature_experiment']}/dev"
    )
    label_base = root / f"data/speaker_graph/{config['label_source_experiment']}"
    training = config["training"]
    rows: list[list[float]] = []
    targets: list[int] = []
    weights: list[float] = []
    synthetic_count = 0
    for session_id in dev_sessions:
        acoustic = torch.load(
            feature_root / f"{session_id}.pt", map_location="cpu", weights_only=True
        )
        candidates = [
            label_base / f"fold_0/train/{session_id}.pt",
            label_base / f"fold_0/val/{session_id}.pt",
        ]
        label_path = next((path for path in candidates if path.is_file()), None)
        if label_path is None:
            raise FileNotFoundError(f"Full-development label feature missing: {session_id}")
        labelled = torch.load(label_path, map_location="cpu", weights_only=True)
        if (
            acoustic.get("uses_speaker_labels") is not False
            or acoustic.get("uses_test_data") is not False
            or labelled.get("uses_test_data") is not False
            or not torch.allclose(acoustic["centers"], labelled["centers"], atol=1e-5)
        ):
            raise RuntimeError(f"Full energy feature audit failed: {session_id}")
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
            rows.append(energy_features(embeddings, candidate, base).tolist())
            targets.append(1)
            weights.append(1.0)
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
                rows.append(
                    energy_features(
                        embeddings,
                        split_candidate,
                        [same_identity_remainder, *base],
                    ).tolist()
                )
                targets.append(0)
                weights.append(1.0)
                synthetic_count += 1

    real_sessions = []
    real_count = 0
    for source_fold in (0, 1):
        metadata_path = root / (
            f"outputs/{config['real_oof_sources']['injected']}/fold_{source_fold}/run_metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for session_id, audit in metadata["fusion_audits"].items():
            if not audit["novel_speaker_count"]:
                continue
            acoustic = torch.load(
                feature_root / f"{session_id}.pt", map_location="cpu", weights_only=True
            )
            labelled = torch.load(
                label_base / f"fold_{source_fold}/val/{session_id}.pt",
                map_location="cpu",
                weights_only=True,
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
                rows.append(energy_features(embeddings, candidate, base_masks).tolist())
                targets.append(int(candidate_identity not in base_identities))
                weights.append(float(training["real_oof_episode_weight"]))
                real_sessions.append(session_id)
                real_count += 1

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

    checkpoint_path = root / config["checkpoint_path"]
    if checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "experiment": config["name"],
        "trained_from_cv_architecture": config["trained_from_cv_architecture"],
        "feature_names": FEATURE_NAMES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficient": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "train_sessions": dev_sessions,
        "real_oof_sessions": sorted(set(real_sessions)),
        "synthetic_examples": synthetic_count,
        "real_oof_examples": real_count,
        "training_accuracy": accuracy,
        "uses_validation_labels": True,
        "uses_test_data": False,
        "uses_test_for_training_or_selection": False,
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
                "sessions": len(dev_sessions),
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
