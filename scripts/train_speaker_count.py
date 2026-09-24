#!/usr/bin/env python3
"""Train a fixed, fold-pure speaker-count classifier without hyperparameter search."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


SEED = 20260827


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--fold", required=True, help="Fold number or 'full'")
    args = parser.parse_args()

    root = args.project_root.resolve()
    feature_path = root / "reports" / "speaker_count_features_v1.json"
    feature_payload = json.loads(feature_path.read_text(encoding="utf-8"))
    if feature_payload.get("uses_speaker_labels") is not False:
        raise RuntimeError("Feature cache is not label-free")
    feature_names = feature_payload["feature_names"]
    feature_rows = feature_payload["sessions"]

    labels = {}
    manifest_path = root / "configs" / "cv" / "folds_v1.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels[row["session_id"]] = int(row["speakers"])

    if args.fold == "full":
        fold: int | str = "full"
        train_sessions = sorted(labels)
        val_sessions: list[str] = []
    else:
        fold = int(args.fold)
        split_dir = root / "data" / "splits" / f"fold_{fold}"
        train_sessions = (split_dir / "train_sessions.txt").read_text(
            encoding="utf-8"
        ).split()
        val_sessions = (split_dir / "val_sessions.txt").read_text(
            encoding="utf-8"
        ).split()
        if set(train_sessions) & set(val_sessions):
            raise RuntimeError("Fold leakage: train and validation sessions overlap")

    def matrix(session_ids: list[str]) -> np.ndarray:
        return np.asarray(
            [[float(feature_rows[sid][name]) for name in feature_names] for sid in session_ids],
            dtype=np.float64,
        )

    train_x = matrix(train_sessions)
    train_y = np.asarray([labels[sid] for sid in train_sessions], dtype=np.int64)
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)

    # Validation labels are read only after fitting and only for an honest report.
    predictions = []
    validation_accuracy = None
    validation_mae = None
    confusion = None
    if val_sessions:
        val_x = matrix(val_sessions)
        val_y = np.asarray([labels[sid] for sid in val_sessions], dtype=np.int64)
        predicted = model.predict(val_x).astype(int)
        probabilities = model.predict_proba(val_x)
        validation_accuracy = float(np.mean(predicted == val_y))
        validation_mae = float(np.mean(np.abs(predicted - val_y)))
        confusion = confusion_matrix(
            val_y, predicted, labels=[2, 3, 4, 5, 6]
        ).tolist()
        for session_id, truth, prediction, probability in zip(
            val_sessions, val_y, predicted, probabilities
        ):
            predictions.append(
                {
                    "session_id": session_id,
                    "reference_count": int(truth),
                    "predicted_count": int(prediction),
                    "probabilities": {
                        str(int(label)): float(value)
                        for label, value in zip(model.classes_, probability)
                    },
                }
            )

    artifact = {
        "model": model,
        "fold": fold,
        "seed": SEED,
        "feature_names": feature_names,
        "train_sessions": train_sessions,
        "validation_sessions": val_sessions,
        "feature_sha256": sha256_file(feature_path),
        "cv_manifest_sha256": sha256_file(manifest_path),
    }
    model_dir = root / "models" / "speaker_count"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"rf_v1_fold_{fold}.joblib"
    joblib.dump(artifact, model_path)

    report = {
        "fold": fold,
        "train_sessions": len(train_sessions),
        "validation_sessions": len(val_sessions),
        "validation_accuracy": validation_accuracy,
        "validation_mae": validation_mae,
        "confusion_labels": [2, 3, 4, 5, 6],
        "confusion_matrix": confusion,
        "predictions": predictions,
        "model_path": str(model_path),
    }
    report_path = root / "reports" / f"speaker_count_rf_v1_fold_{fold}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
