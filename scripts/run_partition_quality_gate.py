#!/usr/bin/env python3
"""Route between joint and modular partitions using acoustic coherence only.

The gate detects catastrophic partition collapse rather than small quality
differences.  It switches the complete session only when a complementary
partition has no fewer speakers and improves coherence by a large factor in
both the raw multi-scale speaker space and a fold-pure learned metric space.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F

from speaker_metric_model import SpeakerMetricAdapter


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_labels(centers: torch.Tensor, segments: list[dict]) -> list[str]:
    if not segments:
        raise RuntimeError("Cannot score an empty speaker partition")
    labels: list[str] = []
    for center in centers.tolist():
        covering = [
            row
            for row in segments
            if float(row["start_time"]) <= center <= float(row["end_time"])
        ]
        if covering:
            selected = min(
                covering,
                key=lambda row: abs(
                    (float(row["start_time"]) + float(row["end_time"])) / 2
                    - center
                ),
            )
        else:
            selected = min(
                segments,
                key=lambda row: min(
                    abs(center - float(row["start_time"])),
                    abs(center - float(row["end_time"])),
                ),
            )
        labels.append(str(selected["speaker"]))
    return labels


def coherence_margin(embeddings: torch.Tensor, labels: list[str]) -> float:
    embeddings = F.normalize(embeddings.float(), dim=1)
    speakers = sorted(set(labels))
    if len(speakers) < 2:
        return 0.0
    indices = torch.as_tensor([speakers.index(label) for label in labels])
    centroids = torch.stack(
        [
            F.normalize(embeddings[indices == speaker].mean(dim=0), dim=0)
            for speaker in range(len(speakers))
        ]
    )
    similarities = embeddings @ centroids.T
    own = similarities[torch.arange(len(embeddings)), indices]
    similarities[torch.arange(len(embeddings)), indices] = -99.0
    return float((own - similarities.max(dim=1).values).mean())


def raw_multiscale_embeddings(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] % 3:
        raise RuntimeError(f"Unexpected multi-scale feature shape: {features.shape}")
    scales = features.float().reshape(len(features), 3, -1)
    scales = F.normalize(scales, dim=2)
    return F.normalize(scales.mean(dim=1), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fold = int(config["fold"] if args.fold is None else args.fold)
    split_dir = root / "data" / "splits" / f"fold_{fold}"
    session_ids = (split_dir / "val_sessions.txt").read_text().split()

    base_dir = root / "outputs" / config["base_experiment"] / f"fold_{fold}"
    alternative_dir = (
        root / "outputs" / config["alternative_experiment"] / f"fold_{fold}"
    )
    feature_dir = (
        root
        / "data"
        / "speaker_features_label_free"
        / config["feature_experiment"]
        / f"fold_{fold}"
        / "val"
    )
    checkpoint_path = root / str(config["metric_checkpoint"]).format(fold=fold)
    required = [
        directory / "sessions" / f"{session_id}.json"
        for directory in (base_dir, alternative_dir)
        for session_id in session_ids
    ] + [feature_dir / f"{session_id}.pt" for session_id in session_ids]
    if not all(path.is_file() for path in required) or not checkpoint_path.is_file():
        raise FileNotFoundError("Partition gate inputs are incomplete")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        set(checkpoint["train_sessions"]) & set(session_ids)
        or checkpoint.get("uses_test_data") is not False
    ):
        raise RuntimeError("Partition gate metric checkpoint audit failed")
    metric = SpeakerMetricAdapter(**checkpoint["model_config"])
    metric.load_state_dict(checkpoint["state_dict"])
    metric.eval()

    gate = config["gate"]
    raw_min_ratio = float(gate["raw_min_ratio"])
    learned_min_ratio = float(gate["learned_min_ratio"])
    preserve_speaker_count = bool(gate.get("preserve_speaker_count", True))
    output_dir = root / "outputs" / config["name"] / f"fold_{fold}"
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict] = []
    decisions: dict[str, dict] = {}
    started = time.time()

    for session_id in session_ids:
        output_path = session_dir / f"{session_id}.json"
        if output_path.exists() and not args.overwrite:
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            output_rows.extend(saved["segments"])
            decisions[session_id] = saved["decision"]
            continue
        base_path = base_dir / "sessions" / f"{session_id}.json"
        alternative_path = alternative_dir / "sessions" / f"{session_id}.json"
        feature_path = feature_dir / f"{session_id}.pt"
        base_segments = json.loads(base_path.read_text())["segments"]
        alternative_segments = json.loads(alternative_path.read_text())["segments"]
        acoustic = torch.load(feature_path, map_location="cpu", weights_only=True)
        if (
            acoustic.get("uses_speaker_labels") is not False
            or acoustic.get("uses_test_data") is not False
        ):
            raise RuntimeError("Partition gate acoustic feature audit failed")
        features = acoustic["features"]
        centers = acoustic["centers"]
        base_labels = frame_labels(centers, base_segments)
        alternative_labels = frame_labels(centers, alternative_segments)
        raw_embeddings = raw_multiscale_embeddings(features)
        with torch.inference_mode():
            learned_embeddings = metric.embed(features.float())

        base_raw = coherence_margin(raw_embeddings, base_labels)
        alternative_raw = coherence_margin(raw_embeddings, alternative_labels)
        base_learned = coherence_margin(learned_embeddings, base_labels)
        alternative_learned = coherence_margin(
            learned_embeddings, alternative_labels
        )
        raw_ratio = alternative_raw / max(base_raw, 1e-6)
        learned_ratio = alternative_learned / max(base_learned, 1e-6)
        base_speakers = len(set(base_labels))
        alternative_speakers = len(set(alternative_labels))
        count_safe = (
            not preserve_speaker_count or alternative_speakers >= base_speakers
        )
        use_alternative = (
            count_safe
            and raw_ratio >= raw_min_ratio
            and learned_ratio >= learned_min_ratio
        )
        selected = alternative_segments if use_alternative else base_segments
        decision = {
            "selected": "alternative" if use_alternative else "base",
            "base_speakers": base_speakers,
            "alternative_speakers": alternative_speakers,
            "count_safe": count_safe,
            "base_raw_margin": base_raw,
            "alternative_raw_margin": alternative_raw,
            "raw_ratio": raw_ratio,
            "base_learned_margin": base_learned,
            "alternative_learned_margin": alternative_learned,
            "learned_ratio": learned_ratio,
        }
        saved = {
            "session_id": session_id,
            "development_only": True,
            "uses_validation_labels": False,
            "uses_test_data": False,
            "base_source_sha256": sha256_file(base_path),
            "alternative_source_sha256": sha256_file(alternative_path),
            "feature_source_sha256": sha256_file(feature_path),
            "metric_checkpoint_sha256": sha256_file(checkpoint_path),
            "decision": decision,
            "segments": selected,
        }
        output_path.write_text(
            json.dumps(saved, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_rows.extend(selected)
        decisions[session_id] = decision
        print(json.dumps({"session": session_id, **decision}), flush=True)

    prediction_path = output_dir / "hyp.seglst.json"
    prediction_path.write_text(
        json.dumps(output_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "experiment": config["name"],
        "fold": fold,
        "development_only": True,
        "uses_validation_labels": False,
        "uses_test_data": False,
        "session_ids": session_ids,
        "config": config,
        "config_sha256": sha256_file(config_path),
        "metric_checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_alternative_sessions": [
            session_id
            for session_id, row in decisions.items()
            if row["selected"] == "alternative"
        ],
        "decisions": decisions,
        "prediction_sha256": sha256_file(prediction_path),
        "elapsed_seconds": round(time.time() - started, 3),
        "argv": sys.argv,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
