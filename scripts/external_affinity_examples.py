"""Public AISHELL-4 training examples for peer-partition affinity pretraining."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from torch.nn import functional as F

from residual_affinity_data import base_same_matrix, domain_descriptor
from train_sorted_slot_diarizer import sequence_features


def parse_rttm(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[0] != "SPEAKER":
            continue
        start, duration = float(fields[3]), float(fields[4])
        if duration < 0.20:
            continue
        rows.append(
            {
                "start_time": start,
                "end_time": start + duration,
                "speaker": fields[7],
            }
        )
    return sorted(rows, key=lambda row: (row["start_time"], row["end_time"], row["speaker"]))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(labels: list[str | int]) -> list[str]:
    mapping: dict[str | int, str] = {}
    return [mapping.setdefault(value, f"spk{len(mapping)}") for value in labels]


def closest_pair(labels: list[str], values: torch.Tensor) -> tuple[str, str]:
    speakers = sorted(set(labels))
    prototypes = {
        speaker: F.normalize(values[[value == speaker for value in labels]].mean(0), dim=0)
        for speaker in speakers
    }
    return max(
        (
            (left, right)
            for position, left in enumerate(speakers)
            for right in speakers[position + 1 :]
        ),
        key=lambda pair: float(prototypes[pair[0]] @ prototypes[pair[1]]),
    )


def merge(labels: list[str], left: str, right: str) -> list[str]:
    return canonical([left if value == right else value for value in labels])


def rare_merge(labels: list[str], values: torch.Tensor) -> list[str]:
    counts = {speaker: labels.count(speaker) for speaker in set(labels)}
    rare = min(counts, key=lambda speaker: (counts[speaker], speaker))
    other = [speaker for speaker in sorted(counts) if speaker != rare]
    rare_value = F.normalize(values[[value == rare for value in labels]].mean(0), dim=0)
    target = max(
        other,
        key=lambda speaker: float(
            rare_value @ F.normalize(values[[value == speaker for value in labels]].mean(0), dim=0)
        ),
    )
    return merge(labels, rare, target)


def local_noise(
    labels: list[str], values: torch.Tensor, rng: random.Random, probability: float
) -> list[str]:
    speakers = sorted(set(labels))
    prototypes = {
        speaker: F.normalize(values[[value == speaker for value in labels]].mean(0), dim=0)
        for speaker in speakers
    }
    output = list(labels)
    for index, speaker in enumerate(labels):
        if rng.random() >= probability or len(speakers) < 2:
            continue
        alternatives = [value for value in speakers if value != speaker]
        output[index] = max(
            alternatives, key=lambda value: float(values[index] @ prototypes[value])
        )
    return canonical(output)


def temporal_split(labels: list[str], rng: random.Random) -> list[str]:
    eligible = [speaker for speaker in sorted(set(labels)) if labels.count(speaker) >= 4]
    if not eligible:
        return canonical(labels)
    speaker = eligible[rng.randrange(len(eligible))]
    positions = [index for index, value in enumerate(labels) if value == speaker]
    boundary = positions[len(positions) // 2]
    return canonical(
        [f"{value}_late" if value == speaker and index >= boundary else value for index, value in enumerate(labels)]
    )


def double_merge(labels: list[str], values: torch.Tensor) -> list[str]:
    output = labels
    for _ in range(2):
        if len(set(output)) <= 2:
            break
        left, right = closest_pair(output, values)
        output = merge(output, left, right)
    return canonical(output)


def simulated_partitions(
    truth: list[str], values: torch.Tensor, candidate_count: int, seed: int
) -> tuple[list[str], torch.Tensor]:
    if candidate_count not in {5, 8}:
        raise ValueError("AISHELL-4 peer simulation supports the frozen five- or eight-peer graphs")
    rng = random.Random(seed)
    close_left, close_right = closest_pair(truth, values)
    closest = merge(truth, close_left, close_right)
    rare = rare_merge(truth, values)
    draw = rng.random()
    if draw < 0.62:
        base = closest
    elif draw < 0.84:
        base = rare
    elif draw < 0.94:
        base = local_noise(truth, values, rng, 0.10)
    else:
        base = canonical(truth)
    if candidate_count == 5:
        # Match the deployed peer semantics: stable first pass, fixed-count
        # reassignment, hard under-count merge, additive missing-speaker split,
        # and an independent joint alternative.
        partitions = [
            base,
            local_noise(truth, values, rng, 0.05),
            double_merge(truth, values),
            temporal_split(truth, rng),
            canonical(truth)
            if rng.random() < 0.70
            else local_noise(truth, values, rng, 0.13),
        ]
    else:
        partitions = [
            base,
            local_noise(truth, values, rng, 0.13),
            temporal_split(truth, rng),
            closest,
            rare,
            double_merge(truth, values),
            local_noise(truth, values, rng, 0.05),
            canonical(truth) if rng.random() < 0.70 else base,
        ]
    relations = torch.stack(
        [base_same_matrix([str(value) for value in labels]) for labels in partitions],
        dim=-1,
    )
    return canonical(base), relations


def acoustic_cluster_partition(distance: np.ndarray) -> list[str]:
    """Create a label-free partition using only pretrained acoustic geometry."""
    count = len(distance)
    if count <= 2:
        return canonical(list(range(count)))
    maximum = min(7, count - 1)
    best: tuple[float, int, np.ndarray] | None = None
    for clusters in range(2, maximum + 1):
        labels = AgglomerativeClustering(
            n_clusters=clusters,
            metric="precomputed",
            linkage="average",
        ).fit_predict(distance)
        score = float(silhouette_score(distance, labels, metric="precomputed"))
        item = (score, -clusters, labels)
        if best is None or item[:2] > best[:2]:
            best = item
    if best is None:
        raise RuntimeError("Acoustic clustering produced no partition")
    return canonical(best[2].tolist())


def acoustic_partitions(values: list[torch.Tensor]) -> tuple[list[str], torch.Tensor]:
    """Five real acoustic clusterers: four encoders plus their distance ensemble."""
    distances = []
    for encoder_values in values:
        normalized = F.normalize(encoder_values.float(), dim=-1)
        distance = (1.0 - normalized @ normalized.T).clamp(0.0, 2.0)
        distance.fill_diagonal_(0.0)
        distances.append(distance.numpy())
    ensemble_distance = np.mean(np.stack(distances), axis=0)
    partitions = [
        acoustic_cluster_partition(ensemble_distance),
        *(acoustic_cluster_partition(distance) for distance in distances),
    ]
    relations = torch.stack(
        [base_same_matrix(partition) for partition in partitions], dim=-1
    )
    return partitions[0], relations


def load_external_affinity_examples(
    root: Path, config: dict, candidate_count: int
) -> list[dict]:
    settings = config["external_pretraining"]
    dataset_root = root / settings["dataset_root"]
    audit = json.loads((dataset_root / "audit.json").read_text(encoding="utf-8"))
    if (
        audit.get("source_split") != "train only"
        or audit.get("competition_test_used") is not False
        or audit.get("external_test_used") is not False
    ):
        raise RuntimeError("External AISHELL-4 provenance audit failed")
    feature_roots = [root / value for value in settings["feature_roots"].values()]
    window = float(settings.get("window_seconds", 240.0))
    stride = float(settings.get("stride_seconds", 180.0))
    minimum_speakers = int(settings.get("minimum_window_speakers", 4))
    maximum_segments = int(settings.get("maximum_window_segments", 64))
    examples = []
    for session in audit["sessions"]:
        session_id = str(session["session_id"])
        rttm_path = dataset_root / "rttm" / f"{session_id}.rttm"
        rows = parse_rttm(rttm_path)
        feature_paths = [directory / f"{session_id}.pt" for directory in feature_roots]
        payloads = [
            torch.load(path, map_location="cpu", weights_only=True) for path in feature_paths
        ]
        feature_sha256 = [sha256_file(path) for path in feature_paths]
        if any(
            payload.get("uses_test_data") is not False
            or payload.get("uses_speaker_labels") is not False
            for payload in payloads
        ):
            raise RuntimeError(f"External feature provenance failed: {session_id}")
        duration = min(float(payload["duration"]) for payload in payloads)
        window_index = 0
        for begin in np.arange(0.0, max(duration - window / 2, 0.1), stride):
            end = min(float(begin + window), duration)
            selected = [
                {
                    **row,
                    "start_time": max(float(row["start_time"]), float(begin)),
                    "end_time": min(float(row["end_time"]), end),
                }
                for row in rows
                if float(row["end_time"]) > begin and float(row["start_time"]) < end
            ]
            selected = [
                row for row in selected if row["end_time"] - row["start_time"] >= 0.20
            ]
            if len(set(row["speaker"] for row in selected)) < minimum_speakers:
                continue
            if len(selected) > maximum_segments:
                indices = np.linspace(0, len(selected) - 1, maximum_segments).round().astype(int)
                selected = [selected[index] for index in sorted(set(indices.tolist()))]
            # A duration-derived token proxy keeps the pair weighting realistic
            # without consuming the external transcript or any language labels.
            segments = []
            for row in selected:
                token_count = max(1, min(30, int(round((row["end_time"] - row["start_time"]) * 3.2))))
                segments.append({**row, "words": " ".join(["x"] * token_count)})
            values, timing = sequence_features(segments, payloads)
            truth = [str(row["speaker"]) for row in segments]
            seed_text = f"{settings.get('seed', 47601)}:{session_id}:{window_index}"
            seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
            partition_mode = str(settings.get("partition_mode", "simulated"))
            if partition_mode == "simulated":
                base, candidate_same = simulated_partitions(
                    truth, F.normalize(values[0].float(), dim=-1), candidate_count, seed
                )
            elif partition_mode == "acoustic_clustering":
                if candidate_count != len(values) + 1:
                    raise RuntimeError(
                        "Acoustic-clustering peers require one ensemble plus every encoder"
                    )
                base, candidate_same = acoustic_partitions(values)
            else:
                raise ValueError(f"Unsupported external partition mode: {partition_mode}")
            base_segments = [
                {**row, "speaker": label} for row, label in zip(segments, base)
            ]
            example_id = f"aishell4:{session_id}:{int(begin):04d}"
            examples.append(
                {
                    "session_id": example_id,
                    "source_fold": -1,
                    "segments": base_segments,
                    "values": values,
                    "timing": timing,
                    "base_same": base_same_matrix(base),
                    "candidate_same": candidate_same,
                    "domain": domain_descriptor(base_segments, values, duration),
                    "targets": base_same_matrix(truth),
                    "truth_labels": truth,
                    "target_count": len(set(truth)),
                    "token_weights": torch.tensor(
                        [max(1, len(row["words"].split())) for row in segments],
                        dtype=torch.float32,
                    ),
                    "external_public_train": True,
                    "external_partition_mode": partition_mode,
                    "source_sha256": sha256_file(rttm_path),
                    "feature_sha256": feature_sha256,
                }
            )
            window_index += 1
    if not examples:
        raise RuntimeError("No AISHELL-4 external affinity examples were constructed")
    return examples
