#!/usr/bin/env python3
"""Arbitrate MOSS/V38 partition disagreements with fold-pure acoustic prototypes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from run_segment_novel_speaker_fusion import assign
from contextual_speaker_metric_model import (
    DualEncoderContextMetric,
    DualEncoderResidualContextMetric,
)
from target_speaker_activity_model import TargetSpeakerActivityNet
from run_partition_quality_gate import frame_labels
from prototype_conflict_model import PrototypeConflictHead
from run_v7_test_submission import validate_segments
from speaker_purity_model import DualEncoderPurityNet
from speaker_metric_model import SpeakerMetricAdapter
from noisy_support_speaker_model import NoisySupportSpeakerRanker, fixed_support


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_tokens(base_segments: list[dict], complementary: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for segment_index, segment in enumerate(base_segments):
        tokens = str(segment["words"]).split()
        if not tokens:
            continue
        start, end = float(segment["start_time"]), float(segment["end_time"])
        step = max(0.0, end - start) / len(tokens)
        for token_index, token in enumerate(tokens):
            token_start = start + token_index * step
            token_end = end if token_index + 1 == len(tokens) else start + (token_index + 1) * step
            interval = (token_start, token_end)
            rows.append(
                {
                    "segment_index": segment_index,
                    "token": token,
                    "start": token_start,
                    "end": token_end,
                    "base": str(segment["speaker"]),
                    "complementary": assign(interval, complementary),
                }
            )
    return rows


def make_nodes(rows: list[dict], max_seconds: float, max_gap: float) -> list[dict]:
    nodes: list[dict] = []
    for token_index, row in enumerate(rows):
        new = (
            not nodes
            or row["base"] != nodes[-1]["base"]
            or row["start"] - nodes[-1]["end"] > max_gap
            or row["end"] - nodes[-1]["start"] > max_seconds
        )
        if new:
            nodes.append(
                {
                    "start": row["start"],
                    "end": row["end"],
                    "base": row["base"],
                    "token_indices": [token_index],
                }
            )
        else:
            nodes[-1]["end"] = row["end"]
            nodes[-1]["token_indices"].append(token_index)
    return nodes


def embed_nodes(
    nodes: list[dict],
    features: torch.Tensor,
    centers: torch.Tensor,
    model: SpeakerMetricAdapter,
) -> torch.Tensor:
    acoustic_rows = []
    for node in nodes:
        selected = torch.nonzero(
            (centers >= node["start"]) & (centers <= node["end"]),
            as_tuple=False,
        ).squeeze(1)
        if not len(selected):
            midpoint = (node["start"] + node["end"]) / 2
            selected = torch.tensor([int(torch.argmin(torch.abs(centers - midpoint)))])
        acoustic_rows.append(features[selected].mean(dim=0))
    with torch.inference_mode():
        return model.embed(torch.stack(acoustic_rows).float())


def pool_contextual_nodes(
    nodes: list[dict], centers: torch.Tensor, frame_embeddings: torch.Tensor
) -> torch.Tensor:
    rows = []
    for node in nodes:
        selected = torch.nonzero(
            (centers >= node["start"]) & (centers <= node["end"]),
            as_tuple=False,
        ).squeeze(1)
        if not len(selected):
            midpoint = (node["start"] + node["end"]) / 2
            selected = torch.tensor([int(torch.argmin(torch.abs(centers - midpoint)))])
        rows.append(F.normalize(frame_embeddings[selected].mean(dim=0), dim=0))
    return torch.stack(rows)


def build_prototype_bank(
    nodes: list[dict],
    rows: list[dict],
    embeddings: torch.Tensor,
    base_speakers: list[str],
) -> tuple[dict[str, torch.Tensor], dict[str, int], list[str], list[str], int]:
    consensus: dict[str, list[torch.Tensor]] = defaultdict(list)
    all_base: dict[str, list[torch.Tensor]] = defaultdict(list)
    node_alternatives: list[str] = []
    for node_index, node in enumerate(nodes):
        labels = [rows[index]["mapped_complementary"] for index in node["token_indices"]]
        durations: dict[str, float] = defaultdict(float)
        for token_index, label in zip(node["token_indices"], labels):
            durations[label] += max(
                0.01, rows[token_index]["end"] - rows[token_index]["start"]
            )
        node_alternatives.append(max(durations, key=durations.get))
        all_base[node["base"]].append(embeddings[node_index])
        if all(label == node["base"] for label in labels):
            consensus[node["base"]].append(embeddings[node_index])
    prototypes = {}
    supports = {}
    fallback_speakers = []
    for speaker in base_speakers:
        values = consensus[speaker]
        if not values:
            values = all_base[speaker]
            fallback_speakers.append(speaker)
        prototypes[speaker] = F.normalize(torch.stack(values).mean(dim=0), dim=0)
        supports[speaker] = len(values)
    return (
        prototypes,
        supports,
        fallback_speakers,
        node_alternatives,
        sum(len(values) for values in consensus.values()),
    )


def dominant_node_labels(
    nodes: list[dict], rows: list[dict], key: str
) -> list[str]:
    labels: list[str] = []
    for node in nodes:
        durations: dict[str, float] = defaultdict(float)
        for token_index in node["token_indices"]:
            row = rows[token_index]
            durations[str(row[key])] += max(0.01, row["end"] - row["start"])
        labels.append(max(durations, key=durations.get))
    return labels


def build_partition_prototypes(
    node_labels: list[str], embeddings: torch.Tensor
) -> dict[str, torch.Tensor]:
    values: dict[str, list[torch.Tensor]] = defaultdict(list)
    for label, embedding in zip(node_labels, embeddings):
        values[label].append(embedding)
    return {
        label: F.normalize(torch.stack(speaker_embeddings).mean(dim=0), dim=0)
        for label, speaker_embeddings in values.items()
    }


def build_node_support_bank(
    nodes: list[dict],
    rows: list[dict],
    embeddings: torch.Tensor,
    base_speakers: list[str],
    support_size: int,
) -> dict[str, torch.Tensor]:
    """Keep individual consensus anchors for contamination-robust set models."""
    consensus: dict[str, list[torch.Tensor]] = defaultdict(list)
    all_base: dict[str, list[torch.Tensor]] = defaultdict(list)
    for node_index, node in enumerate(nodes):
        speaker = str(node["base"])
        all_base[speaker].append(embeddings[node_index])
        if all(
            rows[token_index]["mapped_complementary"] == speaker
            for token_index in node["token_indices"]
        ):
            consensus[speaker].append(embeddings[node_index])
    return {
        speaker: fixed_support(
            torch.stack(consensus[speaker] or all_base[speaker]), support_size
        )
        for speaker in base_speakers
    }


def acoustic_role_mapping(
    base_node_labels: list[str],
    complementary_node_labels: list[str],
    embeddings: torch.Tensor,
    base_speakers: list[str],
    complementary_speakers: list[str],
) -> dict[str, str]:
    """Link independently named partitions with global acoustic identities."""
    base_prototypes = build_partition_prototypes(base_node_labels, embeddings)
    complementary_prototypes = build_partition_prototypes(
        complementary_node_labels, embeddings
    )
    available_base = [speaker for speaker in base_speakers if speaker in base_prototypes]
    available_complementary = [
        speaker
        for speaker in complementary_speakers
        if speaker in complementary_prototypes
    ]
    if not available_base or not available_complementary:
        return {}
    similarity = np.asarray(
        [
            [
                float(complementary_prototypes[left] @ base_prototypes[right])
                for right in available_base
            ]
            for left in available_complementary
        ],
        dtype=np.float64,
    )
    left, right = linear_sum_assignment(-similarity)
    return {
        available_complementary[i]: available_base[j]
        for i, j in zip(left.tolist(), right.tolist())
    }


def has_multi_speaker_overlap(
    start: float, end: float, diarization: list[list[float]]
) -> bool:
    for left_index, left in enumerate(diarization):
        for right in diarization[left_index + 1 :]:
            if int(left[2]) == int(right[2]):
                continue
            overlap_start = max(start, float(left[0]), float(right[0]))
            overlap_end = min(end, float(left[1]), float(right[1]))
            if overlap_end > overlap_start:
                return True
    return False


def rebuild_segments(
    session_id: str, base_segments: list[dict], rows: list[dict]
) -> list[dict]:
    rows_by_segment: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_segment[int(row["segment_index"])].append(row)
    output: list[dict] = []
    for segment_index, original in enumerate(base_segments):
        segment_rows = rows_by_segment.get(segment_index, [])
        if not segment_rows or all(row["final"] == row["base"] for row in segment_rows):
            output.append(dict(original))
            continue
        current: dict | None = None
        for row in segment_rows:
            if current is None or current["speaker"] != row["final"]:
                current = {
                    "session_id": session_id,
                    "speaker": row["final"],
                    "start_time": row["start"],
                    "end_time": row["end"],
                    "tokens": [row["token"]],
                }
                output.append(current)
            else:
                current["end_time"] = row["end"]
                current["tokens"].append(row["token"])
    speaker_map: dict[str, str] = {}
    normalized = []
    for row in sorted(
        output,
        key=lambda item: (
            float(item["start_time"]),
            float(item["end_time"]),
            str(item["speaker"]),
        ),
    ):
        raw = str(row["speaker"])
        speaker_map.setdefault(raw, f"spk{len(speaker_map) + 1}")
        words = row.get("words")
        if words is None:
            words = " ".join(row["tokens"])
        normalized.append(
            {
                "session_id": session_id,
                "speaker": speaker_map[raw],
                "start_time": round(float(row["start_time"]), 2),
                "end_time": round(float(row["end_time"]), 2),
                "words": words,
            }
        )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--session-id")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fold = int(config["fold"] if args.fold is None else args.fold)
    if args.final_test:
        test_dir = (root / "data/test").resolve()
        if test_dir.name != "test":
            raise RuntimeError("Final-test path guard failed")
        wav_paths = sorted((test_dir / "wav").glob("*.wav"))
        session_ids = [path.stem for path in wav_paths]
        validation_ids: list[str] = []
    else:
        wav_paths = []
        validation_ids = (
            root / "data/splits" / f"fold_{fold}" / "val_sessions.txt"
        ).read_text().split()
        session_ids = validation_ids
    suffix = ""
    if args.session_id:
        if args.final_test:
            raise RuntimeError("Per-session diagnostics are development-only")
        if args.session_id not in validation_ids:
            raise RuntimeError("Diagnostic session is outside the frozen validation fold")
        session_ids = [args.session_id]
        suffix = "_diagnostic"

    if args.final_test:
        test_sources = config["test_sources"]
        base_dir = root / test_sources["base"]
        complementary_dir = root / test_sources["complementary"]
        feature_dir = root / test_sources["features"]
    else:
        base_dir = root / "outputs" / config["base_experiment"] / f"fold_{fold}" / "sessions"
        complementary_dir = root / "outputs" / config["complementary_experiment"] / f"fold_{fold}" / "sessions"
        feature_dir = root / "data/speaker_features_label_free" / config["feature_experiment"] / f"fold_{fold}" / "val"
    checkpoint_path = root / str(config["metric_checkpoint"]).format(fold=fold)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if args.final_test:
        if (
            checkpoint.get("training_scope") != "all_official_development"
            or checkpoint.get("uses_test_data") is not False
            or checkpoint.get("uses_test_for_model_selection") is not False
        ):
            raise RuntimeError("Metric checkpoint full-development audit failed")
    elif set(checkpoint["train_sessions"]) & set(validation_ids) or checkpoint.get("uses_test_data") is not False:
        raise RuntimeError("Metric checkpoint fold audit failed")
    model = SpeakerMetricAdapter(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    secondary_feature_dir = None
    secondary_checkpoint_path = None
    secondary_model = None
    if config["arbitration"].get("secondary_feature_experiment"):
        secondary_feature_dir = (
            root / config["test_sources"]["secondary_features"]
            if args.final_test
            else root
            / "data/speaker_features_label_free"
            / config["arbitration"]["secondary_feature_experiment"]
            / f"fold_{fold}"
            / "val"
        )
        secondary_checkpoint_path = root / str(
            config["arbitration"]["secondary_metric_checkpoint"]
        ).format(fold=fold)
        secondary_checkpoint = torch.load(
            secondary_checkpoint_path, map_location="cpu", weights_only=True
        )
        if args.final_test:
            if (
                secondary_checkpoint.get("training_scope")
                != "all_official_development"
                or secondary_checkpoint.get("uses_test_data") is not False
                or secondary_checkpoint.get("uses_test_for_model_selection") is not False
            ):
                raise RuntimeError(
                    "Secondary metric checkpoint full-development audit failed"
                )
        elif (
            set(secondary_checkpoint["train_sessions"]) & set(validation_ids)
            or secondary_checkpoint.get("uses_test_data") is not False
        ):
            raise RuntimeError("Secondary metric checkpoint fold audit failed")
        secondary_model = SpeakerMetricAdapter(**secondary_checkpoint["model_config"])
        secondary_model.load_state_dict(secondary_checkpoint["state_dict"])
        secondary_model.eval()
    purity_checkpoint_path = None
    purity_model = None
    if config["arbitration"].get("purity_checkpoint"):
        purity_checkpoint_path = root / str(
            config["arbitration"]["purity_checkpoint"]
        ).format(fold=fold)
        purity_checkpoint = torch.load(
            purity_checkpoint_path, map_location="cpu", weights_only=True
        )
        if args.final_test:
            if (
                purity_checkpoint.get("training_scope")
                != "all_official_development"
                or purity_checkpoint.get("uses_test_data") is not False
                or purity_checkpoint.get("uses_test_for_model_selection") is not False
            ):
                raise RuntimeError("Purity-head full-development audit failed")
        elif (
            set(purity_checkpoint["train_sessions"]) & set(validation_ids)
            or purity_checkpoint.get("uses_validation_labels") is not False
            or purity_checkpoint.get("uses_test_data") is not False
        ):
            raise RuntimeError("Purity-head fold audit failed")
        if secondary_model is None:
            raise RuntimeError("Purity head requires both acoustic feature spaces")
        purity_model = DualEncoderPurityNet(**purity_checkpoint["model_config"])
        purity_model.load_state_dict(purity_checkpoint["state_dict"])
        purity_model.eval()
    contextual_checkpoint_path = None
    contextual_model = None
    if config["arbitration"].get("contextual_metric_checkpoint"):
        contextual_checkpoint_path = root / str(
            config["arbitration"]["contextual_metric_checkpoint"]
        ).format(fold=fold)
        contextual_checkpoint = torch.load(
            contextual_checkpoint_path, map_location="cpu", weights_only=True
        )
        if args.final_test:
            if (
                contextual_checkpoint.get("training_scope")
                != "all_official_development"
                or contextual_checkpoint.get("uses_test_data") is not False
                or contextual_checkpoint.get("uses_test_for_model_selection") is not False
            ):
                raise RuntimeError(
                    "Contextual metric checkpoint full-development audit failed"
                )
        elif (
            set(contextual_checkpoint["train_sessions"]) & set(validation_ids)
            or contextual_checkpoint.get("uses_validation_labels") is not False
            or contextual_checkpoint.get("uses_test_data") is not False
        ):
            raise RuntimeError("Contextual metric checkpoint fold audit failed")
        contextual_model_type = (
            DualEncoderResidualContextMetric
            if contextual_checkpoint.get("model_class")
            == "dual_pretrained_residual"
            else DualEncoderContextMetric
        )
        contextual_model = contextual_model_type(
            **contextual_checkpoint["model_config"]
        )
        contextual_model.load_state_dict(contextual_checkpoint["state_dict"])
        contextual_model.eval()
    noisy_support_checkpoint_path = None
    noisy_support_model = None
    noisy_support_size = None
    if config["arbitration"].get("noisy_support_checkpoint"):
        noisy_support_checkpoint_path = root / str(
            config["arbitration"]["noisy_support_checkpoint"]
        ).format(fold=fold)
        noisy_support_checkpoint = torch.load(
            noisy_support_checkpoint_path, map_location="cpu", weights_only=True
        )
        if args.final_test:
            if (
                noisy_support_checkpoint.get("training_scope")
                != "all_official_development"
                or noisy_support_checkpoint.get("uses_test_data") is not False
                or noisy_support_checkpoint.get("uses_test_for_model_selection") is not False
            ):
                raise RuntimeError(
                    "Noisy-support ranker full-development audit failed"
                )
        elif (
            set(noisy_support_checkpoint["train_sessions"]) & set(validation_ids)
            or noisy_support_checkpoint.get("uses_validation_labels") is not False
            or noisy_support_checkpoint.get("uses_test_data") is not False
        ):
            raise RuntimeError("Noisy-support ranker fold audit failed")
        noisy_support_model = NoisySupportSpeakerRanker(
            **noisy_support_checkpoint["model_config"]
        )
        noisy_support_model.load_state_dict(noisy_support_checkpoint["state_dict"])
        noisy_support_model.eval()
        noisy_support_size = int(noisy_support_checkpoint["support_size"])
    overlap_guard = config.get("overlap_guard")
    streaming_overlap_dir = None
    offline_overlap_dir = None
    if overlap_guard:
        if args.final_test:
            streaming_overlap_dir = root / config["test_sources"]["streaming_overlap"]
            offline_overlap_dir = root / config["test_sources"]["offline_overlap"]
        else:
            streaming_overlap_dir = (
                root / "outputs" / overlap_guard["streaming_experiment"]
                / f"fold_{fold}" / "sessions"
            )
            offline_overlap_dir = (
                root / "outputs" / overlap_guard["offline_experiment"]
                / f"fold_{fold}" / "sessions"
            )
    target_activity_checkpoint_path = None
    target_activity_model = None
    if config["arbitration"].get("target_activity_checkpoint"):
        target_activity_checkpoint_path = root / str(
            config["arbitration"]["target_activity_checkpoint"]
        ).format(fold=fold)
        target_activity_checkpoint = torch.load(
            target_activity_checkpoint_path, map_location="cpu", weights_only=True
        )
        if (
            set(target_activity_checkpoint["train_sessions"]) & set(validation_ids)
            or target_activity_checkpoint.get("uses_validation_labels") is not False
            or target_activity_checkpoint.get("uses_test_data") is not False
        ):
            raise RuntimeError("Target-activity checkpoint fold audit failed")
        target_activity_model = TargetSpeakerActivityNet(
            **target_activity_checkpoint["model_config"]
        )
        target_activity_model.load_state_dict(target_activity_checkpoint["state_dict"])
        target_activity_model.eval()
    conflict_head = None
    conflict_checkpoint_path = None
    if config["arbitration"].get("conflict_checkpoint"):
        conflict_checkpoint_path = root / str(
            config["arbitration"]["conflict_checkpoint"]
        ).format(fold=fold)
        conflict_checkpoint = torch.load(
            conflict_checkpoint_path, map_location="cpu", weights_only=True
        )
        if (
            set(conflict_checkpoint["train_sessions"]) & set(validation_ids)
            or conflict_checkpoint.get("uses_validation_labels") is not False
            or conflict_checkpoint.get("uses_test_data") is not False
        ):
            raise RuntimeError("Conflict-head fold audit failed")
        conflict_head = PrototypeConflictHead(**conflict_checkpoint["model_config"])
        conflict_head.load_state_dict(conflict_checkpoint["state_dict"])
        conflict_head.eval()

    output_dir = (
        root / "outputs" / config["name"] / "test"
        if args.final_test
        else root / "outputs" / config["name"] / f"fold_{fold}{suffix}"
    )
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    all_segments: list[dict] = []
    audits: dict[str, dict] = {}
    started = time.time()
    for session_id in session_ids:
        output_path = session_dir / f"{session_id}.json"
        if output_path.exists() and not args.overwrite:
            saved = json.loads(output_path.read_text())
            all_segments.extend(saved["segments"])
            audits[session_id] = saved["arbitration_audit"]
            continue
        base_path = base_dir / f"{session_id}.json"
        complementary_path = complementary_dir / f"{session_id}.json"
        feature_path = feature_dir / f"{session_id}.pt"
        secondary_feature_path = (
            secondary_feature_dir / f"{session_id}.pt"
            if secondary_feature_dir is not None
            else None
        )
        streaming_overlap_path = (
            streaming_overlap_dir / f"{session_id}.json"
            if streaming_overlap_dir is not None
            else None
        )
        offline_overlap_path = (
            offline_overlap_dir / f"{session_id}.json"
            if offline_overlap_dir is not None
            else None
        )
        base_segments = json.loads(base_path.read_text())["segments"]
        complementary = json.loads(complementary_path.read_text())["segments"]
        streaming_diarization: list[list[float]] = []
        offline_diarization: list[list[float]] = []
        if streaming_overlap_path is not None and offline_overlap_path is not None:
            streaming_payload = json.loads(streaming_overlap_path.read_text())
            offline_payload = json.loads(offline_overlap_path.read_text())
            streaming_diarization = streaming_payload.get("raw_diarization") or []
            offline_diarization = offline_payload.get("diarization") or []
        acoustic = torch.load(feature_path, map_location="cpu", weights_only=True)
        if args.final_test:
            if (
                acoustic.get("uses_speaker_labels") is not False
                or acoustic.get("uses_test_data") is not True
                or acoustic.get("uses_test_for_training") is not False
                or acoustic.get("uses_test_for_model_selection") is not False
            ):
                raise RuntimeError("Final-test acoustic feature audit failed")
        elif acoustic.get("uses_speaker_labels") is not False or acoustic.get("uses_test_data") is not False:
            raise RuntimeError("Acoustic feature purity audit failed")
        rows = build_tokens(base_segments, complementary)
        base_speakers = sorted({row["base"] for row in rows})
        complementary_speakers = sorted({row["complementary"] for row in rows})
        contingency = np.zeros((len(complementary_speakers), len(base_speakers)), dtype=np.float64)
        for row in rows:
            contingency[complementary_speakers.index(row["complementary"]), base_speakers.index(row["base"])] += max(0.01, row["end"] - row["start"])
        left, right = linear_sum_assignment(-contingency)
        mapping = {
            complementary_speakers[i]: base_speakers[j]
            for i, j in zip(left.tolist(), right.tolist())
        }
        unmatched = sorted(set(complementary_speakers) - set(mapping))
        mapping.update(
            {speaker: f"novel_{index}" for index, speaker in enumerate(unmatched, start=1)}
        )
        for row in rows:
            row["mapped_complementary"] = mapping[row["complementary"]]
            row["final"] = row["base"]

        nodes = make_nodes(
            rows,
            float(config["node"]["max_seconds"]),
            float(config["node"]["max_gap_seconds"]),
        )
        embeddings = embed_nodes(
            nodes, acoustic["features"], acoustic["centers"], model
        )
        raw_node_complementary = dominant_node_labels(
            nodes, rows, "complementary"
        )
        base_node_labels = [str(node["base"]) for node in nodes]
        primary_acoustic_mapping = acoustic_role_mapping(
            base_node_labels,
            raw_node_complementary,
            embeddings,
            base_speakers,
            complementary_speakers,
        )
        (
            prototypes,
            prototype_supports,
            fallback_speakers,
            node_complementary,
            consensus_anchor_nodes,
        ) = build_prototype_bank(nodes, rows, embeddings, base_speakers)
        secondary_acoustic = None
        secondary_embeddings = None
        secondary_prototypes = None
        secondary_acoustic_mapping: dict[str, str] = {}
        purity_probabilities = None
        contextual_embeddings = None
        contextual_prototypes = None
        noisy_support_embeddings = None
        noisy_support_bank = None
        target_activity_probabilities = None
        if secondary_feature_path is not None and secondary_model is not None:
            secondary_acoustic = torch.load(
                secondary_feature_path, map_location="cpu", weights_only=True
            )
            if args.final_test:
                if (
                    secondary_acoustic.get("uses_speaker_labels") is not False
                    or secondary_acoustic.get("uses_test_data") is not True
                    or secondary_acoustic.get("uses_test_for_training") is not False
                    or secondary_acoustic.get("uses_test_for_model_selection") is not False
                ):
                    raise RuntimeError(
                        "Secondary final-test acoustic feature audit failed"
                    )
            elif (
                secondary_acoustic.get("uses_speaker_labels") is not False
                or secondary_acoustic.get("uses_test_data") is not False
            ):
                raise RuntimeError("Secondary acoustic feature purity audit failed")
            secondary_embeddings = embed_nodes(
                nodes,
                secondary_acoustic["features"],
                secondary_acoustic["centers"],
                secondary_model,
            )
            secondary_prototypes = build_prototype_bank(
                nodes, rows, secondary_embeddings, base_speakers
            )[0]
            secondary_acoustic_mapping = acoustic_role_mapping(
                base_node_labels,
                raw_node_complementary,
                secondary_embeddings,
                base_speakers,
                complementary_speakers,
            )
            if noisy_support_model is not None:
                noisy_support_embeddings = torch.cat(
                    [
                        F.normalize(embeddings, dim=-1),
                        F.normalize(secondary_embeddings, dim=-1),
                    ],
                    dim=-1,
                )
                noisy_support_bank = build_node_support_bank(
                    nodes,
                    rows,
                    noisy_support_embeddings,
                    base_speakers,
                    int(noisy_support_size),
                )
            if purity_model is not None:
                if not torch.allclose(
                    acoustic["centers"], secondary_acoustic["centers"], atol=1e-5
                ):
                    raise RuntimeError("Purity feature center mismatch")
                with torch.inference_mode():
                    purity_probabilities = torch.softmax(
                        purity_model(
                            acoustic["features"].float().unsqueeze(0),
                            secondary_acoustic["features"].float().unsqueeze(0),
                        ).squeeze(0),
                        dim=1,
                    )
            if contextual_model is not None:
                with torch.inference_mode():
                    contextual_frames, _ = contextual_model.encode(
                        acoustic["features"].float(),
                        secondary_acoustic["features"].float(),
                    )
                contextual_embeddings = pool_contextual_nodes(
                    nodes,
                    acoustic["centers"].float(),
                    contextual_frames.squeeze(0),
                )
                contextual_prototypes = build_prototype_bank(
                    nodes, rows, contextual_embeddings, base_speakers
                )[0]
            if target_activity_model is not None:
                labels = frame_labels(acoustic["centers"].float(), base_segments)
                primary_target_prototypes = []
                secondary_target_prototypes = []
                for speaker in base_speakers:
                    consensus_intervals = [
                        (float(node["start"]), float(node["end"]))
                        for node in nodes
                        if node["base"] == speaker
                        and all(
                            rows[token_index]["mapped_complementary"] == speaker
                            for token_index in node["token_indices"]
                        )
                    ]
                    selected = torch.tensor(
                        [
                            index
                            for index, center in enumerate(
                                acoustic["centers"].float().tolist()
                            )
                            if any(start <= center <= end for start, end in consensus_intervals)
                        ],
                        dtype=torch.long,
                    )
                    if not len(selected):
                        selected = torch.tensor(
                            [
                                index
                                for index, label in enumerate(labels)
                                if label == speaker
                            ],
                            dtype=torch.long,
                        )
                    if not len(selected):
                        selected = torch.arange(len(labels), dtype=torch.long)
                    primary_target_prototypes.append(
                        acoustic["features"][selected].float().mean(dim=0)
                    )
                    secondary_target_prototypes.append(
                        secondary_acoustic["features"][selected].float().mean(dim=0)
                    )
                with torch.inference_mode():
                    target_activity_probabilities = torch.sigmoid(
                        target_activity_model(
                            acoustic["features"].float(),
                            secondary_acoustic["features"].float(),
                            torch.stack(primary_target_prototypes),
                            torch.stack(secondary_target_prototypes),
                        ).squeeze(0)
                    )

        acoustic_changes = novel_changes = changed_tokens = purity_rejections = 0
        pretrained_overlap_rejections = 0
        role_link_rejections = 0
        contextual_metric_rejections = 0
        noisy_support_proposals = 0
        noisy_support_rejections = 0
        target_activity_rejections = 0
        accepted_changes: list[dict] = []
        probability_threshold = float(config["arbitration"]["same_speaker_probability"])
        for node_index, node in enumerate(nodes):
            alternative = node_complementary[node_index]
            if alternative == node["base"]:
                continue
            if alternative.startswith("novel_"):
                selected = alternative
                novel_changes += 1
            elif alternative in prototypes:
                if conflict_head is not None:
                    with torch.inference_mode():
                        conflict_logit = conflict_head(
                            embeddings[node_index].unsqueeze(0),
                            prototypes[node["base"]].unsqueeze(0),
                            prototypes[alternative].unsqueeze(0),
                            torch.tensor([float(prototype_supports[node["base"]])]),
                            torch.tensor([float(prototype_supports[alternative])]),
                        )
                    choose_alternative = bool((conflict_logit > 0).item())
                else:
                    current_cosine = embeddings[node_index] @ prototypes[node["base"]]
                    alternative_cosine = embeddings[node_index] @ prototypes[alternative]
                    current_probability = torch.sigmoid(
                        current_cosine * model.logit_scale.exp().clamp(max=30.0) + model.logit_bias
                    )
                    alternative_probability = torch.sigmoid(
                        alternative_cosine * model.logit_scale.exp().clamp(max=30.0) + model.logit_bias
                    )
                    choose_alternative = (
                        float(alternative_probability) > probability_threshold
                        and float(current_probability) < probability_threshold
                    )
                    if (
                        choose_alternative
                        and secondary_model is not None
                        and secondary_embeddings is not None
                        and secondary_prototypes is not None
                    ):
                        secondary_current_cosine = (
                            secondary_embeddings[node_index]
                            @ secondary_prototypes[node["base"]]
                        )
                        secondary_alternative_cosine = (
                            secondary_embeddings[node_index]
                            @ secondary_prototypes[alternative]
                        )
                        secondary_current_probability = torch.sigmoid(
                            secondary_current_cosine
                            * secondary_model.logit_scale.exp().clamp(max=30.0)
                            + secondary_model.logit_bias
                        )
                        secondary_alternative_probability = torch.sigmoid(
                            secondary_alternative_cosine
                            * secondary_model.logit_scale.exp().clamp(max=30.0)
                            + secondary_model.logit_bias
                        )
                        choose_alternative = (
                            float(secondary_alternative_probability)
                            > probability_threshold
                            and float(secondary_current_probability)
                            < probability_threshold
                        )
                    if (
                        noisy_support_model is not None
                        and noisy_support_embeddings is not None
                        and noisy_support_bank is not None
                    ):
                        with torch.inference_mode():
                            noisy_support_logit = noisy_support_model(
                                noisy_support_embeddings[node_index],
                                noisy_support_bank[node["base"]],
                                noisy_support_bank[alternative],
                            )
                        noisy_support_choice = bool(
                            (noisy_support_logit > 0).item()
                        )
                        if noisy_support_choice:
                            noisy_support_proposals += 1
                        combination = config["arbitration"].get(
                            "noisy_support_combination", "union"
                        )
                        if combination == "union":
                            choose_alternative = (
                                choose_alternative or noisy_support_choice
                            )
                        elif combination == "intersection":
                            if choose_alternative and not noisy_support_choice:
                                noisy_support_rejections += 1
                            choose_alternative = (
                                choose_alternative and noisy_support_choice
                            )
                        elif combination == "replace_static":
                            choose_alternative = noisy_support_choice
                        else:
                            raise ValueError(
                                f"Unknown noisy-support combination: {combination}"
                            )
                    contextual_mode = config["arbitration"].get(
                        "contextual_metric_mode", "guard"
                    )
                    if (
                        contextual_mode == "replace_static"
                        and contextual_model is not None
                        and contextual_embeddings is not None
                        and contextual_prototypes is not None
                    ):
                        contextual_current = contextual_model.pair_logits(
                            contextual_embeddings[node_index],
                            contextual_prototypes[node["base"]],
                        )
                        contextual_alternative = contextual_model.pair_logits(
                            contextual_embeddings[node_index],
                            contextual_prototypes[alternative],
                        )
                        choose_alternative = bool(
                            (contextual_alternative > 0).item()
                            and (contextual_current < 0).item()
                        )
                    if choose_alternative and purity_probabilities is not None:
                        centers = acoustic["centers"].float()
                        selected_frames = torch.nonzero(
                            (centers >= node["start"]) & (centers <= node["end"]),
                            as_tuple=False,
                        ).squeeze(1)
                        if not len(selected_frames):
                            midpoint = (node["start"] + node["end"]) / 2
                            selected_frames = torch.tensor(
                                [int(torch.argmin(torch.abs(centers - midpoint)))]
                            )
                        node_purity = purity_probabilities[selected_frames].mean(dim=0)
                        choose_alternative = int(torch.argmax(node_purity)) == 1
                        if not choose_alternative:
                            purity_rejections += 1
                    if (
                        choose_alternative
                        and contextual_mode == "guard"
                        and contextual_model is not None
                        and contextual_embeddings is not None
                        and contextual_prototypes is not None
                    ):
                        contextual_current = contextual_model.pair_logits(
                            contextual_embeddings[node_index],
                            contextual_prototypes[node["base"]],
                        )
                        contextual_alternative = contextual_model.pair_logits(
                            contextual_embeddings[node_index],
                            contextual_prototypes[alternative],
                        )
                        choose_alternative = bool(
                            (contextual_alternative > 0).item()
                            and (contextual_current < 0).item()
                        )
                        if not choose_alternative:
                            contextual_metric_rejections += 1
                    if choose_alternative and target_activity_probabilities is not None:
                        centers = acoustic["centers"].float()
                        selected_frames = torch.nonzero(
                            (centers >= node["start"]) & (centers <= node["end"]),
                            as_tuple=False,
                        ).squeeze(1)
                        if not len(selected_frames):
                            midpoint = (node["start"] + node["end"]) / 2
                            selected_frames = torch.tensor(
                                [int(torch.argmin(torch.abs(centers - midpoint)))]
                            )
                        activity = target_activity_probabilities[selected_frames].mean(dim=0)
                        current_activity = float(activity[base_speakers.index(node["base"])])
                        alternative_activity = float(activity[base_speakers.index(alternative)])
                        choose_alternative = (
                            alternative_activity > 0.5
                            and current_activity < 0.5
                        )
                        if not choose_alternative:
                            target_activity_rejections += 1
                    if choose_alternative and overlap_guard:
                        detected_overlap = has_multi_speaker_overlap(
                            float(node["start"]), float(node["end"]),
                            streaming_diarization,
                        ) or has_multi_speaker_overlap(
                            float(node["start"]), float(node["end"]),
                            offline_diarization,
                        )
                        if detected_overlap:
                            choose_alternative = False
                            pretrained_overlap_rejections += 1
                    if choose_alternative and config["arbitration"].get(
                        "require_dual_acoustic_role_link"
                    ):
                        raw_alternative = raw_node_complementary[node_index]
                        temporal_target = mapping.get(raw_alternative)
                        linked_by_both = (
                            temporal_target == alternative
                            and primary_acoustic_mapping.get(raw_alternative)
                            == alternative
                            and secondary_acoustic_mapping.get(raw_alternative)
                            == alternative
                        )
                        if not linked_by_both:
                            choose_alternative = False
                            role_link_rejections += 1
                if choose_alternative:
                    selected = alternative
                    acoustic_changes += 1
                else:
                    continue
            else:
                continue
            accepted_changes.append(
                {
                    "start": round(float(node["start"]), 3),
                    "end": round(float(node["end"]), 3),
                    "base": str(node["base"]),
                    "alternative": str(selected),
                    "tokens": [
                        rows[token_index]["token"]
                        for token_index in node["token_indices"]
                    ],
                }
            )
            for token_index in node["token_indices"]:
                rows[token_index]["final"] = selected
                changed_tokens += 1
        segments = rebuild_segments(session_id, base_segments, rows)
        audit = {
            "tokens": len(rows),
            "nodes": len(nodes),
            "base_speakers": len(base_speakers),
            "complementary_speakers": len(complementary_speakers),
            "unmatched_complementary_speakers": unmatched,
            "consensus_anchor_nodes": consensus_anchor_nodes,
            "fallback_prototype_speakers": fallback_speakers,
            "acoustic_changed_nodes": acoustic_changes,
            "novel_changed_nodes": novel_changes,
            "changed_tokens": changed_tokens,
            "purity_rejected_nodes": purity_rejections,
            "pretrained_overlap_rejected_nodes": pretrained_overlap_rejections,
            "role_link_rejected_nodes": role_link_rejections,
            "contextual_metric_rejected_nodes": contextual_metric_rejections,
            "noisy_support_proposed_nodes": noisy_support_proposals,
            "noisy_support_rejected_nodes": noisy_support_rejections,
            "target_activity_rejected_nodes": target_activity_rejections,
            "accepted_changes": accepted_changes,
            "temporal_role_mapping": mapping,
            "primary_acoustic_role_mapping": primary_acoustic_mapping,
            "secondary_acoustic_role_mapping": secondary_acoustic_mapping,
        }
        saved = {
            "session_id": session_id,
            "development_only": not args.final_test,
            "final_test_inference": args.final_test,
            "uses_validation_labels": False,
            "uses_test_data": args.final_test,
            "uses_test_for_training": False,
            "uses_test_for_model_selection": False,
            "base_source_sha256": sha256_file(base_path),
            "complementary_source_sha256": sha256_file(complementary_path),
            "feature_source_sha256": sha256_file(feature_path),
            "metric_checkpoint_sha256": sha256_file(checkpoint_path),
            "secondary_feature_source_sha256": (
                sha256_file(secondary_feature_path)
                if secondary_feature_path is not None
                else None
            ),
            "secondary_metric_checkpoint_sha256": (
                sha256_file(secondary_checkpoint_path)
                if secondary_checkpoint_path is not None
                else None
            ),
            "purity_checkpoint_sha256": (
                sha256_file(purity_checkpoint_path)
                if purity_checkpoint_path is not None
                else None
            ),
            "contextual_metric_checkpoint_sha256": (
                sha256_file(contextual_checkpoint_path)
                if contextual_checkpoint_path is not None
                else None
            ),
            "noisy_support_checkpoint_sha256": (
                sha256_file(noisy_support_checkpoint_path)
                if noisy_support_checkpoint_path is not None
                else None
            ),
            "target_activity_checkpoint_sha256": (
                sha256_file(target_activity_checkpoint_path)
                if target_activity_checkpoint_path is not None
                else None
            ),
            "streaming_overlap_source_sha256": (
                sha256_file(streaming_overlap_path)
                if streaming_overlap_path is not None
                else None
            ),
            "offline_overlap_source_sha256": (
                sha256_file(offline_overlap_path)
                if offline_overlap_path is not None
                else None
            ),
            "arbitration_audit": audit,
            "segments": segments,
        }
        output_path.write_text(
            json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        all_segments.extend(segments)
        audits[session_id] = audit
        print(json.dumps({"session": session_id, **audit}), flush=True)
    prediction_path = output_dir / "hyp.seglst.json"
    prediction_path.write_text(
        json.dumps(all_segments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validation = validate_segments(all_segments, wav_paths) if args.final_test else None
    submission_path = None
    if args.final_test:
        submission_path = (
            (root / args.submission).resolve()
            if args.submission is not None
            else root / "submissions" / f"{config['name']}.seglst.json"
        )
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        submission_path.write_bytes(prediction_path.read_bytes())
    metadata = {
        "experiment": config["name"],
        "fold": None if args.final_test else fold,
        "development_only": not args.final_test,
        "final_test_inference": args.final_test,
        "uses_validation_labels": False,
        "uses_test_data": args.final_test,
        "uses_test_for_training": False,
        "uses_test_for_model_selection": False,
        "session_ids": session_ids,
        "audits": audits,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "secondary_checkpoint_sha256": (
            sha256_file(secondary_checkpoint_path)
            if secondary_checkpoint_path is not None
            else None
        ),
        "purity_checkpoint_sha256": (
            sha256_file(purity_checkpoint_path)
            if purity_checkpoint_path is not None
            else None
        ),
        "contextual_metric_checkpoint_sha256": (
            sha256_file(contextual_checkpoint_path)
            if contextual_checkpoint_path is not None
            else None
        ),
        "noisy_support_checkpoint_sha256": (
            sha256_file(noisy_support_checkpoint_path)
            if noisy_support_checkpoint_path is not None
            else None
        ),
        "target_activity_checkpoint_sha256": (
            sha256_file(target_activity_checkpoint_path)
            if target_activity_checkpoint_path is not None
            else None
        ),
        "overlap_guard": overlap_guard,
        "conflict_checkpoint_sha256": (
            sha256_file(conflict_checkpoint_path)
            if conflict_checkpoint_path is not None
            else None
        ),
        "config_sha256": sha256_file(config_path),
        "prediction_sha256": sha256_file(prediction_path),
        "submission_sha256": (
            sha256_file(submission_path) if submission_path is not None else None
        ),
        "submission_audit": validation,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.final_test:
        print(
            json.dumps(
                {
                    "submission": str(submission_path),
                    "sha256": metadata["submission_sha256"],
                    **validation,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
