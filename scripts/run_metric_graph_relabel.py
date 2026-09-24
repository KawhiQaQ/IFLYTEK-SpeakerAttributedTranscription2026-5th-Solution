#!/usr/bin/env python3
"""Relabel FireRed tokens with a learned, fold-pure speaker-affinity graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.cluster import AgglomerativeClustering

from speaker_metric_model import SpeakerMetricAdapter


SPECIAL_TAG = re.compile(r"<\|[^|]+\|>")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?|[\u3400-\u4dbf\u4e00-\u9fff]", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", SPECIAL_TAG.sub(" ", text)).lower())


def overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def midpoint_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    midpoint = (left[0] + left[1]) / 2
    return 0.0 if right[0] <= midpoint <= right[1] else min(abs(midpoint - right[0]), abs(midpoint - right[1]))


def initial_track(interval: tuple[float, float], tracks: list[dict]) -> str:
    return str(max(tracks, key=lambda row: (
        overlap(interval, (float(row["start_time"]), float(row["end_time"]))),
        -midpoint_distance(interval, (float(row["start_time"]), float(row["end_time"]))),
    ))["speaker"])


def token_rows(asr: dict, tracks: list[dict], session_id: str) -> list[dict]:
    tokens = normalize_tokens(str(asr["text"]))
    timestamps = asr.get("timestamp")
    if not isinstance(timestamps, list) or len(tokens) != len(timestamps):
        raise RuntimeError(f"Token/timestamp mismatch for {session_id}")
    rows = []
    for token, timestamp in zip(tokens, timestamps):
        start, end = float(timestamp[0]) / 1000.0, float(timestamp[1]) / 1000.0
        if end <= start:
            continue
        rows.append({"token": token, "start": start, "end": end, "initial_track": initial_track((start, end), tracks)})
    return rows


def make_nodes(rows: list[dict], max_seconds: float, max_gap: float) -> list[dict]:
    nodes: list[dict] = []
    for token_index, row in enumerate(rows):
        new = (
            not nodes
            or row["initial_track"] != nodes[-1]["initial_track"]
            or row["start"] - nodes[-1]["end"] > max_gap
            or row["end"] - nodes[-1]["start"] > max_seconds
        )
        if new:
            nodes.append({"start": row["start"], "end": row["end"], "initial_track": row["initial_track"], "token_indices": [token_index]})
        else:
            nodes[-1]["end"] = row["end"]
            nodes[-1]["token_indices"].append(token_index)
    return nodes


def node_features(nodes: list[dict], features: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    output = []
    for node in nodes:
        selected = torch.nonzero((centers >= node["start"]) & (centers <= node["end"]), as_tuple=False).squeeze(1)
        if not len(selected):
            midpoint = (node["start"] + node["end"]) / 2
            selected = torch.tensor([int(torch.argmin(torch.abs(centers - midpoint)))])
        output.append(features[selected].mean(dim=0))
    return torch.stack(output)


def cluster_nodes(
    embeddings: torch.Tensor,
    model: SpeakerMetricAdapter,
    posterior_threshold: float,
    min_speakers: int,
    max_speakers: int,
) -> tuple[np.ndarray, dict]:
    count = len(embeddings)
    if count < min_speakers:
        return np.zeros(count, dtype=np.int64), {"raw_cluster_count": 1, "bounded_cluster_count": 1}
    cosine = embeddings @ embeddings.T
    logits = cosine * model.logit_scale.exp().clamp(max=30.0) + model.logit_bias
    posterior = torch.sigmoid(logits).detach().float().cpu().numpy()
    distance = np.clip(1.0 - posterior, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    threshold_distance = 1.0 - posterior_threshold
    raw = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold_distance,
        metric="precomputed", linkage="average",
    ).fit_predict(distance)
    raw_count = len(set(raw.tolist()))
    bounded_count = min(max(raw_count, min_speakers), min(max_speakers, count))
    labels = raw if bounded_count == raw_count else AgglomerativeClustering(
        n_clusters=bounded_count, metric="precomputed", linkage="average"
    ).fit_predict(distance)
    return labels.astype(np.int64), {
        "raw_cluster_count": raw_count, "bounded_cluster_count": bounded_count,
        "mean_same_speaker_posterior": float(posterior[np.triu_indices(count, 1)].mean()) if count > 1 else 1.0,
    }


def group_tokens(rows: list[dict], session_id: str, max_gap: float) -> list[dict]:
    speaker_map: dict[int, str] = {}
    grouped: list[dict] = []
    for row in rows:
        raw_speaker = int(row["raw_speaker"])
        if raw_speaker not in speaker_map:
            speaker_map[raw_speaker] = f"spk{len(speaker_map) + 1}"
        speaker = speaker_map[raw_speaker]
        if not grouped or grouped[-1]["speaker"] != speaker or row["start"] - grouped[-1]["end_time"] > max_gap:
            grouped.append({"session_id": session_id, "speaker": speaker, "start_time": row["start"], "end_time": row["end"], "tokens": [row["token"]]})
        else:
            grouped[-1]["end_time"] = row["end"]
            grouped[-1]["tokens"].append(row["token"])
    return [{"session_id": row["session_id"], "speaker": row["speaker"], "start_time": round(row["start_time"], 2), "end_time": round(row["end_time"], 2), "words": " ".join(row["tokens"])} for row in grouped]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--raw-metric-ablation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    fold = int(config["fold"] if args.fold is None else args.fold)
    split_dir = root / "data" / "splits" / f"fold_{fold}"
    session_ids = (split_dir / "val_sessions.txt").read_text().split()
    if args.max_sessions:
        session_ids = session_ids[:args.max_sessions]
    feature_root = root / "data" / "speaker_features_label_free" / config["feature_experiment"] / f"fold_{fold}" / "val"
    feature_audit = json.loads((feature_root / "metadata.json").read_text())
    if feature_audit.get("uses_speaker_labels") is not False or feature_audit.get("uses_test_data") is not False:
        raise RuntimeError("Label-free feature audit failed")
    checkpoint_path = root / str(config["metric_checkpoint"]).format(fold=fold)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if set(checkpoint["train_sessions"]) & set(session_ids) or checkpoint.get("uses_test_data") is not False:
        raise RuntimeError("Metric checkpoint fold audit failed")
    model = SpeakerMetricAdapter(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    suffix = "_raw_metric_ablation" if args.raw_metric_ablation else ""
    if args.max_sessions:
        suffix += "_smoke"
    output_dir = root / "outputs" / f"{config['name']}{suffix}" / f"fold_{fold}"
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    asr_dir = root / "outputs" / config["asr_source_experiment"] / f"fold_{fold}" / "sessions"
    track_dir = root / "outputs" / config["track_source_experiment"] / f"fold_{fold}" / "sessions"
    all_segments, audits = [], {}
    started = time.time()
    for position, session_id in enumerate(session_ids, start=1):
        output_path = session_dir / f"{session_id}.json"
        if output_path.exists() and not args.overwrite:
            payload = json.loads(output_path.read_text())
            all_segments.extend(payload["segments"])
            audits[session_id] = payload["graph_audit"]
            continue
        asr_path = asr_dir / f"{session_id}.json"
        track_path = track_dir / f"{session_id}.json"
        feature_path = feature_root / f"{session_id}.pt"
        asr = json.loads(asr_path.read_text())["raw_result"][0]
        tracks = json.loads(track_path.read_text())["segments"]
        feature_payload = torch.load(feature_path, map_location="cpu", weights_only=True)
        if feature_payload.get("uses_speaker_labels") is not False:
            raise RuntimeError("Feature file contains speaker labels")
        rows = token_rows(asr, tracks, session_id)
        nodes = make_nodes(rows, float(config["node"]["max_seconds"]), float(config["node"]["max_gap_seconds"]))
        features = node_features(nodes, feature_payload["features"].float(), feature_payload["centers"].float())
        with torch.inference_mode():
            if args.raw_metric_ablation:
                multiscale = torch.nn.functional.normalize(features.reshape(len(features), int(config["model"]["scales"]), -1), dim=-1)
                embeddings = torch.nn.functional.normalize(multiscale.mean(dim=1), dim=-1)
            else:
                embeddings = model.embed(features)
        labels, graph_audit = cluster_nodes(
            embeddings, model, float(config["clustering"]["same_speaker_posterior"]),
            int(config["clustering"]["min_speakers"]), int(config["clustering"]["max_speakers"]),
        )
        for node, label in zip(nodes, labels.tolist()):
            for token_index in node["token_indices"]:
                rows[token_index]["raw_speaker"] = label
        segments = group_tokens(rows, session_id, float(config["assignment"]["max_gap_seconds"]))
        graph_audit.update({"nodes": len(nodes), "segments": len(segments), "raw_metric_ablation": args.raw_metric_ablation})
        payload = {
            "session_id": session_id, "development_only": True,
            "uses_validation_labels": False, "uses_test_data": False,
            "wav_sha256": feature_payload["wav_sha256"],
            "asr_source_sha256": sha256_file(asr_path), "track_source_sha256": sha256_file(track_path),
            "feature_source_sha256": sha256_file(feature_path), "metric_checkpoint_sha256": sha256_file(checkpoint_path),
            "graph_audit": graph_audit, "segments": segments,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        all_segments.extend(segments)
        audits[session_id] = graph_audit
        print(json.dumps({"session": session_id, "position": position, "total": len(session_ids), **graph_audit}), flush=True)
    prediction_path = output_dir / "hyp.seglst.json"
    prediction_path.write_text(json.dumps(all_segments, ensure_ascii=False, indent=2) + "\n")
    metadata = {
        "experiment": config["name"], "fold": fold, "development_only": True,
        "uses_validation_labels": False, "uses_test_data": False,
        "raw_metric_ablation": args.raw_metric_ablation, "session_ids": session_ids,
        "graph_audits": audits, "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path), "prediction_sha256": sha256_file(prediction_path),
        "elapsed_seconds": round(time.time() - started, 2), "argv": sys.argv,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
