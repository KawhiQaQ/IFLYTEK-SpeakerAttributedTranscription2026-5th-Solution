#!/usr/bin/env python3
"""Run the full-development speaker metric graph on final-test caches."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from run_metric_graph_relabel import (
    cluster_nodes,
    group_tokens,
    make_nodes,
    node_features,
    sha256_file,
    token_rows,
)
from speaker_metric_model import SpeakerMetricAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    test_dir = (root / "data/test").resolve()
    if test_dir.name != "test":
        raise RuntimeError("Final-test path guard failed")
    session_ids = [path.stem for path in sorted((test_dir / "wav").glob("*.wav"))]
    if args.max_sessions:
        session_ids = session_ids[: args.max_sessions]
    feature_root = (
        root
        / "data/speaker_features_label_free"
        / config["feature_experiment"]
        / ("test_smoke" if args.max_sessions else "test")
    )
    feature_audit = json.loads((feature_root / "metadata.json").read_text())
    if (
        feature_audit.get("uses_speaker_labels") is not False
        or feature_audit.get("uses_test_for_training") is not False
        or feature_audit.get("uses_test_for_model_selection") is not False
    ):
        raise RuntimeError("Final-test label-free feature audit failed")
    checkpoint_path = root / config["metric_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("training_scope") != "all_official_development"
        or checkpoint.get("uses_test_data") is not False
        or checkpoint.get("uses_test_for_model_selection") is not False
    ):
        raise RuntimeError("Full-development checkpoint audit failed")
    model = SpeakerMetricAdapter(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    output_dir = (
        root
        / "outputs"
        / config["name"]
        / ("test_smoke" if args.max_sessions else "test")
    )
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    asr_dir = root / config["asr_source_dir"]
    track_dir = root / config["track_source_dir"]
    all_segments: list[dict] = []
    audits: dict[str, dict] = {}
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
        feature_payload = torch.load(
            feature_path, map_location="cpu", weights_only=True
        )
        if feature_payload.get("uses_speaker_labels") is not False:
            raise RuntimeError("Feature file contains speaker labels")
        rows = token_rows(asr, tracks, session_id)
        nodes = make_nodes(
            rows,
            float(config["node"]["max_seconds"]),
            float(config["node"]["max_gap_seconds"]),
        )
        features = node_features(
            nodes,
            feature_payload["features"].float(),
            feature_payload["centers"].float(),
        )
        with torch.inference_mode():
            embeddings = model.embed(features)
        labels, graph_audit = cluster_nodes(
            embeddings,
            model,
            float(config["clustering"]["same_speaker_posterior"]),
            int(config["clustering"]["min_speakers"]),
            int(config["clustering"]["max_speakers"]),
        )
        for node, label in zip(nodes, labels.tolist()):
            for token_index in node["token_indices"]:
                rows[token_index]["raw_speaker"] = label
        segments = group_tokens(
            rows, session_id, float(config["assignment"]["max_gap_seconds"])
        )
        graph_audit.update({"nodes": len(nodes), "segments": len(segments)})
        payload = {
            "session_id": session_id,
            "final_test_inference": True,
            "uses_test_for_training": False,
            "uses_test_for_model_selection": False,
            "wav_sha256": feature_payload["wav_sha256"],
            "asr_source_sha256": sha256_file(asr_path),
            "track_source_sha256": sha256_file(track_path),
            "feature_source_sha256": sha256_file(feature_path),
            "metric_checkpoint_sha256": sha256_file(checkpoint_path),
            "graph_audit": graph_audit,
            "segments": segments,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        all_segments.extend(segments)
        audits[session_id] = graph_audit
        print(
            json.dumps(
                {
                    "session": session_id,
                    "position": position,
                    "total": len(session_ids),
                    **graph_audit,
                }
            ),
            flush=True,
        )
    prediction_path = output_dir / "hyp.seglst.json"
    prediction_path.write_text(
        json.dumps(all_segments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "experiment": config["name"],
        "final_test_inference": True,
        "uses_test_for_training": False,
        "uses_test_for_model_selection": False,
        "session_ids": session_ids,
        "graph_audits": audits,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path),
        "prediction_sha256": sha256_file(prediction_path),
        "elapsed_seconds": round(time.time() - started, 2),
        "argv": sys.argv,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
