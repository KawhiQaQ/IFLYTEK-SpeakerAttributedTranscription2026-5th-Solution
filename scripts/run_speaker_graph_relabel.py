#!/usr/bin/env python3
"""Assign cached ASR tokens with a learned session-level speaker graph."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from run_sortformer_relabel import group_rows, sha256_file, token_rows
from speaker_graph_model import SpeakerGraphAttractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--session-id")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fold = int(config["fold"] if args.fold is None else args.fold)
    feature_root = root / "data" / "speaker_graph" / config["name"] / f"fold_{fold}"
    metadata = json.loads((feature_root / "metadata.json").read_text())
    session_ids = metadata["subsets"]["val"]["sessions"]
    if args.session_id:
        if args.session_id not in session_ids:
            raise RuntimeError("Session is outside the frozen validation fold")
        session_ids = [args.session_id]

    checkpoint_path = root / str(config["output_dir"]).format(fold=fold) / "speaker_graph.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if set(checkpoint["train_sessions"]) & set(session_ids):
        raise RuntimeError("Checkpoint leakage into validation sessions")
    if checkpoint.get("uses_test_data") is not False:
        raise RuntimeError("Checkpoint audit failed")
    model = SpeakerGraphAttractor(
        input_dimension=int(checkpoint["input_dimension"]), **checkpoint["model_config"]
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.cuda().eval()

    asr_root = (
        root
        / "outputs"
        / config["assignment"]["asr_source_experiment"]
        / f"fold_{fold}"
        / "sessions"
    )
    suffix = "_diagnostic" if args.session_id else ""
    output_dir = root / "outputs" / config["name"] / f"fold_{fold}{suffix}"
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    all_segments: list[dict] = []
    predicted_counts: dict[str, int] = {}
    started = time.time()

    for session_id in session_ids:
        output_path = session_dir / f"{session_id}.json"
        if output_path.exists() and not args.overwrite:
            saved = json.loads(output_path.read_text())
            all_segments.extend(saved["segments"])
            predicted_counts[session_id] = int(saved["predicted_speaker_count"])
            continue
        feature_payload = torch.load(
            feature_root / "val" / f"{session_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        features = feature_payload["features"].cuda()
        centers = feature_payload["centers"].float()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            outputs = model(features)
        predicted_count = int(outputs["count_logits"].argmax(dim=-1).item()) + 2
        existence = outputs["existence_logits"].squeeze(0)
        selected = existence.topk(predicted_count).indices
        activity = outputs["activity_logits"].squeeze(0).float().cpu()

        asr_path = asr_root / f"{session_id}.json"
        asr = json.loads(asr_path.read_text())
        rows = token_rows(asr["raw_result"][0], session_id)
        for row in rows:
            inside = (centers >= float(row["start"])) & (centers <= float(row["end"]))
            if inside.any():
                scores = activity[inside][:, selected.cpu()].mean(dim=0)
            else:
                nearest = (centers - (float(row["start"]) + float(row["end"])) / 2).abs().argmin()
                scores = activity[nearest, selected.cpu()]
            row["raw_speaker"] = int(selected[int(scores.argmax())].item())
        segments = group_rows(
            rows, session_id, float(config["assignment"]["max_gap_seconds"])
        )
        payload = {
            "session_id": session_id,
            "development_only": True,
            "uses_validation_labels": False,
            "uses_test_data": False,
            "predicted_speaker_count": predicted_count,
            "segments": segments,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        all_segments.extend(segments)
        predicted_counts[session_id] = predicted_count
        print(
            json.dumps(
                {"session": session_id, "speakers": predicted_count, "segments": len(segments)}
            ),
            flush=True,
        )

    prediction_path = output_dir / "hyp.seglst.json"
    prediction_path.write_text(
        json.dumps(all_segments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    run_metadata = {
        "experiment": config["name"],
        "fold": fold,
        "development_only": True,
        "uses_validation_labels": False,
        "uses_test_data": False,
        "session_ids": session_ids,
        "predicted_speaker_counts": predicted_counts,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prediction_sha256": sha256_file(prediction_path),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_metadata), flush=True)


if __name__ == "__main__":
    main()
