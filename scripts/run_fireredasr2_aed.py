#!/usr/bin/env python3
"""Cache label-free FireRedASR2-AED text and native word timestamps."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from firered_aed_lora import load_adapter
from run_sortformer_relabel import normalize_tokens, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--session-id", action="append")
    parser.add_argument("--full-dev", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fold = int(config["fold"] if args.fold is None else args.fold)
    validation_ids = (
        sorted(path.stem for path in (root / "data/dev/wav").glob("*.wav"))
        if args.full_dev
        else (
            root / "data" / "splits" / f"fold_{fold}" / "val_sessions.txt"
        ).read_text().split()
    )
    session_ids = validation_ids
    suffix = ""
    if args.session_id:
        if args.full_dev:
            raise RuntimeError("--session-id and --full-dev are mutually exclusive")
        if not set(args.session_id) <= set(validation_ids):
            raise RuntimeError("Session is outside the frozen validation fold")
        session_ids = args.session_id
        suffix = "_diagnostic"

    model_config = config["model"]
    inference = config["inference"]
    repo_path = (root / model_config["repo"]).resolve()
    model_path = (root / model_config["directory"]).resolve()
    checkpoint_path = model_path / model_config["checkpoint"]
    if not repo_path.is_dir() or not checkpoint_path.is_file():
        raise FileNotFoundError("FireRedASR2 repository or public checkpoint is missing")
    sys.path.insert(0, str(repo_path))
    from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

    asr_config = FireRedAsr2Config(
        use_gpu=bool(inference["use_gpu"]),
        use_half=bool(inference["use_half"]),
        beam_size=int(inference["beam_size"]),
        nbest=int(inference["nbest"]),
        decode_max_len=int(inference["decode_max_len"]),
        softmax_smoothing=float(inference["softmax_smoothing"]),
        aed_length_penalty=float(inference["aed_length_penalty"]),
        eos_penalty=float(inference["eos_penalty"]),
        return_timestamp=bool(inference["return_timestamp"]),
    )
    model = FireRedAsr2.from_pretrained(
        str(model_config["type"]), str(model_path), asr_config
    )
    adapter_path = None
    adapter_metadata = None
    adapter_targets = None
    if model_config.get("adapter_path"):
        adapter_path = root / str(model_config["adapter_path"]).format(fold=fold)
        adapter_targets, adapter_metadata = load_adapter(model.model, adapter_path)
        validation_ids_set = set(validation_ids)
        if (
            int(adapter_metadata.get("fold", -1)) != fold
            or set(adapter_metadata.get("train_sessions", [])) & validation_ids_set
            or adapter_metadata.get("uses_validation_labels") is not False
            or adapter_metadata.get("uses_test_data") is not False
            or adapter_metadata.get("completed_schedule") is not True
        ):
            raise RuntimeError("FireRed adapter fold-provenance audit failed")
        model.model.eval()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    output_dir = (
        root / "outputs" / config["name"]
        if args.full_dev
        else root / "outputs" / config["name"] / f"fold_{fold}{suffix}"
    )
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for position, session_id in enumerate(session_ids, start=1):
        output_path = session_dir / f"{session_id}.json"
        if output_path.exists() and not args.overwrite:
            continue
        reuse_path = None
        if args.full_dev and config.get("reuse_fold_experiment"):
            for reuse_fold in range(5):
                candidate = root / (
                    f"outputs/{config['reuse_fold_experiment']}/fold_{reuse_fold}/sessions/{session_id}.json"
                )
                if candidate.is_file():
                    reuse_path = candidate
                    break
        if reuse_path is not None and not args.overwrite:
            output_path.write_bytes(reuse_path.read_bytes())
            print(json.dumps({"session": session_id, "status": "reused_fold_cache"}), flush=True)
            continue
        wav_path = root / "data" / "dev" / "wav" / f"{session_id}.wav"
        session_started = time.time()
        result = model.transcribe([session_id], [str(wav_path)])[0]
        timestamps = result.get("timestamp")
        if not isinstance(timestamps, list) or not timestamps:
            raise RuntimeError(f"Missing native timestamps for {session_id}")
        tokens: list[str] = []
        aligned: list[list[float]] = []
        confidence = float(result.get("confidence", 0.0))
        confidences: list[float] = []
        for item in timestamps:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                raise RuntimeError(f"Malformed timestamp for {session_id}: {item!r}")
            normalized = normalize_tokens(str(item[0]))
            for token in normalized:
                tokens.append(token)
                aligned.append([float(item[1]) * 1000, float(item[2]) * 1000])
                confidences.append(confidence)
        if not tokens or len(tokens) != len(aligned):
            raise RuntimeError(f"Timestamp normalization failed for {session_id}")
        payload = {
            "session_id": session_id,
            "development_only": True,
            "uses_validation_labels": False,
            "uses_test_data": False,
            "wav_sha256": sha256_file(wav_path),
            "model_checkpoint_sha256": checkpoint_sha256,
            "raw_result": [
                {
                    "text": " ".join(tokens),
                    "timestamp": aligned,
                    "confidence": confidences,
                    "generative_text": result.get("text", ""),
                    "session_confidence": confidence,
                    "rtf": result.get("rtf"),
                }
            ],
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "session": session_id,
                    "position": position,
                    "total": len(session_ids),
                    "tokens": len(tokens),
                    "confidence": confidence,
                    "elapsed_seconds": round(time.time() - session_started, 2),
                }
            ),
            flush=True,
        )

    metadata = {
        "experiment": config["name"],
        "fold": fold,
        "scope": "full_dev" if args.full_dev else "validation",
        "development_only": True,
        "uses_validation_labels": False,
        "uses_test_data": False,
        "session_ids": session_ids,
        "reuse_fold_experiment": config.get("reuse_fold_experiment"),
        "model_path": str(model_path),
        "model_checkpoint_sha256": checkpoint_sha256,
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "adapter_sha256": sha256_file(adapter_path) if adapter_path is not None else None,
        "adapter_target_modules": adapter_targets,
        "config_sha256": sha256_file(config_path),
        "max_cuda_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
