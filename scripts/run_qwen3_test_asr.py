#!/usr/bin/env python3
"""Run frozen Qwen3-ASR once on the official test audio for final inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from run_qwen3_asr import (
    aligned_tokens,
    duration_seconds,
    normalize_tokens,
    raw_result_from_tokens,
    sha256_file,
)


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
    test_dir = (root / "data" / "test").resolve()
    if test_dir.name != "test":
        raise RuntimeError("Final-test guard failed")
    wav_paths = sorted((test_dir / "wav").glob("*.wav"))
    if args.max_sessions is not None:
        wav_paths = wav_paths[: args.max_sessions]
    if not wav_paths or not all(path.parent == test_dir / "wav" for path in wav_paths):
        raise RuntimeError("Test audio discovery failed")

    model_config = config["model"]
    asr_path = (root / model_config["checkpoint"]).resolve()
    aligner_path = (root / model_config["forced_aligner"]).resolve()
    if not asr_path.is_dir() or not aligner_path.is_dir():
        raise FileNotFoundError("Missing frozen Qwen ASR or forced aligner")
    output_name = "test_asr_qwen3_1.7b"
    if args.max_sessions is not None:
        output_name += "_smoke"
    output_dir = root / "outputs" / output_name
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)

    from qwen_asr import Qwen3ASRModel

    dtype = getattr(torch, str(model_config["dtype"]))
    model = Qwen3ASRModel.from_pretrained(
        str(asr_path),
        dtype=dtype,
        device_map=model_config["device_map"],
        forced_aligner=str(aligner_path),
        forced_aligner_kwargs={
            "dtype": dtype,
            "device_map": model_config["device_map"],
        },
        max_inference_batch_size=int(model_config["max_inference_batch_size"]),
        max_new_tokens=int(model_config["max_new_tokens"]),
    )

    started = time.time()
    session_hashes: dict[str, str] = {}
    for position, wav_path in enumerate(wav_paths, start=1):
        session_id = wav_path.stem
        session_path = session_dir / f"{session_id}.json"
        if session_path.exists() and not args.overwrite:
            session_hashes[session_id] = sha256_file(session_path)
            print(json.dumps({"session": session_id, "status": "resumed"}), flush=True)
            continue
        session_started = time.time()
        duration = duration_seconds(wav_path)
        results = model.transcribe(
            audio=str(wav_path),
            language=model_config["language"],
            return_time_stamps=True,
        )
        if len(results) != 1:
            raise RuntimeError(f"Unexpected Qwen result count for {session_id}")
        rows = aligned_tokens(results[0], duration, session_id)
        raw_result = raw_result_from_tokens(rows)
        if len(normalize_tokens(raw_result["text"])) != len(raw_result["timestamp"]):
            raise RuntimeError(f"Token/timestamp contract failed for {session_id}")
        payload = {
            "session_id": session_id,
            "wav_sha256": sha256_file(wav_path),
            "raw_text": str(results[0].text),
            "language": str(results[0].language),
            "raw_result": [raw_result],
        }
        session_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        session_hashes[session_id] = sha256_file(session_path)
        print(
            json.dumps(
                {
                    "session": session_id,
                    "position": position,
                    "total": len(wav_paths),
                    "tokens": len(rows),
                    "elapsed_seconds": round(time.time() - session_started, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    metadata = {
        "final_test_inference": True,
        "uses_test_for_training": False,
        "uses_test_for_model_selection": False,
        "smoke_only": args.max_sessions is not None,
        "session_ids": [path.stem for path in wav_paths],
        "session_hashes": session_hashes,
        "config": config,
        "config_sha256": sha256_file(config_path),
        "model_files": {
            "asr_config_sha256": sha256_file(asr_path / "config.json"),
            "aligner_config_sha256": sha256_file(aligner_path / "config.json"),
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("qwen-asr", "transformers", "torch")
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "argv": sys.argv,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
