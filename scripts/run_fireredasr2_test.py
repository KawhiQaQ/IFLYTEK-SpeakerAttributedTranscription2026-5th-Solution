#!/usr/bin/env python3
"""Run a frozen FireRedASR2 AED or LLM branch on official test audio."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from run_fireredasr2_llm import align_to_source_timestamps
from run_sortformer_relabel import normalize_tokens, sha256_file


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
    inference = config["inference"]
    model_type = str(model_config["type"])
    if model_type not in {"aed", "llm"}:
        raise RuntimeError(f"Unsupported FireRed test branch: {model_type}")
    repo_path = (root / model_config["repo"]).resolve()
    model_path = (root / model_config["directory"]).resolve()
    checkpoint_path = model_path / "model.pth.tar"
    if not repo_path.is_dir() or not checkpoint_path.is_file():
        raise FileNotFoundError("FireRedASR2 repository or checkpoint is missing")
    sys.path.insert(0, str(repo_path))
    from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

    if model_type == "aed":
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
    else:
        asr_config = FireRedAsr2Config(
            use_gpu=bool(inference["use_gpu"]),
            use_half=bool(inference["use_half"]),
            beam_size=int(inference["beam_size"]),
            decode_max_len=int(inference["decode_max_len"]),
            decode_min_len=int(inference["decode_min_len"]),
            repetition_penalty=float(inference["repetition_penalty"]),
            llm_length_penalty=float(inference["llm_length_penalty"]),
            temperature=float(inference["temperature"]),
        )
    model = FireRedAsr2.from_pretrained(model_type, str(model_path), asr_config)

    output_name = f"test_fireredasr2_{model_type}"
    if args.max_sessions is not None:
        output_name += "_smoke"
    output_dir = root / "outputs" / output_name
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    timestamp_dir = root / "outputs" / (
        "test_fireredasr2_aed_smoke" if args.max_sessions is not None else "test_fireredasr2_aed"
    ) / "sessions"
    checkpoint_sha256 = sha256_file(checkpoint_path)
    session_hashes = {}
    started = time.time()
    for position, wav_path in enumerate(wav_paths, start=1):
        session_id = wav_path.stem
        output_path = session_dir / f"{session_id}.json"
        if output_path.exists() and not args.overwrite:
            session_hashes[session_id] = sha256_file(output_path)
            print(json.dumps({"session": session_id, "status": "resumed"}), flush=True)
            continue
        session_started = time.time()
        result = model.transcribe([session_id], [str(wav_path)])[0]
        if model_type == "aed":
            native = result.get("timestamp")
            if not isinstance(native, list) or not native:
                raise RuntimeError(f"Missing AED timestamps for {session_id}")
            tokens, timestamps = [], []
            for item in native:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    raise RuntimeError(f"Malformed timestamp for {session_id}: {item!r}")
                normalized = normalize_tokens(str(item[0]))
                tokens.extend(normalized)
                timestamps.extend(
                    [[float(item[1]) * 1000, float(item[2]) * 1000]] * len(normalized)
                )
            alignment = None
        else:
            tokens = normalize_tokens(str(result.get("text", "")))
            timestamp_path = timestamp_dir / f"{session_id}.json"
            source_result = json.loads(timestamp_path.read_text(encoding="utf-8"))["raw_result"][0]
            timestamps, alignment = align_to_source_timestamps(source_result, tokens, session_id)
        if not tokens or len(tokens) != len(timestamps):
            raise RuntimeError(f"FireRed output contract failed for {session_id}")
        payload = {
            "session_id": session_id,
            "final_test_inference": True,
            "uses_test_for_training": False,
            "uses_test_for_model_selection": False,
            "wav_sha256": sha256_file(wav_path),
            "model_checkpoint_sha256": checkpoint_sha256,
            "alignment_audit": alignment,
            "raw_result": [
                {
                    "text": " ".join(tokens),
                    "timestamp": timestamps,
                    "generative_text": result.get("text", ""),
                    "session_confidence": result.get("confidence"),
                    "rtf": result.get("rtf"),
                }
            ],
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        session_hashes[session_id] = sha256_file(output_path)
        print(
            json.dumps(
                {
                    "session": session_id,
                    "position": position,
                    "total": len(wav_paths),
                    "tokens": len(tokens),
                    "elapsed_seconds": round(time.time() - session_started, 2),
                }
            ),
            flush=True,
        )

    metadata = {
        "final_test_inference": True,
        "uses_test_for_training": False,
        "uses_test_for_model_selection": False,
        "branch": model_type,
        "smoke_only": args.max_sessions is not None,
        "session_ids": [path.stem for path in wav_paths],
        "session_hashes": session_hashes,
        "model_checkpoint_sha256": checkpoint_sha256,
        "config_sha256": sha256_file(config_path),
        "max_cuda_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
