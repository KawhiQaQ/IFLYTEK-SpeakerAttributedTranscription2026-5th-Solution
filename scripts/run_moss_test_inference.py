#!/usr/bin/env python3
"""Run frozen MOSS-Transcribe-Diarize once on official test audio."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoProcessor

from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    generate_transcription,
)

from run_moss_transcribe_diarize import duration_seconds, normalize_tokens
from run_sortformer_relabel import sha256_file


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
    model_path = (root / model_config["checkpoint"]).resolve()
    device = torch.device(model_config["device"])
    dtype = getattr(torch, model_config["dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), trust_remote_code=True, dtype="auto"
    ).to(dtype=dtype, device=device).eval()
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)

    output_name = "test_moss_transcribe_diarize"
    if args.max_sessions is not None:
        output_name += "_smoke"
    output_dir = root / "outputs" / output_name
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    all_segments: list[dict] = []
    session_hashes = {}
    started = time.time()
    for position, wav_path in enumerate(wav_paths, start=1):
        session_id = wav_path.stem
        session_path = session_dir / f"{session_id}.json"
        if session_path.exists() and not args.overwrite:
            payload = json.loads(session_path.read_text(encoding="utf-8"))
            all_segments.extend(payload["segments"])
            session_hashes[session_id] = sha256_file(session_path)
            print(json.dumps({"session": session_id, "status": "resumed"}), flush=True)
            continue
        session_started = time.time()
        duration = duration_seconds(wav_path)
        result = generate_transcription(
            model,
            processor,
            build_transcription_messages(wav_path),
            max_new_tokens=int(model_config["max_new_tokens"]),
            do_sample=False,
            device=device,
            dtype=dtype,
        )
        segments = []
        for segment in parse_transcript(result["text"]):
            tokens = normalize_tokens(segment.text)
            start = max(0.0, min(duration, float(segment.start)))
            end = max(0.0, min(duration, float(segment.end)))
            if tokens and end > start:
                segments.append(
                    {
                        "session_id": session_id,
                        "speaker": segment.speaker.lower(),
                        "start_time": round(start, 2),
                        "end_time": round(end, 2),
                        "words": " ".join(tokens),
                    }
                )
        if not segments:
            raise RuntimeError(f"MOSS produced no parseable segments for {session_id}")
        payload = {
            "session_id": session_id,
            "final_test_inference": True,
            "uses_test_for_training": False,
            "uses_test_for_model_selection": False,
            "wav_sha256": sha256_file(wav_path),
            "raw_text": result["text"],
            "prompt_tokens": result["prompt_len"],
            "generated_tokens": result["generated_tokens"],
            "segments": segments,
        }
        session_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        all_segments.extend(segments)
        session_hashes[session_id] = sha256_file(session_path)
        print(
            json.dumps(
                {
                    "session": session_id,
                    "position": position,
                    "total": len(wav_paths),
                    "segments": len(segments),
                    "speakers": len({row["speaker"] for row in segments}),
                    "tokens": sum(len(row["words"].split()) for row in segments),
                    "elapsed_seconds": round(time.time() - session_started, 2),
                }
            ),
            flush=True,
        )

    prediction_path = output_dir / "hyp.seglst.json"
    prediction_path.write_text(
        json.dumps(all_segments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "final_test_inference": True,
        "uses_test_for_training": False,
        "uses_test_for_model_selection": False,
        "smoke_only": args.max_sessions is not None,
        "session_ids": [path.stem for path in wav_paths],
        "session_hashes": session_hashes,
        "config_sha256": sha256_file(config_path),
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("moss-transcribe-diarize", "transformers", "torch")
        },
        "prediction_sha256": sha256_file(prediction_path),
        "elapsed_seconds": round(time.time() - started, 2),
        "argv": sys.argv,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
