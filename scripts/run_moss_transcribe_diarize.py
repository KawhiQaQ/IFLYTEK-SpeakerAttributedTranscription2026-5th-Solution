#!/usr/bin/env python3
"""Run public MOSS-Transcribe-Diarize on frozen development folds.

The model jointly generates transcript text, segment timestamps, and anonymous
speaker labels.  Inference never reads development references or test audio.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import wave
from pathlib import Path

import torch
import yaml

from run_sortformer_relabel import normalize_tokens


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def verified_weight(model_path: Path, artifact: dict) -> tuple[Path, str]:
    weight = model_path / str(artifact["filename"])
    if not weight.is_file():
        raise FileNotFoundError(weight)
    expected_size = int(artifact["size_bytes"])
    if weight.stat().st_size != expected_size:
        raise RuntimeError(f"Incomplete model weight: {weight.stat().st_size} != {expected_size}")
    actual_sha = sha256_file(weight)
    if actual_sha != str(artifact["sha256"]):
        raise RuntimeError(f"Model SHA256 mismatch: {actual_sha}")
    return weight, actual_sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--session-id", action="append")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fold = int(config["fold"] if args.fold is None else args.fold)
    split_dir = root / "data" / "splits" / f"fold_{fold}"
    validation_ids = (split_dir / "val_sessions.txt").read_text(encoding="utf-8").split()
    session_ids = validation_ids
    if args.session_id:
        unknown = set(args.session_id) - set(validation_ids)
        if unknown:
            raise RuntimeError(f"Sessions outside frozen validation fold: {sorted(unknown)}")
        session_ids = args.session_id

    model_cfg = config["model"]
    repo_path = (root / model_cfg["repo"]).resolve()
    model_path = (root / model_cfg["checkpoint"]).resolve()
    if not (repo_path / "moss_transcribe_diarize" / "__init__.py").is_file():
        raise FileNotFoundError(f"Missing pinned MOSS source repository: {repo_path}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing MOSS checkpoint: {model_path}")
    weight_path, weight_sha = verified_weight(model_path, model_cfg["artifact"])

    sys.path.insert(0, str(repo_path))
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        generate_transcription,
    )

    device = torch.device(str(model_cfg["device"]))
    dtype = getattr(torch, str(model_cfg["dtype"]))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), trust_remote_code=True, dtype="auto", local_files_only=True,
    ).to(dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True,
    )

    suffix = "_diagnostic" if args.session_id else ""
    output_dir = root / "outputs" / config["name"] / f"fold_{fold}{suffix}"
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    all_segments: list[dict] = []
    started = time.time()

    for position, session_id in enumerate(session_ids, start=1):
        output_path = session_dir / f"{session_id}.json"
        if output_path.exists() and not args.overwrite:
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            all_segments.extend(saved["segments"])
            print(json.dumps({"session": session_id, "status": "resumed"}), flush=True)
            continue

        wav_path = root / "data" / "dev" / "wav" / f"{session_id}.wav"
        if wav_path.parent.name != "wav" or wav_path.parent.parent.name != "dev":
            raise RuntimeError("Development input guard failed")
        if not wav_path.is_file():
            raise FileNotFoundError(wav_path)
        duration = duration_seconds(wav_path)
        session_started = time.time()
        result = generate_transcription(
            model,
            processor,
            build_transcription_messages(wav_path),
            max_length=int(config["inference"]["max_length"]),
            max_new_tokens=int(config["inference"]["max_new_tokens"]),
            do_sample=False,
            device=device,
            dtype=dtype,
        )
        parsed = parse_transcript(str(result["text"]))
        if not parsed:
            raise RuntimeError(f"MOSS produced no parseable segments for {session_id}")

        speaker_map: dict[str, str] = {}
        segments: list[dict] = []
        token_rows: list[dict] = []
        for item in parsed:
            tokens = normalize_tokens(str(item.text))
            if not tokens:
                continue
            raw_speaker = str(item.speaker)
            if raw_speaker not in speaker_map:
                speaker_map[raw_speaker] = f"spk{len(speaker_map) + 1}"
            start = max(0.0, min(duration, float(item.start)))
            end = max(0.0, min(duration, float(item.end)))
            if end <= start:
                continue
            segments.append({
                "session_id": session_id,
                "speaker": speaker_map[raw_speaker],
                "start_time": round(start, 2),
                "end_time": round(end, 2),
                "words": " ".join(tokens),
            })
            width = (end - start) / len(tokens)
            for index, token in enumerate(tokens):
                token_rows.append({
                    "token": token,
                    "start": start + index * width,
                    "end": end if index == len(tokens) - 1 else start + (index + 1) * width,
                })
        if not segments:
            raise RuntimeError(f"MOSS produced no valid segments for {session_id}")
        raw_result = [{
            "text": " ".join(row["token"] for row in token_rows),
            "timestamp": [
                [round(row["start"] * 1000, 3), round(row["end"] * 1000, 3)]
                for row in token_rows
            ],
        }]
        payload = {
            "session_id": session_id,
            "development_only": True,
            "uses_validation_labels": False,
            "uses_test_data": False,
            "wav_sha256": sha256_file(wav_path),
            "model_weight_sha256": weight_sha,
            "raw_text": str(result["text"]),
            "generated_tokens": int(result["generated_tokens"]),
            "speaker_map": speaker_map,
            "raw_result": raw_result,
            "segments": segments,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        all_segments.extend(segments)
        print(json.dumps({
            "session": session_id,
            "position": position,
            "total": len(session_ids),
            "segments": len(segments),
            "speakers": len(speaker_map),
            "tokens": len(token_rows),
            "generated_tokens": int(result["generated_tokens"]),
            "elapsed_seconds": round(time.time() - session_started, 2),
        }, ensure_ascii=False), flush=True)

    prediction_path = output_dir / "hyp.seglst.json"
    prediction_path.write_text(
        json.dumps(all_segments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "experiment": config["name"],
        "fold": fold,
        "diagnostic_only": bool(args.session_id),
        "development_only": True,
        "uses_validation_labels": False,
        "uses_test_data": False,
        "joint_text_time_speaker_generation": True,
        "session_ids": session_ids,
        "model_weight": str(weight_path),
        "model_weight_sha256": weight_sha,
        "source_commit": str(model_cfg["source_commit"]),
        "versions": {
            "transformers": importlib.metadata.version("transformers"),
            "torch": torch.__version__,
        },
        "config_sha256": sha256_file(config_path),
        "prediction_sha256": sha256_file(prediction_path),
        "max_cuda_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "elapsed_seconds": round(time.time() - started, 2),
        "argv": sys.argv,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
