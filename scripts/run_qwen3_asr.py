#!/usr/bin/env python3
"""Run Qwen3-ASR with forced alignment on one frozen development fold."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import sys
import time
import unicodedata
import wave
from pathlib import Path

import torch
import yaml


SPECIAL_TAG = re.compile(r"<\|[^|]+\|>")
TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+(?:'[a-z0-9]+)?|[\u3400-\u4dbf\u4e00-\u9fff]",
    flags=re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", SPECIAL_TAG.sub(" ", text)).lower()
    return TOKEN_PATTERN.findall(text)


def duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def aligned_tokens(result, duration: float, session_id: str) -> list[dict]:
    items = getattr(result, "time_stamps", None)
    if items is None:
        items = getattr(result, "items", None)
    if not items:
        raise RuntimeError(f"Qwen forced aligner returned no timestamps for {session_id}")
    rows: list[dict] = []
    for item in items:
        tokens = normalize_tokens(str(item.text))
        if not tokens:
            continue
        start = max(0.0, min(duration, float(item.start_time)))
        end = max(0.0, min(duration, float(item.end_time)))
        if end <= start:
            if start >= duration:
                start = max(0.0, duration - 0.02)
                end = duration
            else:
                end = min(duration, start + 0.02)
        width = (end - start) / len(tokens)
        for index, token in enumerate(tokens):
            token_start = start + index * width
            token_end = end if index == len(tokens) - 1 else start + (index + 1) * width
            rows.append({"token": token, "start": token_start, "end": token_end})
    if not rows:
        raise RuntimeError(f"Qwen forced aligner returned no usable tokens for {session_id}")
    return rows


def raw_result_from_tokens(rows: list[dict]) -> dict:
    return {
        "text": " ".join(row["token"] for row in rows),
        "timestamp": [
            [round(1000 * row["start"], 3), round(1000 * row["end"], 3)]
            for row in rows
        ],
    }


def single_speaker_segments(rows: list[dict], session_id: str, max_gap: float = 0.8) -> list[dict]:
    grouped: list[dict] = []
    for row in rows:
        if not grouped or row["start"] - grouped[-1]["end_time"] > max_gap:
            grouped.append(
                {
                    "session_id": session_id,
                    "speaker": "spk1",
                    "start_time": row["start"],
                    "end_time": row["end"],
                    "tokens": [row["token"]],
                }
            )
        else:
            grouped[-1]["end_time"] = row["end"]
            grouped[-1]["tokens"].append(row["token"])
    return [
        {
            "session_id": row["session_id"],
            "speaker": row["speaker"],
            "start_time": round(row["start_time"], 2),
            "end_time": round(row["end_time"], 2),
            "words": " ".join(row["tokens"]),
        }
        for row in grouped
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--session-id", action="append")
    parser.add_argument("--full-dev", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fold = int(config["fold"] if args.fold is None else args.fold)
    dev_dir = (root / "data" / "dev").resolve()
    if dev_dir.name != "dev":
        raise RuntimeError("Development-only guard failed")
    split_dir = root / "data" / "splits" / f"fold_{fold}"
    validation_ids = (split_dir / "val_sessions.txt").read_text(encoding="utf-8").split()
    session_ids = (
        sorted(path.stem for path in (dev_dir / "wav").glob("*.wav"))
        if args.full_dev
        else validation_ids
    )
    if args.session_id:
        if args.full_dev or args.max_sessions is not None:
            raise RuntimeError("--session-id cannot be combined with --full-dev or --max-sessions")
        unknown = set(args.session_id) - set(validation_ids)
        if unknown:
            raise RuntimeError(f"Sessions outside frozen validation fold: {sorted(unknown)}")
        session_ids = args.session_id
    if args.max_sessions is not None:
        session_ids = session_ids[: args.max_sessions]
    wav_paths = [dev_dir / "wav" / f"{session_id}.wav" for session_id in session_ids]
    if not session_ids or not all(
        path.is_file() and path.parent == dev_dir / "wav" for path in wav_paths
    ):
        raise RuntimeError("Validation input guard failed")

    model_config = config["model"]
    asr_path = (root / model_config["checkpoint"]).resolve()
    aligner_path = (root / model_config["forced_aligner"]).resolve()
    if not asr_path.is_dir() or not aligner_path.is_dir():
        raise FileNotFoundError(f"Missing Qwen model or aligner: {asr_path}, {aligner_path}")

    suffix = (
        "_diagnostic" if args.session_id
        else "_smoke" if args.max_sessions is not None
        else ""
    )
    output_dir = (
        root / "outputs" / config["name"]
        if args.full_dev
        else root / "outputs" / config["name"] / f"fold_{fold}{suffix}"
    )
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
    all_segments: list[dict] = []
    for position, (session_id, wav_path) in enumerate(zip(session_ids, wav_paths), start=1):
        session_path = session_dir / f"{session_id}.json"
        if session_path.exists() and not args.overwrite:
            saved = json.loads(session_path.read_text(encoding="utf-8"))
            all_segments.extend(saved["segments"])
            print(json.dumps({"session": session_id, "status": "resumed"}), flush=True)
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
            shutil.copy2(reuse_path, session_path)
            saved = json.loads(session_path.read_text(encoding="utf-8"))
            all_segments.extend(saved["segments"])
            print(json.dumps({"session": session_id, "status": "reused_fold_cache"}), flush=True)
            continue
        session_started = time.time()
        duration = duration_seconds(wav_path)
        results = model.transcribe(
            audio=str(wav_path),
            language=model_config["language"],
            return_time_stamps=True,
        )
        if len(results) != 1:
            raise RuntimeError(f"Unexpected Qwen result count for {session_id}: {len(results)}")
        token_rows = aligned_tokens(results[0], duration, session_id)
        raw_result = raw_result_from_tokens(token_rows)
        if len(normalize_tokens(raw_result["text"])) != len(raw_result["timestamp"]):
            raise RuntimeError(f"Token/timestamp contract failed for {session_id}")
        segments = single_speaker_segments(token_rows, session_id)
        payload = {
            "session_id": session_id,
            "wav_sha256": sha256_file(wav_path),
            "raw_text": str(results[0].text),
            "language": str(results[0].language),
            "raw_result": [raw_result],
            "segments": segments,
        }
        session_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        all_segments.extend(segments)
        print(
            json.dumps(
                {
                    "session": session_id,
                    "position": position,
                    "total": len(session_ids),
                    "tokens": len(token_rows),
                    "elapsed_seconds": round(time.time() - session_started, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    prediction_path = output_dir / "hyp.seglst.json"
    prediction_path.write_text(
        json.dumps(all_segments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "experiment": config["name"],
        "fold": fold,
        "smoke_only": args.max_sessions is not None,
        "diagnostic_only": bool(args.session_id),
        "scope": "full_dev" if args.full_dev else "validation",
        "development_only": True,
        "uses_validation_labels": False,
        "session_ids": session_ids,
        "session_list_sha256": (
            None if args.full_dev else sha256_file(split_dir / "val_sessions.txt")
        ),
        "reuse_fold_experiment": config.get("reuse_fold_experiment"),
        "cv_manifest_sha256": sha256_file(root / "configs" / "cv" / "folds_v1.csv"),
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
        "prediction_sha256": sha256_file(prediction_path),
        "argv": sys.argv,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
