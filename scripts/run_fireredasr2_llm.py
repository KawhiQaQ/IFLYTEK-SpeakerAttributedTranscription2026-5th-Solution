#!/usr/bin/env python3
"""Cache FireRedASR2-LLM text on the independent AED native time axis."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from run_sortformer_relabel import normalize_tokens, sha256_file


def edit_alignment(source: list[str], target: list[str]) -> tuple[list[int | None], int]:
    """Map target tokens monotonically to source tokens with unit edit costs."""
    n, m = len(source), len(target)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0], back[i][0] = i, 1
    for j in range(1, m + 1):
        costs[0][j], back[0][j] = j, 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = costs[i - 1][j - 1] + (source[i - 1] != target[j - 1])
            deletion = costs[i - 1][j] + 1
            insertion = costs[i][j - 1] + 1
            choice = min((diagonal, 0), (deletion, 1), (insertion, 2))
            costs[i][j], back[i][j] = choice
    mapping: list[int | None] = [None] * m
    i, j = n, m
    while i or j:
        step = back[i][j]
        if step == 0:
            mapping[j - 1] = i - 1
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return mapping, costs[n][m]


def align_to_source_timestamps(
    source_result: dict, target_tokens: list[str], session_id: str
) -> tuple[list[list[float]], dict]:
    source_tokens = normalize_tokens(str(source_result["text"]))
    source_timestamps = source_result.get("timestamp")
    if not isinstance(source_timestamps, list) or len(source_tokens) != len(source_timestamps):
        raise RuntimeError(f"Invalid AED timestamp source for {session_id}")
    mapping, distance = edit_alignment(source_tokens, target_tokens)
    timestamps = []
    for target_index, source_index in enumerate(mapping):
        if source_index is None:
            source_index = min(
                len(source_tokens) - 1,
                max(0, round((target_index + 0.5) * len(source_tokens) / len(target_tokens) - 0.5)),
            )
        start, end = map(float, source_timestamps[source_index][:2])
        if end <= start:
            end = start + 1.0
        timestamps.append([start, end])
    return timestamps, {
        "source_tokens": len(source_tokens),
        "target_tokens": len(target_tokens),
        "edit_distance": int(distance),
        "normalized_edit_rate": distance / max(len(source_tokens), len(target_tokens), 1),
        "inserted_target_tokens": sum(index is None for index in mapping),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--session-id")
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
        if args.session_id not in validation_ids:
            raise RuntimeError("Session is outside the frozen validation fold")
        session_ids = [args.session_id]
        suffix = "_diagnostic"

    model_config = config["model"]
    inference = config["inference"]
    repo_path = (root / model_config["repo"]).resolve()
    model_path = (root / model_config["directory"]).resolve()
    if not repo_path.is_dir() or not (model_path / "model.pth.tar").is_file():
        raise FileNotFoundError("FireRedASR2-LLM repository or checkpoint is missing")
    sys.path.insert(0, str(repo_path))
    from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

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
    model = FireRedAsr2.from_pretrained(str(model_config["type"]), str(model_path), asr_config)
    adapter_sha256 = sha256_file(model_path / "model.pth.tar")
    timestamp_dir = (
        root / "outputs" / config["timestamp_source_experiment"] / "sessions"
        if args.full_dev
        else root
        / "outputs"
        / config["timestamp_source_experiment"]
        / f"fold_{fold}{suffix}"
        / "sessions"
    )
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
        timestamp_path = timestamp_dir / f"{session_id}.json"
        timestamp_payload = json.loads(timestamp_path.read_text())
        source_result = timestamp_payload["raw_result"][0]
        session_started = time.time()
        result = model.transcribe([session_id], [str(wav_path)])[0]
        tokens = normalize_tokens(str(result.get("text", "")))
        if not tokens:
            raise RuntimeError(f"FireRedASR2-LLM returned no tokens for {session_id}")
        timestamps, alignment = align_to_source_timestamps(source_result, tokens, session_id)
        payload = {
            "session_id": session_id,
            "development_only": True,
            "uses_validation_labels": False,
            "uses_test_data": False,
            "wav_sha256": sha256_file(wav_path),
            "model_adapter_sha256": adapter_sha256,
            "timestamp_source_sha256": sha256_file(timestamp_path),
            "alignment_audit": alignment,
            "raw_result": [
                {
                    "text": " ".join(tokens),
                    "timestamp": timestamps,
                    "generative_text": result.get("text", ""),
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
                    "alignment": alignment,
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
        "model_adapter_sha256": adapter_sha256,
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
