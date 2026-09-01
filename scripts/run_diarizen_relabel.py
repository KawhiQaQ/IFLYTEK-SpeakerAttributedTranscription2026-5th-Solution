#!/usr/bin/env python3
"""Relabel cached Qwen tokens with overlap-aware DiariZen speaker tracks.

The program only reads frozen validation audio and cached label-free ASR output.
It never reads reference segments or speaker labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

from run_sortformer_relabel import group_rows, interval_overlap, sha256_file, token_rows


def install_runtime_compatibility() -> None:
    """Bridge pyannote 3.1 to the newer PyTorch runtime in this workspace."""
    if not hasattr(np, "NaN"):
        np.NaN = np.nan  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = namedtuple(  # type: ignore[attr-defined]
            "AudioMetaData",
            "sample_rate num_frames num_channels bits_per_sample encoding",
        )
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]

    def soundfile_load(path, *args, **kwargs):
        import soundfile as sf

        values, sample_rate = sf.read(
            str(path), dtype="float32", always_2d=True
        )
        return torch.from_numpy(values.T.copy()), sample_rate

    torchaudio.load = soundfile_load  # type: ignore[assignment]

    original_load = torch.load

    def trusted_checkpoint_load(*args, **kwargs):
        # PyTorch >=2.6 changed this default. Both accepted checkpoints are
        # pinned by SHA-256 below and come from the public model repositories.
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = trusted_checkpoint_load  # type: ignore[assignment]


def turn_distance(interval: tuple[float, float], turn: list[object]) -> float:
    if interval[1] < float(turn[0]):
        return float(turn[0]) - interval[1]
    if float(turn[1]) < interval[0]:
        return interval[0] - float(turn[1])
    return 0.0


def assign_tracks(rows: list[dict], turns: list[list[object]]) -> dict[str, int]:
    """Assign each timestamped ASR token to the strongest temporal track."""
    if not turns:
        raise RuntimeError("DiariZen returned no speaker tracks")
    label_to_index = {
        label: index for index, label in enumerate(sorted({str(t[2]) for t in turns}))
    }
    uncovered = 0
    ambiguous = 0
    for row in rows:
        interval = (float(row["start"]), float(row["end"]))
        overlaps = [
            interval_overlap(interval, (float(turn[0]), float(turn[1])))
            for turn in turns
        ]
        best_overlap = max(overlaps)
        if best_overlap <= 0.0:
            uncovered += 1
            chosen = min(
                range(len(turns)),
                key=lambda i: (turn_distance(interval, turns[i]), str(turns[i][2])),
            )
        else:
            candidates = [
                i for i, overlap in enumerate(overlaps) if overlap == best_overlap
            ]
            ambiguous += int(len({str(turns[i][2]) for i in candidates}) > 1)
            midpoint = (interval[0] + interval[1]) / 2
            chosen = min(
                candidates,
                key=lambda i: (
                    abs(midpoint - (float(turns[i][0]) + float(turns[i][1])) / 2),
                    str(turns[i][2]),
                ),
            )
        row["raw_speaker"] = label_to_index[str(turns[chosen][2])]
    return {"uncovered_tokens": uncovered, "ambiguous_tokens": ambiguous}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", help="Fold number, or 'full' for test deployment")
    parser.add_argument("--session-id")
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fold_value = config["fold"] if args.fold is None else args.fold
    fold: int | str = "full" if str(fold_value) == "full" else int(fold_value)
    scope = str(config.get("scope", "dev"))
    audio_dir = (root / "data" / scope).resolve()
    if scope == "test":
        frozen_ids = sorted(path.stem for path in (audio_dir / "wav").glob("*.wav"))
        if fold != "full" or len(frozen_ids) != 394:
            raise RuntimeError("Exact full-fit test deployment guard failed")
    else:
        if audio_dir.name != "dev" or fold == "full":
            raise RuntimeError("Development-only input guard failed")
        frozen_ids = (
            root / "data" / "splits" / f"fold_{fold}" / "val_sessions.txt"
        ).read_text(encoding="utf-8").split()
    if args.session_id is not None:
        if args.session_id not in frozen_ids:
            raise RuntimeError("Requested session is not in the frozen validation fold")
        session_ids = [args.session_id]
    else:
        session_ids = frozen_ids[: args.max_sessions]
    if not session_ids:
        raise RuntimeError("No validation sessions selected")

    diar_cfg = config["diarizen"]
    source_root = (root / diar_cfg["source_root"]).resolve()
    model_dir = (
        root / str(diar_cfg["model_dir"]).format(fold=fold)
    ).resolve()
    model_checkpoint = model_dir / "pytorch_model.bin"
    embedding_checkpoint = (root / diar_cfg["embedding_checkpoint"]).resolve()
    if sha256_file(model_checkpoint) != diar_cfg["model_sha256"]:
        raise RuntimeError("DiariZen checkpoint SHA-256 mismatch")
    if sha256_file(embedding_checkpoint) != diar_cfg["embedding_sha256"]:
        raise RuntimeError("Speaker embedding checkpoint SHA-256 mismatch")
    sys.path.insert(0, str(source_root))
    dependency_namespace = root / "models/runtime/diarizen_pydeps/pyannote"
    if dependency_namespace.is_dir():
        import pyannote

        pyannote.__path__.append(str(dependency_namespace))
    install_runtime_compatibility()
    from diarizen.pipelines.inference import DiariZenPipeline

    asr_dir = (
        root / config["test_asr_root"]
        if scope == "test"
        else root / "outputs" / config["asr_source_experiment"] / f"fold_{fold}" / "sessions"
    )
    wav_paths = [audio_dir / "wav" / f"{sid}.wav" for sid in session_ids]
    asr_paths = [asr_dir / f"{sid}.json" for sid in session_ids]
    if not all(path.is_file() for path in wav_paths + asr_paths):
        raise FileNotFoundError("Validation audio or cached Qwen output is incomplete")

    suffix = "_diagnostic" if args.session_id or args.max_sessions else ""
    output_dir = root / "outputs" / config["name"] / (
        f"test{suffix}" if scope == "test" else f"fold_{fold}{suffix}"
    )
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    pipeline = DiariZenPipeline(
        diarizen_hub=model_dir,
        embedding_model=str(embedding_checkpoint),
    )
    started = time.time()
    all_segments: list[dict] = []
    predicted_counts: dict[str, int] = {}
    for position, (session_id, wav_path, asr_path) in enumerate(
        zip(session_ids, wav_paths, asr_paths), start=1
    ):
        result_path = session_dir / f"{session_id}.json"
        if result_path.exists() and not args.overwrite:
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            all_segments.extend(saved["segments"])
            predicted_counts[session_id] = int(saved["predicted_speaker_count"])
            continue
        session_started = time.time()
        annotation = pipeline(str(wav_path), sess_name=session_id)
        turns = [
            [float(turn.start), float(turn.end), str(speaker)]
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        source = json.loads(asr_path.read_text(encoding="utf-8"))
        rows = token_rows(source["raw_result"][0], session_id)
        assignment_audit = assign_tracks(rows, turns)
        segments = group_rows(
            rows, session_id, float(config["assignment"]["max_gap_seconds"])
        )
        predicted_count = len({str(turn[2]) for turn in turns})
        predicted_counts[session_id] = predicted_count
        payload = {
            "session_id": session_id,
            "development_only": scope != "test",
            "final_test_inference": scope == "test",
            "inference_only": True,
            "uses_validation_labels": False,
            "uses_test_data": scope == "test",
            "uses_test_for_training_or_selection": False,
            "wav_sha256": sha256_file(wav_path),
            "asr_source_sha256": sha256_file(asr_path),
            "predicted_speaker_count": predicted_count,
            "diarization": turns,
            "assignment_audit": assignment_audit,
            "segments": segments,
        }
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        all_segments.extend(segments)
        print(
            json.dumps(
                {
                    "session": session_id,
                    "position": position,
                    "total": len(session_ids),
                    "speakers": predicted_count,
                    **assignment_audit,
                    "elapsed_seconds": round(time.time() - session_started, 2),
                }
            ),
            flush=True,
        )

    prediction_path = output_dir / "hyp.seglst.json"
    prediction_path.write_text(
        json.dumps(all_segments, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "experiment": config["name"],
        "fold": fold,
        "development_only": scope != "test",
        "final_test_inference": scope == "test",
        "inference_only": True,
        "uses_validation_labels": False,
        "uses_test_data": scope == "test",
        "uses_test_for_training_or_selection": False,
        "session_ids": session_ids,
        "predicted_speaker_counts": predicted_counts,
        "model_sha256": diar_cfg["model_sha256"],
        "embedding_sha256": diar_cfg["embedding_sha256"],
        "prediction_sha256": sha256_file(prediction_path),
        "config_sha256": sha256_file(config_path),
        "max_cuda_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
