#!/usr/bin/env python3
"""Extract label-free session features for supervised speaker-count prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


SEED = 20260827


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pad(values: list[float], size: int) -> list[float]:
    return (values + [0.0] * size)[:size]


def summarize_embeddings(embeddings: np.ndarray, auto_labels: np.ndarray) -> dict[str, float]:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    similarity = cosine_similarity(embeddings)
    off_diagonal = similarity[~np.eye(len(similarity), dtype=bool)]
    if not len(off_diagonal):
        off_diagonal = np.array([1.0])
    nearest = np.partition(similarity, -2, axis=1)[:, -2] if len(similarity) > 1 else np.ones(1)

    features: dict[str, float] = {}
    for name, value in zip(
        ("min", "p10", "p25", "median", "p75", "p90", "max"),
        np.quantile(off_diagonal, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]),
    ):
        features[f"cosine_{name}"] = float(value)
    features["cosine_mean"] = float(off_diagonal.mean())
    features["cosine_std"] = float(off_diagonal.std())
    for name, value in zip(
        ("p10", "median", "p90"), np.quantile(nearest, [0.1, 0.5, 0.9])
    ):
        features[f"nearest_cosine_{name}"] = float(value)
    features["nearest_cosine_mean"] = float(nearest.mean())
    features["nearest_cosine_std"] = float(nearest.std())

    # The same local-neighbour graph family used by spectral diarization, exposed
    # as fixed-size eigengap features rather than hand-tuned count thresholds.
    adjacency = similarity.copy()
    np.fill_diagonal(adjacency, 0.0)
    keep = min(6, max(len(adjacency) - 1, 1))
    for index in range(len(adjacency)):
        remove = np.argsort(adjacency[index])[: max(len(adjacency) - keep, 0)]
        adjacency[index, remove] = 0.0
    adjacency = 0.5 * (adjacency + adjacency.T)
    laplacian = np.diag(np.abs(adjacency).sum(axis=1)) - adjacency
    eigenvalues = np.linalg.eigvalsh(laplacian)
    first_eigenvalues = pad([float(value) for value in eigenvalues[:8]], 8)
    eigengaps = pad([float(value) for value in np.diff(eigenvalues[:9])], 8)
    for index, value in enumerate(first_eigenvalues):
        features[f"laplacian_eigenvalue_{index}"] = value
    for index, value in enumerate(eigengaps):
        features[f"laplacian_eigengap_{index + 1}"] = value

    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    explained = singular_values**2
    explained = explained / max(float(explained.sum()), 1e-12)
    for index, value in enumerate(pad([float(value) for value in explained[:8]], 8)):
        features[f"embedding_variance_ratio_{index + 1}"] = value

    unique, counts = np.unique(auto_labels, return_counts=True)
    cluster_ratios = sorted((counts / counts.sum()).tolist(), reverse=True)
    features["auto_speaker_count"] = float(len(unique))
    for index, value in enumerate(pad([float(value) for value in cluster_ratios], 6)):
        features[f"auto_cluster_ratio_{index + 1}"] = value
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    input_dir = (root / "data" / args.split).resolve()
    if input_dir.name != args.split:
        raise RuntimeError("Input split guard failed")
    wav_paths = sorted((input_dir / "wav").glob("*.wav"))
    if not wav_paths or not all(path.parent == input_dir / "wav" for path in wav_paths):
        raise RuntimeError("Audio discovery failed")

    output_name = (
        "speaker_count_features_v1.json"
        if args.split == "dev"
        else "test_speaker_count_features_v1.json"
    )
    output_path = root / "reports" / output_name
    if output_path.exists() and not args.overwrite:
        print(json.dumps({"status": "exists", "path": str(output_path)}))
        return

    toolkit_dir = root / "third_party" / "3D-Speaker"
    sys.path.insert(0, str(toolkit_dir))
    os.environ.setdefault("MODELSCOPE_CACHE", str(root / "models" / "modelscope"))
    from speakerlab.bin.infer_diarization import Diarization3Dspeaker
    from speakerlab.utils.fileio import load_audio

    diarizer = Diarization3Dspeaker(
        device="cuda:0",
        include_overlap=False,
        model_cache_dir=str(root / "models" / "modelscope" / "3dspeaker"),
    )

    started = time.time()
    sessions = {}
    feature_names = None
    for position, wav_path in enumerate(wav_paths, start=1):
        np.random.seed(SEED)
        wav = load_audio(str(wav_path), None, diarizer.fs)
        vad = diarizer.do_vad(wav)
        chunks = [chunk for start, end in vad for chunk in diarizer.chunk(start, end)]
        if len(chunks) < 2:
            raise RuntimeError(f"Too few speech chunks for {wav_path.stem}: {len(chunks)}")
        embeddings = diarizer.do_emb_extraction(chunks, wav)
        auto_labels = np.asarray(diarizer.cluster(embeddings, speaker_num=None))
        features = summarize_embeddings(embeddings, auto_labels)
        with wave.open(str(wav_path), "rb") as handle:
            duration = handle.getnframes() / handle.getframerate()
        vad_duration = sum(float(end) - float(start) for start, end in vad)
        features.update(
            {
                "audio_duration": float(duration),
                "vad_duration": float(vad_duration),
                "speech_ratio": float(vad_duration / max(duration, 1e-12)),
                "num_vad_regions": float(len(vad)),
                "num_embedding_chunks": float(len(chunks)),
            }
        )
        ordered = dict(sorted(features.items()))
        if feature_names is None:
            feature_names = list(ordered)
        elif list(ordered) != feature_names:
            raise RuntimeError("Feature schema drift")
        if not all(np.isfinite(value) for value in ordered.values()):
            raise RuntimeError(f"Non-finite feature for {wav_path.stem}")
        sessions[wav_path.stem] = ordered
        print(
            json.dumps(
                {
                    "session": wav_path.stem,
                    "position": position,
                    "total": len(wav_paths),
                    "chunks": len(chunks),
                    "auto_count": int(len(np.unique(auto_labels))),
                }
            ),
            flush=True,
        )

    payload = {
        "split": args.split,
        "development_only": args.split == "dev",
        "final_test_inference": args.split == "test",
        "uses_transcripts": False,
        "uses_speaker_labels": False,
        "seed": SEED,
        "feature_names": feature_names,
        "sessions": sessions,
        "elapsed_seconds": round(time.time() - started, 2),
        "wav_hashes": {path.stem: sha256_file(path) for path in wav_paths},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"path": str(output_path), "sessions": len(sessions)}))


if __name__ == "__main__":
    main()
