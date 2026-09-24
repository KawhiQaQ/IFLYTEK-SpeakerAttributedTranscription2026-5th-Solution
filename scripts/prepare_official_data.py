#!/usr/bin/env python3
"""Validate the official data layout and materialize the released CV splits."""

from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path


def wav_format(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as audio:
        return audio.getframerate(), audio.getsampwidth(), audio.getnchannels()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.project_root.resolve()

    dev_wavs = sorted((root / "data/dev/wav").glob("*.wav"))
    test_wavs = sorted((root / "data/test/wav").glob("*.wav"))
    reference_path = root / "data/dev/ref.seglst.json"
    sample_path = root / "data/sample_submission/submit_sample.json"
    if not dev_wavs or not test_wavs or not reference_path.is_file() or not sample_path.is_file():
        raise FileNotFoundError("official data layout is incomplete; see README.md")
    invalid = [str(path) for path in [*dev_wavs, *test_wavs] if wav_format(path) != (16000, 2, 1)]
    if invalid:
        raise RuntimeError(f"expected 16 kHz, 16-bit, mono WAV files: {invalid[:5]}")

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_ids = {str(row["session_id"]) for row in reference}
    dev_ids = {path.stem for path in dev_wavs}
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    sample_ids = {str(row["session_id"]) for row in sample}
    test_ids = {path.stem for path in test_wavs}
    if reference_ids != dev_ids:
        raise RuntimeError("development WAV/reference session mismatch")
    if sample_ids != test_ids:
        raise RuntimeError("test WAV/sample-submission session mismatch")

    with (root / "configs/cv/folds_v1.csv").open(encoding="utf-8", newline="") as handle:
        fold_rows = list(csv.DictReader(handle))
    manifest_ids = {row["session_id"] for row in fold_rows}
    if manifest_ids != dev_ids:
        raise RuntimeError("released fold manifest does not match the official development set")
    split_root = root / "data/splits"
    for fold in range(5):
        val = sorted(row["session_id"] for row in fold_rows if int(row["fold"]) == fold)
        train = sorted(dev_ids - set(val))
        output = split_root / f"fold_{fold}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "train_sessions.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
        (output / "val_sessions.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "development_sessions": len(dev_wavs),
                "test_sessions": len(test_wavs),
                "folds": 5,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
