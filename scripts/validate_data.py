#!/usr/bin/env python3
"""Validate the official Xunfei speaker-attributed transcription data."""

from __future__ import annotations

import argparse
import json
import wave
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_KEYS = {"session_id", "speaker", "start_time", "end_time", "words"}


def wav_info(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as wav:
        return wav.getframerate(), wav.getsampwidth(), wav.getnchannels()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?", type=Path, default=Path("data"))
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    dev_wavs = sorted((data_dir / "dev" / "wav").glob("*.wav"))
    test_wavs = sorted((data_dir / "test" / "wav").glob("*.wav"))
    ref_path = data_dir / "dev" / "ref.seglst.json"
    sample_path = data_dir / "sample_submission" / "submit_sample.json"

    assert dev_wavs, "No development WAV files found"
    assert test_wavs, "No test WAV files found"

    bad_wavs = {
        str(path): info
        for path in dev_wavs + test_wavs
        if (info := wav_info(path)) != (16000, 2, 1)
    }
    assert not bad_wavs, f"Unexpected WAV formats: {bad_wavs}"

    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    assert isinstance(ref, list) and ref, "Reference must be a non-empty JSON list"

    by_session: dict[str, list[dict]] = defaultdict(list)
    for index, segment in enumerate(ref):
        missing = REQUIRED_KEYS - segment.keys()
        assert not missing, f"Reference segment {index} misses {sorted(missing)}"
        assert isinstance(segment["session_id"], str) and segment["session_id"]
        assert isinstance(segment["speaker"], str) and segment["speaker"]
        assert isinstance(segment["words"], str) and segment["words"].strip()
        assert segment["words"] == segment["words"].strip()
        assert "  " not in segment["words"], f"Double space in segment {index}"
        assert 0 <= segment["start_time"] < segment["end_time"]
        by_session[segment["session_id"]].append(segment)

    dev_ids = {path.stem for path in dev_wavs}
    assert set(by_session) == dev_ids, "Development WAV/reference session mismatch"

    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    assert isinstance(sample, list) and sample, "Submission sample must be a JSON list"
    test_ids = {path.stem for path in test_wavs}
    sample_ids = {row["session_id"] for row in sample}
    assert sample_ids == test_ids, "Test WAV/submission sample session mismatch"

    speaker_distribution = Counter(
        len({segment["speaker"] for segment in segments})
        for segments in by_session.values()
    )
    summary = {
        "data_dir": str(data_dir),
        "wav_format": {"sample_rate": 16000, "sample_width_bytes": 2, "channels": 1},
        "dev_wavs": len(dev_wavs),
        "test_wavs": len(test_wavs),
        "reference_segments": len(ref),
        "speaker_count_distribution": dict(sorted(speaker_distribution.items())),
        "sample_submission_rows": len(sample),
        "status": "ok",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
