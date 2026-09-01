#!/usr/bin/env python3
"""Evaluate one frozen CV fold and generate compact failure-mode slices."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path


def weighted_summary(rows: list[dict]) -> dict:
    errors = sum(int(row["errors"]) for row in rows)
    length = sum(int(row["length"]) for row in rows)
    return {
        "sessions": len(rows),
        "errors": errors,
        "length": length,
        "tcpwer": errors / max(length, 1),
    }


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Memory-efficient Levenshtein distance for a speaker-agnostic WER proxy."""
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1]


def concatenate_tokens(rows: list[dict]) -> list[str]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["start_time"]),
            float(row["end_time"]),
            str(row["speaker"]),
        ),
    )
    return [token for row in ordered for token in str(row["words"]).split()]


def union_duration(rows: list[dict]) -> float:
    intervals = sorted(
        (float(row["start_time"]), float(row["end_time"]))
        for row in rows
        if float(row["end_time"]) > float(row["start_time"])
    )
    if not intervals:
        return 0.0
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def pearson(rows: list[dict], left: str, right: str) -> float | None:
    if len(rows) < 2:
        return None
    xs = [float(row[left]) for row in rows]
    ys = [float(row[right]) for row in rows]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    root = args.project_root.resolve()
    output_dir = root / "outputs" / args.experiment / f"fold_{args.fold}{args.output_suffix}"
    hypothesis = output_dir / "hyp.seglst.json"
    reference = root / "data" / "splits" / f"fold_{args.fold}" / "val_ref.seglst.json"
    if not hypothesis.is_file() or not reference.is_file():
        raise FileNotFoundError(f"Missing hypothesis or reference: {hypothesis}, {reference}")

    reference_rows = json.loads(reference.read_text(encoding="utf-8"))
    hypothesis_rows = json.loads(hypothesis.read_text(encoding="utf-8"))
    ref_sessions = {row["session_id"] for row in reference_rows}
    hyp_sessions = {row["session_id"] for row in hypothesis_rows}
    if ref_sessions != hyp_sessions:
        raise RuntimeError(
            f"Validation coverage mismatch: missing={sorted(ref_sessions - hyp_sessions)}, "
            f"extra={sorted(hyp_sessions - ref_sessions)}"
        )

    average_path = output_dir / "tcpwer.json"
    per_reco_path = output_dir / "tcpwer_per_reco.json"
    command = [
        "meeteval-wer",
        "tcpwer",
        "-r",
        str(reference),
        "-h",
        str(hypothesis),
        "--collar",
        "5",
        "--average-out",
        str(average_path),
        "--per-reco-out",
        str(per_reco_path),
    ]
    subprocess.run(command, check=True)

    average = json.loads(average_path.read_text(encoding="utf-8"))
    per_reco_raw = json.loads(per_reco_path.read_text(encoding="utf-8"))
    if not isinstance(per_reco_raw, dict):
        raise RuntimeError(f"Unexpected per-reco output: {type(per_reco_raw)}")

    manifest = {}
    with (root / "configs" / "cv" / "folds_v1.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            manifest[row["session_id"]] = row

    ref_by_session: dict[str, list[dict]] = defaultdict(list)
    hyp_by_session: dict[str, list[dict]] = defaultdict(list)
    for row in reference_rows:
        ref_by_session[row["session_id"]].append(row)
    for row in hypothesis_rows:
        hyp_by_session[row["session_id"]].append(row)

    session_rows = []
    for session_id, score in per_reco_raw.items():
        ref_rows = ref_by_session[session_id]
        hyp_rows = hyp_by_session[session_id]
        ref_tokens = concatenate_tokens(ref_rows)
        hyp_tokens = concatenate_tokens(hyp_rows)
        text_errors = edit_distance(ref_tokens, hyp_tokens)
        ref_speakers = len({row["speaker"] for row in ref_rows})
        hyp_speakers = len({row["speaker"] for row in hyp_rows})
        ref_speech_duration = union_duration(ref_rows)
        hyp_speech_duration = union_duration(hyp_rows)
        row = {
            "session_id": session_id,
            "errors": int(score["errors"]),
            "length": int(score["length"]),
            "tcpwer": float(score["error_rate"]),
            "speakers": int(manifest[session_id]["speakers"]),
            "overlap_ratio": float(manifest[session_id]["overlap_ratio"]),
            "duration": float(manifest[session_id]["duration"]),
            "reference_speakers": ref_speakers,
            "hypothesis_speakers": hyp_speakers,
            "speaker_count_delta": hyp_speakers - ref_speakers,
            "absolute_speaker_count_error": abs(hyp_speakers - ref_speakers),
            "speaker_agnostic_text_errors": text_errors,
            "speaker_agnostic_text_length": len(ref_tokens),
            "speaker_agnostic_text_wer": text_errors / max(len(ref_tokens), 1),
            "reference_speech_duration": ref_speech_duration,
            "hypothesis_speech_duration": hyp_speech_duration,
            "speech_coverage_ratio": hyp_speech_duration / max(ref_speech_duration, 1e-12),
        }
        session_rows.append(row)

    overlap_values = sorted(row["overlap_ratio"] for row in session_rows)
    lower = overlap_values[len(overlap_values) // 3]
    upper = overlap_values[(2 * len(overlap_values)) // 3]
    slices: dict[str, list[dict]] = defaultdict(list)
    for row in session_rows:
        slices[f"speakers={row['speakers']}"] .append(row)
        if row["overlap_ratio"] <= lower:
            overlap_bin = "low"
        elif row["overlap_ratio"] <= upper:
            overlap_bin = "mid"
        else:
            overlap_bin = "high"
        slices[f"overlap={overlap_bin}"].append(row)

    text_errors = sum(row["speaker_agnostic_text_errors"] for row in session_rows)
    text_length = sum(row["speaker_agnostic_text_length"] for row in session_rows)
    speaker_deltas = [row["speaker_count_delta"] for row in session_rows]

    report = {
        "primary": average,
        "coverage": {"reference_sessions": len(ref_sessions), "hypothesis_sessions": len(hyp_sessions)},
        "slices": {key: weighted_summary(rows) for key, rows in sorted(slices.items())},
        "diagnostics": {
            "speaker_agnostic_chronological_wer_proxy": text_errors / max(text_length, 1),
            "speaker_agnostic_text_errors": text_errors,
            "speaker_agnostic_text_length": text_length,
            "speaker_count_exact_sessions": sum(delta == 0 for delta in speaker_deltas),
            "speaker_count_over_sessions": sum(delta > 0 for delta in speaker_deltas),
            "speaker_count_under_sessions": sum(delta < 0 for delta in speaker_deltas),
            "speaker_count_mae": sum(abs(delta) for delta in speaker_deltas) / len(speaker_deltas),
            "mean_speech_coverage_ratio": sum(
                row["speech_coverage_ratio"] for row in session_rows
            )
            / len(session_rows),
            "correlations_with_tcpwer": {
                "overlap_ratio": pearson(session_rows, "tcpwer", "overlap_ratio"),
                "absolute_speaker_count_error": pearson(
                    session_rows, "tcpwer", "absolute_speaker_count_error"
                ),
                "speaker_agnostic_text_wer": pearson(
                    session_rows, "tcpwer", "speaker_agnostic_text_wer"
                ),
                "speech_coverage_ratio": pearson(
                    session_rows, "tcpwer", "speech_coverage_ratio"
                ),
            },
        },
        "sessions": sorted(session_rows, key=lambda row: row["session_id"]),
        "worst_sessions": sorted(session_rows, key=lambda row: row["tcpwer"], reverse=True)[:8],
        "command": command,
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
