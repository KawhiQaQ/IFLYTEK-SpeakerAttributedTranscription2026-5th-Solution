#!/usr/bin/env python3
"""Measure role-partition complementarity on an immutable transcript segmentation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_CANDIDATES = {
    "v173": "v173_wespeaker_novel_existence_energy",
    "v82": "v82_moss_native_substitution_consensus",
    "v84": "v84_moss_consensus_v20_tracks",
    "v74": "v74_moss_transcribe_diarize",
    "v20": "v20_diarization_quality_router",
    "v23": "v23_qwen3_diarizen_fold_adapted",
    "v16": "v16_qwen3_sortformer_offline_public_capacity_route",
    "v12": "v12_qwen3_sortformer_v2_1_universal",
    "v8": "v8_qwen3_sortformer_6spk",
}


def rows_from_payload(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
        return payload["segments"]
    raise TypeError(f"Unsupported SegLST payload: {type(payload)}")


def ordered(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["start_time"]),
            float(row["end_time"]),
            str(row["speaker"]),
        ),
    )


def map_partition(base_rows: list[dict], candidate_rows: list[dict]) -> list[str]:
    """Map a candidate partition to base segments using label-free temporal overlap."""
    candidate_rows = ordered(candidate_rows)
    mapped = []
    for base in ordered(base_rows):
        start = float(base["start_time"])
        end = float(base["end_time"])
        midpoint = (start + end) / 2
        scores: dict[str, float] = defaultdict(float)
        nearest: tuple[float, str] | None = None
        for row in candidate_rows:
            other_start = float(row["start_time"])
            other_end = float(row["end_time"])
            speaker = str(row["speaker"])
            scores[speaker] += max(0.0, min(end, other_end) - max(start, other_start))
            distance = abs(midpoint - (other_start + other_end) / 2)
            item = (distance, speaker)
            if nearest is None or item < nearest:
                nearest = item
        positive = [(score, speaker) for speaker, score in scores.items() if score > 0]
        if positive:
            mapped.append(max(positive)[1])
        elif nearest is not None:
            mapped.append(nearest[1])
        else:
            raise RuntimeError("Candidate partition has no segments")
    return mapped


def pair_relations(labels: list[str]) -> list[bool]:
    return [
        labels[left] == labels[right]
        for left in range(len(labels))
        for right in range(left + 1, len(labels))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--source", default=DEFAULT_CANDIDATES["v173"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    work_root = output.parent / f".{output.stem}_artifacts"
    work_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "source_experiment": args.source,
        "text_and_timing_immutable": True,
        "uses_test_data": False,
        "folds": {},
    }
    aggregate: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    oracle_errors = oracle_length = 0
    for fold in args.folds:
        ids = (root / f"data/splits/fold_{fold}/val_sessions.txt").read_text().split()
        role_rows: dict[str, list[dict]] = defaultdict(list)
        pair_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for session_id in ids:
            source_path = root / f"outputs/{args.source}/fold_{fold}/sessions/{session_id}.json"
            base = ordered(rows_from_payload(json.loads(source_path.read_text())))
            partitions: dict[str, list[str]] = {}
            for name, experiment in DEFAULT_CANDIDATES.items():
                path = root / f"outputs/{experiment}/fold_{fold}/sessions/{session_id}.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                candidate = rows_from_payload(json.loads(path.read_text()))
                labels = (
                    [str(row["speaker"]) for row in base]
                    if name == "v173"
                    else map_partition(base, candidate)
                )
                partitions[name] = labels
                role_rows[name].extend(
                    [{**row, "speaker": label} for row, label in zip(base, labels)]
                )
            base_relation = pair_relations(partitions["v173"])
            for name, labels in partitions.items():
                relation = pair_relations(labels)
                pair_stats[name]["pairs"] += len(relation)
                pair_stats[name]["agrees_with_v173"] += sum(
                    left == right for left, right in zip(relation, base_relation)
                )

        fold_scores = {}
        per_session_scores: dict[str, dict[str, dict]] = defaultdict(dict)
        reference = root / f"data/splits/fold_{fold}/val_ref.seglst.json"
        for name, rows in role_rows.items():
            hypothesis = work_root / f"fold_{fold}_{name}.seglst.json"
            average = work_root / f"fold_{fold}_{name}_average.json"
            per_reco = work_root / f"fold_{fold}_{name}_per_reco.json"
            hypothesis.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
            subprocess.run(
                [
                    str(Path(sys.executable).with_name("meeteval-wer")),
                    "tcpwer", "-r", str(reference), "-h", str(hypothesis),
                    "--collar", "5", "--average-out", str(average), "--per-reco-out", str(per_reco),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            score = json.loads(average.read_text())
            fold_scores[name] = score
            aggregate[name]["errors"] += int(score["errors"])
            aggregate[name]["length"] += int(score["length"])
            for session_id, value in json.loads(per_reco.read_text()).items():
                per_session_scores[session_id][name] = value
        session_oracle = []
        for session_id, values in sorted(per_session_scores.items()):
            best_name, best = min(values.items(), key=lambda item: (int(item[1]["errors"]), item[0]))
            oracle_errors += int(best["errors"])
            oracle_length += int(best["length"])
            session_oracle.append(
                {
                    "session_id": session_id,
                    "best": best_name,
                    "errors": int(best["errors"]),
                    "length": int(best["length"]),
                    "v173_errors": int(values["v173"]["errors"]),
                    "gain_over_v173": int(values["v173"]["errors"]) - int(best["errors"]),
                }
            )
        report["folds"][str(fold)] = {
            "systems": fold_scores,
            "pair_agreement_with_v173": {
                name: {
                    **stats,
                    "rate": stats["agrees_with_v173"] / max(stats["pairs"], 1),
                }
                for name, stats in pair_stats.items()
            },
            "session_oracle": session_oracle,
        }
    report["aggregate"] = {
        name: {
            **stats,
            "tcpwer": stats["errors"] / max(stats["length"], 1),
        }
        for name, stats in aggregate.items()
    }
    report["session_oracle"] = {
        "errors": oracle_errors,
        "length": oracle_length,
        "tcpwer": oracle_errors / max(oracle_length, 1),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "aggregate": report["aggregate"], "session_oracle": report["session_oracle"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
