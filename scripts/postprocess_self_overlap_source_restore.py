#!/usr/bin/env python3
"""Repair physically impossible same-speaker overlaps from a frozen source partition.

The rule is label-free at inference time.  It only edits a refined segment when:

1. two refined segments assigned to the same speaker overlap materially; and
2. the frozen native source assigns the two word/time regions to different roles.

The replacement role is recovered from unchanged source/refined correspondences
within the same session.  Words and timestamps are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path


def words(row: dict) -> tuple[str, ...]:
    return tuple(str(row["words"]).split())


def overlap(a: dict, b: dict) -> float:
    return max(0.0, min(float(a["end_time"]), float(b["end_time"])) - max(float(a["start_time"]), float(b["start_time"])))


def canonical_speaker(value: str) -> str:
    match = re.fullmatch(r"(?:s|spk)0*(\d+)", str(value))
    return f"spk{int(match.group(1))}" if match else str(value)


def source_match_score(refined: dict, source: dict) -> tuple[float, ...]:
    if refined["session_id"] != source["session_id"]:
        return (-1.0,)
    rw, sw = words(refined), words(source)
    exact_words = float(rw == sw)
    contained = float(bool(rw) and len(rw) <= len(sw) and any(sw[i : i + len(rw)] == rw for i in range(len(sw) - len(rw) + 1)))
    shared = len(Counter(rw) & Counter(sw)) / max(1, len(rw))
    ov = overlap(refined, source)
    duration = max(1e-6, float(refined["end_time"]) - float(refined["start_time"]))
    exact_time = float(abs(float(refined["start_time"]) - float(source["start_time"])) < 0.011 and abs(float(refined["end_time"]) - float(source["end_time"])) < 0.011)
    return (exact_words, contained, shared, ov / duration, exact_time, ov)


def best_source(refined: dict, source_rows: list[dict]) -> dict | None:
    candidates = [(source_match_score(refined, row), row) for row in source_rows]
    candidates = [(score, row) for score, row in candidates if score[-1] > 0.0]
    if not candidates:
        return None
    score, row = max(candidates, key=lambda item: item[0])
    if score[0] == 0.0 and score[1] == 0.0 and score[2] < 0.5:
        return None
    return row


def infer_source_role_map(refined_rows: list[dict], source_rows: list[dict], conflict_indices: set[int]) -> dict[str, str]:
    votes: dict[str, Counter] = defaultdict(Counter)
    for index, row in enumerate(refined_rows):
        if index in conflict_indices:
            continue
        src = best_source(row, source_rows)
        if src is None:
            continue
        score = source_match_score(row, src)
        if score[0] or score[4]:
            votes[str(src["speaker"])][str(row["speaker"])] += max(1, len(words(row)))
    mapping = {}
    for source_speaker, counts in votes.items():
        mapping[source_speaker] = counts.most_common(1)[0][0]
    return mapping


def process_session(refined_rows: list[dict], source_rows: list[dict], min_overlap: float) -> tuple[list[dict], list[dict]]:
    rows = deepcopy(refined_rows)
    conflicts = []
    conflict_indices: set[int] = set()
    for i, a in enumerate(rows):
        for j in range(i + 1, len(rows)):
            b = rows[j]
            ov = overlap(a, b)
            if a["speaker"] == b["speaker"] and ov >= min_overlap:
                conflicts.append((i, j, ov))
                conflict_indices.update((i, j))

    role_map = infer_source_role_map(rows, source_rows, conflict_indices)
    decisions = []
    edited_indices: set[int] = set()
    for i, j, ov in conflicts:
        if i in edited_indices or j in edited_indices:
            continue
        a, b = rows[i], rows[j]
        sa, sb = best_source(a, source_rows), best_source(b, source_rows)
        if sa is None or sb is None or sa["speaker"] == sb["speaker"]:
            continue

        proposals = {}
        for index, src in ((i, sa), (j, sb)):
            current = str(rows[index]["speaker"])
            canonical_source = canonical_speaker(str(src["speaker"]))
            if canonical_source == current:
                proposals[index] = current
                continue
            # If this source segment was split by the refiner, prefer the
            # dominant non-conflicting refined role inside that exact source span.
            local = Counter()
            for k, candidate in enumerate(rows):
                if k in conflict_indices or candidate["speaker"] == current:
                    continue
                if overlap(candidate, src) > 0 and source_match_score(candidate, src)[2] >= 0.5:
                    local[str(candidate["speaker"])] += max(1, len(words(candidate)))
            proposals[index] = (
                local.most_common(1)[0][0]
                if local
                else role_map.get(str(src["speaker"]), canonical_source)
            )

        # The edit must actually resolve the physical contradiction.  Ambiguous
        # source mappings that collapse both regions to the same role are left
        # untouched.
        proposed_i = proposals.get(i) or str(rows[i]["speaker"])
        proposed_j = proposals.get(j) or str(rows[j]["speaker"])
        if proposed_i == proposed_j:
            continue

        safe = True
        for index, proposed in ((i, proposed_i), (j, proposed_j)):
            if proposed == rows[index]["speaker"]:
                continue
            for k, other in enumerate(rows):
                if k in (i, j):
                    continue
                if other["speaker"] == proposed and overlap(rows[index], other) >= min_overlap:
                    safe = False
                    break
            if not safe:
                break
        if not safe:
            continue

        for index, src, proposed in ((i, sa, proposed_i), (j, sb, proposed_j)):
            current = str(rows[index]["speaker"])
            if proposed == current:
                continue
            before = rows[index].copy()
            rows[index]["speaker"] = proposed
            edited_indices.add(index)
            decisions.append({
                "session_id": rows[index]["session_id"],
                "start_time": rows[index]["start_time"],
                "end_time": rows[index]["end_time"],
                "words": rows[index]["words"],
                "from": current,
                "to": proposed,
                "source_speaker": src["speaker"],
                "trigger_overlap_seconds": round(ov, 4),
                "before": before,
            })
    return rows, decisions


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--min-overlap", type=float, default=0.20)
    args = parser.parse_args()

    refined = json.loads(args.refined.read_text(encoding="utf-8"))
    source = json.loads(args.source.read_text(encoding="utf-8"))
    refined_by_session: dict[str, list[dict]] = defaultdict(list)
    source_by_session: dict[str, list[dict]] = defaultdict(list)
    for row in refined:
        refined_by_session[str(row["session_id"])].append(row)
    for row in source:
        source_by_session[str(row["session_id"])].append(row)

    output = []
    decisions = []
    for session_id in sorted(refined_by_session):
        rows, session_decisions = process_session(
            refined_by_session[session_id], source_by_session.get(session_id, []), args.min_overlap
        )
        output.extend(rows)
        decisions.extend(session_decisions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_path = args.audit or args.output.with_suffix(".audit.json")
    audit = {
        "rule": "material_self_overlap_source_restore",
        "min_overlap_seconds": args.min_overlap,
        "words_changed": False,
        "timestamps_changed": False,
        "uses_reference_labels": False,
        "uses_test_for_selection": False,
        "refined_sha256": sha256(args.refined),
        "source_sha256": sha256(args.source),
        "output_sha256": sha256(args.output),
        "sessions": len(refined_by_session),
        "segments": len(output),
        "decisions": decisions,
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "audit": str(audit_path), "decisions": len(decisions), "sha256": audit["output_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
