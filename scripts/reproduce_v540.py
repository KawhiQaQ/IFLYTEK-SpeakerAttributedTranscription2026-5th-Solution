#!/usr/bin/env python3
"""Reproduce the byte-identical V540 submission from frozen V538 and V95."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


EXPECTED = {
    "refined": "ac25cc0bccc28912a078dde026bb788ade59c9d729114b0422b32492c0a594be",
    "source": "f050dee0c73105320c86b17e0a5822db87d31b0f65b73417a602b3f4888f002f",
    "output": "033b699bfcfbcb9da3a6ae9aa0188d3d31b2de1bc8e82fca8a422d20415d6630",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {label}: {path}. Restore artifacts/releases/v540_runtime.tar.gz first."
        )
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual} ({path})"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    refined = root / "submissions/v538_voxconverse_context_v174.seglst.json"
    source = root / "submissions/v95_moss_acoustic_graph_fusion.seglst.json"
    output = (
        args.output.resolve()
        if args.output
        else root / "submissions/v540_reproduced.seglst.json"
    )
    audit = (
        args.audit.resolve()
        if args.audit
        else output.with_name(f"{output.stem}.audit.json")
    )

    require_hash(refined, EXPECTED["refined"], "V538 parent")
    require_hash(source, EXPECTED["source"], "V95 source")

    command = [
        sys.executable,
        str(root / "scripts/postprocess_self_overlap_source_restore.py"),
        "--refined",
        str(refined),
        "--source",
        str(source),
        "--output",
        str(output),
        "--audit",
        str(audit),
        "--min-overlap",
        "0.20",
    ]
    subprocess.run(command, cwd=root, check=True)
    require_hash(output, EXPECTED["output"], "V540 output")

    payload = json.loads(audit.read_text(encoding="utf-8"))
    if len(payload.get("decisions", [])) != 11:
        raise RuntimeError("Unexpected V540 decision count")
    print(
        json.dumps(
            {
                "status": "byte-identical",
                "output": str(output),
                "decisions": 11,
                "sha256": EXPECTED["output"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
