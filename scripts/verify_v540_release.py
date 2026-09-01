#!/usr/bin/env python3
"""Verify the immutable V540 release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[tuple[str, str]]:
    entries = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise RuntimeError(f"Malformed manifest line {line_number}: {line!r}")
        entries.append((fields[0], fields[1].strip().lstrip("*")))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Report missing ignored artifacts without failing.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = root / "release/v540/MANIFEST.sha256"
    failures = []
    verified = []
    missing = []
    for expected, relative in load_manifest(manifest):
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
        else:
            verified.append(relative)

    result = {
        "manifest": str(manifest),
        "verified": len(verified),
        "missing": missing,
        "mismatches": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures or (missing and not args.allow_missing):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
