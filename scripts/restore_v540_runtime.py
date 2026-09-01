#!/usr/bin/env python3
"""Safely restore the verified V540 runtime bundle into a clean checkout."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
from pathlib import Path, PurePosixPath


EXPECTED_ARCHIVE_SHA256 = (
    "8b91ede519af571af9ff7e982d5d63def7b83852427dfcdc2da490843a03c370"
)
RUNTIME_PREFIXES = {"data", "models", "outputs", "reports", "submissions"}
SOURCE_PREFIXES = {"configs", "docs", "scripts", "third_party"}
FINAL_SUBMISSION_FILES = {
    "v95_moss_acoustic_graph_fusion.seglst.json",
    "v174_wespeaker_novel_existence_energy.seglst.json",
    "v174_wespeaker_novel_existence_energy.audit.json",
    "v538_voxconverse_context_v174.seglst.json",
    "v538_voxconverse_context_v174.audit.json",
    "v540_v538_self_overlap_restore_v2.seglst.json",
    "v540_v538_self_overlap_restore_v2.audit.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def safe_relative(name: str) -> PurePosixPath:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe archive member: {name}")
    return relative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--include-source", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    archive = args.archive.resolve()
    root = args.root.resolve()
    actual = sha256(archive)
    if actual != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(
            f"Archive SHA-256 mismatch: expected {EXPECTED_ARCHIVE_SHA256}, got {actual}"
        )

    allowed = set(RUNTIME_PREFIXES)
    if args.include_source:
        allowed.update(SOURCE_PREFIXES)

    extracted = 0
    skipped = 0
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            relative = safe_relative(member.name)
            if not relative.parts or relative.parts[0] not in allowed:
                skipped += 1
                continue
            if (
                relative.parts[0] == "submissions"
                and len(relative.parts) == 2
                and relative.parts[1] not in FINAL_SUBMISSION_FILES
            ):
                skipped += 1
                continue
            if member.issym() or member.islnk():
                raise RuntimeError(f"Links are not allowed in the runtime archive: {member.name}")
            destination = root.joinpath(*relative.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                skipped += 1
                continue
            if destination.exists() and not args.overwrite:
                source = bundle.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Cannot read archive member: {member.name}")
                incoming_hash = sha256_stream(source)
                if destination.is_file() and sha256(destination) == incoming_hash:
                    skipped += 1
                    continue
                raise FileExistsError(
                    f"Refusing to replace {destination}; pass --overwrite after reviewing it"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot read archive member: {member.name}")
            with destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted += 1

    print(
        f"restored archive={archive} root={root} extracted_files={extracted} "
        f"skipped_members={skipped}"
    )


if __name__ == "__main__":
    main()
