#!/usr/bin/env python3
"""Materialize the fixed public VoxConverse training subset from its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
import wave
from pathlib import Path


class Redirect308Handler(urllib.request.HTTPRedirectHandler):
    def http_error_308(self, req, fp, code, msg, headers):
        return self.http_error_302(req, fp, 302, msg, headers)


OPENER = urllib.request.build_opener(Redirect308Handler())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_range(url: str, start: int, end: int, timeout: int) -> bytes:
    expected = end - start + 1
    last_error: Exception | None = None
    for attempt in range(5):
        request = urllib.request.Request(
            url,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "cast-reproducibility-package/1.0",
            },
        )
        try:
            with OPENER.open(request, timeout=timeout) as response:
                payload = response.read()
                status = getattr(response, "status", None)
            if status == 206 and len(payload) == expected:
                return payload
            last_error = RuntimeError(
                f"range response mismatch: status={status}, "
                f"expected={expected}, received={len(payload)}"
            )
        except Exception as error:  # network errors are retried below
            last_error = error
        if attempt < 4:
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"failed to download byte range from {url}: {last_error}")


def write_rttm(path: Path, session_id: str, segments: list[dict]) -> None:
    lines = []
    for row in segments:
        start = float(row["start"])
        duration = float(row["end"]) - start
        lines.append(
            f"SPEAKER {session_id} 1 {start:.4f} {duration:.4f} "
            f"<NA> <NA> {row['speaker']} <NA> <NA>"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", nargs="?", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/voxconverse_fixed_48.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/external/voxconverse_fixed_48"),
    )
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    manifest_path = (root / args.manifest).resolve()
    output_root = (root / args.output_root).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("subset_id") != "voxconverse_fixed_48":
        raise RuntimeError("unexpected external-data manifest")

    wav_root = output_root / "wav"
    rttm_root = output_root / "rttm"
    wav_root.mkdir(parents=True, exist_ok=True)
    rttm_root.mkdir(parents=True, exist_ok=True)
    endpoint = args.endpoint.rstrip("/")
    repository = manifest["mirror_repository"]
    completed = []

    for index, row in enumerate(manifest["sessions"], start=1):
        session_id = str(row["session_id"])
        wav_path = wav_root / f"{session_id}.wav"
        rttm_path = rttm_root / f"{session_id}.rttm"
        expected_wav = str(row["wav_sha256"])
        expected_rttm = str(row["rttm_sha256"])
        valid_existing = (
            wav_path.is_file()
            and rttm_path.is_file()
            and sha256_file(wav_path) == expected_wav
            and sha256_file(rttm_path) == expected_rttm
        )
        if valid_existing and not args.overwrite:
            completed.append(
                {
                    "session_id": session_id,
                    "wav_sha256": expected_wav,
                    "rttm_sha256": expected_rttm,
                }
            )
            print(json.dumps({"session": session_id, "status": "verified"}))
            continue

        quoted = urllib.parse.quote(str(row["source_path"]), safe="/")
        url = f"{endpoint}/datasets/{repository}/resolve/main/{quoted}"
        start, end = map(int, row["source_byte_range"])
        frames = fetch_range(url, start, end, args.timeout)
        with wave.open(str(wav_path), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(16000)
            target.writeframes(frames)
        write_rttm(rttm_path, session_id, row["segments"])
        actual_wav = sha256_file(wav_path)
        actual_rttm = sha256_file(rttm_path)
        if actual_wav != expected_wav or actual_rttm != expected_rttm:
            raise RuntimeError(
                f"hash mismatch for {session_id}: "
                f"wav={actual_wav}, rttm={actual_rttm}"
            )
        completed.append(
            {
                "session_id": session_id,
                "wav_sha256": expected_wav,
                "rttm_sha256": expected_rttm,
            }
        )
        print(
            json.dumps(
                {
                    "session": session_id,
                    "position": index,
                    "total": len(manifest["sessions"]),
                    "status": "downloaded",
                }
            ),
            flush=True,
        )

    audit = {
        "dataset": manifest["dataset"],
        "version": manifest["version"],
        "subset_id": manifest["subset_id"],
        "sessions": completed,
        "manifest_sha256": sha256_file(manifest_path),
    }
    (output_root / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_root": str(output_root), "sessions": len(completed)}))


if __name__ == "__main__":
    main()
