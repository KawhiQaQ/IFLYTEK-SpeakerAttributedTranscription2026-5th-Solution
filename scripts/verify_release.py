#!/usr/bin/env python3
"""Verify the self-contained release, custom checkpoints, and optional inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FINAL_SHA256 = "033b699bfcfbcb9da3a6ae9aa0188d3d31b2de1bc8e82fca8a422d20415d6630"

# Accepted only for the optional Baidu backup published before release-layout
# metadata was made portable. Tensor values are identical to the packaged
# checkpoints; the differing bytes are absolute-path/config metadata.
LEGACY_CHECKPOINT_SHA256 = {
    "models/data_experts/contextual_speaker_transformer/fold_0/contextual_metric.pt": {
        "4edb68802191085176e4b03b2866fe2e38a3ee9187ad2a90bae0875b76f095e1"
    },
    "models/data_experts/contextual_speaker_transformer/full/contextual_metric.pt": {
        "5bcc4685f3532983137c120325c1893b55c541f1f0d1ce50625d4de60a1eac79"
    },
    "models/sortformer_finetune/v14_v2_1_synthetic/fold_full/sortformer_fold_full.nemo": {
        "3579848b4fa01a7e36505877d80d413e7b938f422101b00769fc144e17ef6177"
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_model_path(root: Path, relative: str | Path) -> Path:
    """Resolve models from either the runtime link or the audit-package layout."""
    relative = Path(relative)
    runtime_path = root / relative
    if runtime_path.exists() or relative.parts[:1] != ("models",):
        return runtime_path
    return root.parent / "user_data" / "model_data" / Path(*relative.parts[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", nargs="?", type=Path, default=Path("."))
    parser.add_argument("--require-official-data", action="store_true")
    parser.add_argument("--require-public-models", action="store_true")
    parser.add_argument(
        "--allow-retrained-checkpoints",
        action="store_true",
        help="Require every custom checkpoint but do not enforce released hashes.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    problems: list[str] = []

    checksum_path = root / "checkpoints.sha256"
    checked = 0
    legacy_metadata = []
    if not checksum_path.is_file():
        problems.append("missing checkpoints.sha256")
    else:
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split(maxsplit=1)
            path = resolve_model_path(root, relative.strip())
            if not path.is_file():
                problems.append(f"missing custom checkpoint: {relative}")
            elif not args.allow_retrained_checkpoints:
                actual = sha256_file(path)
                if actual != expected:
                    if actual in LEGACY_CHECKPOINT_SHA256.get(relative.strip(), set()):
                        legacy_metadata.append(relative.strip())
                    else:
                        problems.append(f"custom checkpoint hash mismatch: {relative}")
            checked += 1

    reference = root / "reference/final_submission.seglst.json"
    if not reference.is_file() or sha256_file(reference) != FINAL_SHA256:
        problems.append("reference final submission is missing or corrupted")

    required_sources = [
        "third_party/3D-Speaker/speakerlab",
        "third_party/FireRedASR2S/fireredasr2s",
        "third_party/MOSS-Transcribe-Diarize/moss_transcribe_diarize",
        "third_party/DiariZen/pyannote-audio/pyannote/audio",
    ]
    for relative in required_sources:
        if not (root / relative).is_dir():
            problems.append(f"missing vendored source: {relative}")

    if args.require_official_data:
        for relative in [
            "data/dev/ref.seglst.json",
            "data/dev/wav",
            "data/test/wav",
            "data/sample_submission/submit_sample.json",
        ]:
            if not (root / relative).exists():
                problems.append(f"missing official-data input: {relative}")

    if args.require_public_models:
        import yaml

        manifest = yaml.safe_load((root / "configs/public_models.yaml").read_text())
        for model in manifest["models"]:
            model_relative = Path(model["local_dir"])
            directory = resolve_model_path(root, model_relative)
            if not directory.is_dir():
                problems.append(f"missing public model directory: {model_relative}")
            for filename, expected in model.get("files", {}).items():
                path = directory / filename
                if not path.is_file():
                    problems.append(f"missing public model file: {model_relative / filename}")
                elif sha256_file(path) != expected:
                    problems.append(f"public model hash mismatch: {model_relative / filename}")
            if model["name"] == "MOSS-Transcribe-Diarize" and directory.is_dir():
                tokenizer_config = directory / "tokenizer_config.json"
                if not tokenizer_config.is_file():
                    problems.append("missing MOSS tokenizer_config.json")
                else:
                    tokenizer_payload = json.loads(
                        tokenizer_config.read_text(encoding="utf-8")
                    )
                    if isinstance(tokenizer_payload.get("extra_special_tokens"), list):
                        problems.append(
                            "MOSS tokenizer compatibility patch is not installed; "
                            "rerun scripts/download_public_models.py --skip-existing"
                        )

    result = {
        "status": "ok" if not problems else "failed",
        "custom_checkpoints_checked": checked,
        "custom_checkpoint_hashes_enforced": not args.allow_retrained_checkpoints,
        "legacy_metadata_checkpoints": legacy_metadata,
        "reference_submission_sha256": FINAL_SHA256,
        "problems": problems,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
