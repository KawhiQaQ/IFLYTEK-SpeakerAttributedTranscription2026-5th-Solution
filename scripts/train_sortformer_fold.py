#!/usr/bin/env python3
"""Fold-pure supervised adaptation of streaming Sortformer on official data."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import wave
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf, open_dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def prepare_manifest(
    root: Path,
    fold: int | str,
    max_speakers: int,
    output_dir: Path,
    augmentation: dict | None = None,
) -> tuple[Path, list[str], dict[str, int], dict | None]:
    if fold == "full":
        train_sessions = {
            path.stem for path in (root / "data" / "dev" / "wav").glob("*.wav")
        }
        validation_sessions: set[str] = set()
    else:
        split_dir = root / "data" / "splits" / f"fold_{fold}"
        train_sessions = set(
            (split_dir / "train_sessions.txt").read_text(encoding="utf-8").split()
        )
        validation_sessions = set(
            (split_dir / "val_sessions.txt").read_text(encoding="utf-8").split()
        )
    if train_sessions & validation_sessions:
        raise RuntimeError("Frozen CV train/validation leakage")
    references = json.loads(
        (root / "data" / "dev" / "ref.seglst.json").read_text(encoding="utf-8")
    )
    by_session: dict[str, list[dict]] = {session_id: [] for session_id in train_sessions}
    for row in references:
        if row["session_id"] in by_session:
            by_session[row["session_id"]].append(row)
    speaker_counts = {
        session_id: len({row["speaker"] for row in rows})
        for session_id, rows in by_session.items()
    }
    selected = sorted(
        session_id
        for session_id in train_sessions
        if by_session[session_id] and speaker_counts[session_id] <= max_speakers
    )
    if not selected:
        raise RuntimeError("No eligible fold-pure training sessions")

    rttm_dir = output_dir / "rttm"
    rttm_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for session_id in selected:
        wav_path = (root / "data" / "dev" / "wav" / f"{session_id}.wav").resolve()
        rttm_path = (rttm_dir / f"{session_id}.rttm").resolve()
        rttm_lines = []
        for row in sorted(
            by_session[session_id], key=lambda item: (item["start_time"], item["end_time"])
        ):
            start = float(row["start_time"])
            segment_duration = float(row["end_time"]) - start
            if segment_duration <= 0:
                continue
            rttm_lines.append(
                f"SPEAKER {session_id} 1 {start:.3f} {segment_duration:.3f} "
                f"<NA> <NA> {row['speaker']} <NA> <NA>"
            )
        rttm_path.write_text("\n".join(rttm_lines) + "\n", encoding="utf-8")
        manifest_rows.append(
            {
                "audio_filepath": str(wav_path),
                "duration": duration_seconds(wav_path),
                "offset": 0,
                "rttm_filepath": str(rttm_path),
                "uniq_id": session_id,
            }
        )
    augmentation_audit = None
    if augmentation:
        augmentation_manifest = (
            root / str(augmentation["manifest"]).format(fold=fold)
        ).resolve()
        augmentation_audit_path = (
            root / str(augmentation["audit"]).format(fold=fold)
        ).resolve()
        if not augmentation_manifest.is_file() or not augmentation_audit_path.is_file():
            raise FileNotFoundError("Missing configured Sortformer augmentation")
        augmentation_audit = json.loads(
            augmentation_audit_path.read_text(encoding="utf-8")
        )
        if augmentation_audit.get("uses_validation_labels") is not False:
            raise RuntimeError("Augmentation uses validation labels")
        if augmentation_audit.get("uses_test_data") is not False:
            raise RuntimeError("Augmentation uses test data")
        if str(augmentation_audit.get("fold")) != str(fold):
            raise RuntimeError("Augmentation fold mismatch")
        augmentation_sources = set(augmentation_audit.get("source_sessions", []))
        if not augmentation_sources <= train_sessions:
            raise RuntimeError("Augmentation contains non-training source sessions")
        if augmentation_sources & validation_sessions:
            raise RuntimeError("Augmentation leaks validation sessions")
        if augmentation_audit.get("manifest_sha256") != sha256_file(
            augmentation_manifest
        ):
            raise RuntimeError("Augmentation manifest hash mismatch")
        for line in augmentation_manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not Path(row["audio_filepath"]).is_file() or not Path(
                row["rttm_filepath"]
            ).is_file():
                raise FileNotFoundError("Incomplete augmentation artifact")
            manifest_rows.append(row)

    manifest_path = output_dir / "train_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    audit = {
        "fold": fold,
        "uses_validation_labels": False,
        "uses_test_data": False,
        "max_speakers": max_speakers,
        "selected_train_sessions": selected,
        "excluded_train_sessions_above_capacity": sorted(train_sessions - set(selected)),
        "validation_sessions": sorted(validation_sessions),
        "train_validation_intersection": [],
        "speaker_counts": {session_id: speaker_counts[session_id] for session_id in selected},
        "augmentation": augmentation_audit,
        "natural_session_count": len(selected),
        "total_training_session_count": len(manifest_rows),
        "manifest_sha256": sha256_file(manifest_path),
    }
    (output_dir / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path, selected, speaker_counts, augmentation_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", help="Fold number, or 'full' for the final model")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Minimal backward-pass smoke override. Omit for the configured full run.",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fold_value = config["fold"] if args.fold is None else args.fold
    fold: int | str = "full" if str(fold_value) == "full" else int(fold_value)
    if fold != "full":
        split_dir = (root / "data" / "splits" / f"fold_{fold}").resolve()
        if split_dir.parent.name != "splits":
            raise RuntimeError("Development-only fold guard failed")
    smoke = args.max_steps is not None
    run_dir = (
        root
        / "outputs"
        / "training"
        / config["name"]
        / f"fold_{fold}{'_smoke' if smoke else ''}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, selected_sessions, _, augmentation_audit = prepare_manifest(
        root,
        fold,
        int(config["data"]["max_speakers"]),
        run_dir,
        config.get("augmentation"),
    )

    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from nemo.collections.asr.models import SortformerEncLabelModel

    checkpoint_path = (root / config["base_checkpoint"]).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    callback = ModelCheckpoint(
        dirpath=str(run_dir / "lightning_checkpoints"),
        save_last=True,
        save_top_k=0,
    )
    trainer = Trainer(
        accelerator=config["training"]["accelerator"],
        devices=int(config["training"]["devices"]),
        precision=config["training"]["precision"],
        max_epochs=1 if smoke else int(config["training"]["max_epochs"]),
        max_steps=args.max_steps if smoke else -1,
        gradient_clip_val=float(config["training"]["gradient_clip_val"]),
        log_every_n_steps=int(config["training"]["log_every_n_steps"]),
        enable_checkpointing=True,
        callbacks=[callback],
        logger=CSVLogger(str(run_dir), name="lightning_logs"),
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )
    model = SortformerEncLabelModel.restore_from(
        str(checkpoint_path), map_location=torch.device("cpu"), trainer=trainer
    )
    if bool(config["training"]["freeze_acoustic_encoder"]):
        model.encoder.freeze()
    if bool(config["training"].get("freeze_transformer_encoder", False)):
        for parameter in model.transformer_encoder.parameters():
            parameter.requires_grad = False
        model.transformer_encoder.eval()
    with open_dict(model.cfg):
        model.cfg.optim.lr = float(config["training"]["learning_rate"])
        model.cfg.optim.sched.warmup_steps = int(config["training"]["warmup_steps"])
        model.cfg.train_ds.manifest_filepath = str(manifest_path)
        model.cfg.train_ds.session_len_sec = float(config["data"]["session_len_seconds"])
        model.cfg.train_ds.batch_size = int(config["data"]["batch_size"])
        model.cfg.train_ds.num_workers = int(config["data"]["num_workers"])
        model.cfg.train_ds.shuffle = True
        model.cfg.train_ds.pin_memory = True
    model.setup_training_data(OmegaConf.create(OmegaConf.to_container(model.cfg.train_ds)))

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(
        json.dumps(
            {
                "fold": fold,
                "smoke": smoke,
                "training_sessions": len(selected_sessions),
                "synthetic_training_sessions": int(
                    augmentation_audit.get("mixture_count", 0)
                    if augmentation_audit
                    else 0
                ),
                "freeze_transformer_encoder": bool(
                    config["training"].get("freeze_transformer_encoder", False)
                ),
                "trainable_parameters": trainable,
                "total_parameters": total,
                "uses_validation_labels": False,
                "uses_test_data": False,
            }
        ),
        flush=True,
    )
    started = time.time()
    trainer.fit(model)
    output_checkpoint = (
        run_dir / "sortformer_smoke.nemo"
        if smoke
        else (root / str(config["output_checkpoint"]).format(fold=fold)).resolve()
    )
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_to(str(output_checkpoint))
    metadata = {
        "experiment": config["name"],
        "fold": fold,
        "smoke_only": smoke,
        "development_only": True,
        "uses_validation_labels": False,
        "uses_test_data": False,
        "training_sessions": selected_sessions,
        "training_session_count": len(selected_sessions),
        "synthetic_training_session_count": int(
            augmentation_audit.get("mixture_count", 0) if augmentation_audit else 0
        ),
        "freeze_transformer_encoder": bool(
            config["training"].get("freeze_transformer_encoder", False)
        ),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "base_checkpoint_sha256": sha256_file(checkpoint_path),
        "output_checkpoint": str(output_checkpoint),
        "output_checkpoint_sha256": sha256_file(output_checkpoint),
        "config": config,
        "config_sha256": sha256_file(config_path),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
