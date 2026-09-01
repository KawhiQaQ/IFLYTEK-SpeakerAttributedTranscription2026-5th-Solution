#!/usr/bin/env python3
"""Pretrain the exact V174 contextual/purity heads on public external data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from contextual_speaker_metric_model import DualEncoderContextMetric
from external_affinity_examples import parse_rttm
from speaker_purity_model import DualEncoderPurityNet
from train_speaker_metric_fold import pure_frame_labels, sampled_pairs
from train_speaker_purity_head import target_classes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def activity_targets(rows: list[dict], centers: torch.Tensor) -> torch.Tensor:
    speakers = sorted({str(row["speaker"]) for row in rows})
    if len(speakers) < 2:
        raise RuntimeError("External activity target has fewer than two speakers")
    hop = float(torch.median(centers[1:] - centers[:-1]))
    left = centers.float() - hop / 2
    right = centers.float() + hop / 2
    targets = torch.zeros(len(centers), len(speakers), dtype=torch.float32)
    for row in rows:
        index = speakers.index(str(row["speaker"]))
        start = float(row["start_time"])
        end = float(row["end_time"])
        overlap = (torch.minimum(right, torch.tensor(end)) - torch.maximum(left, torch.tensor(start))).clamp_min(0)
        targets[:, index] += overlap / hop
    return targets.clamp_max(1.0)


def load_external(root: Path, config: dict, smoke: bool) -> tuple[list[dict], dict]:
    settings = config["external_pretraining"]
    dataset_root = root / settings["dataset_root"]
    audit_path = dataset_root / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_requirements = settings.get("audit_requirements")
    if audit_requirements:
        failed = {
            key: {"expected": expected, "actual": audit.get(key)}
            for key, expected in audit_requirements.items()
            if audit.get(key) != expected
        }
        if failed:
            raise RuntimeError(
                f"External dataset provenance audit failed: {failed}"
            )
        if (
            str(audit.get("source_split", "")).startswith("test")
            and settings.get("allow_public_external_test_split") is not True
        ):
            raise RuntimeError("Public external test split was not explicitly allowed")
    elif (
        audit.get("source_split") != "train only"
        or audit.get("competition_test_used") is not False
        or audit.get("external_test_used") is not False
    ):
        raise RuntimeError("External dataset provenance audit failed")
    primary_root = root / settings["primary_feature_root"]
    secondary_root = root / settings["secondary_feature_root"]
    sessions = audit["sessions"][:2] if smoke else audit["sessions"]
    examples = []
    for row in sessions:
        session_id = str(row["session_id"])
        primary_path = primary_root / f"{session_id}.pt"
        secondary_path = secondary_root / f"{session_id}.pt"
        rttm_path = dataset_root / "rttm" / f"{session_id}.rttm"
        primary = torch.load(primary_path, map_location="cpu", weights_only=True)
        secondary = torch.load(secondary_path, map_location="cpu", weights_only=True)
        for payload in (primary, secondary):
            if (
                payload.get("uses_test_data") is not False
                or payload.get("uses_speaker_labels") is not False
            ):
                raise RuntimeError(f"External feature provenance failed: {session_id}")
        if not torch.allclose(primary["centers"], secondary["centers"], atol=1e-5):
            raise RuntimeError(f"External feature centers differ: {session_id}")
        targets = activity_targets(parse_rttm(rttm_path), primary["centers"].float())
        examples.append(
            {
                "session_id": (
                    f"{settings.get('dataset_tag', 'external')}::{session_id}"
                ),
                "primary": primary["features"].float(),
                "secondary": secondary["features"].float(),
                "targets": targets,
                "source_sha256": sha256_file(rttm_path),
            }
        )
    return examples, {"audit_path": str(audit_path), "audit_sha256": sha256_file(audit_path)}


def load_official(root: Path, config: dict, fold: int, subset: str, sessions: list[str]) -> list[dict]:
    primary_root = root / "data/speaker_graph" / config["official"]["primary_feature_experiment"] / f"fold_{fold}" / subset
    secondary_root = root / "data/speaker_graph" / config["official"]["secondary_feature_experiment"] / f"fold_{fold}" / subset
    examples = []
    for session_id in sessions:
        primary = torch.load(primary_root / f"{session_id}.pt", map_location="cpu", weights_only=True)
        secondary = torch.load(secondary_root / f"{session_id}.pt", map_location="cpu", weights_only=True)
        if primary.get("uses_test_data") is not False or secondary.get("uses_test_data") is not False:
            raise RuntimeError(f"Official feature provenance failed: {session_id}")
        if not torch.allclose(primary["centers"], secondary["centers"], atol=1e-5):
            raise RuntimeError(f"Official feature centers differ: {session_id}")
        examples.append(
            {
                "session_id": session_id,
                "primary": primary["features"].float(),
                "secondary": secondary["features"].float(),
                "targets": primary["targets"].float(),
            }
        )
    return examples


def crop_jobs(examples: list[dict], settings: dict, seed: int, epoch: int) -> list[tuple[dict, int, int]]:
    window = int(settings["window_frames"])
    per_session = int(settings["windows_per_session"])
    jobs = []
    for position, example in enumerate(examples):
        length = len(example["targets"])
        if length <= window:
            starts = [0]
        else:
            stride = max(1, window // 2)
            candidates = list(range(0, length - window + 1, stride))
            if candidates[-1] != length - window:
                candidates.append(length - window)
            eligible = []
            for start in candidates:
                target = example["targets"][start : start + window]
                indices, labels = pure_frame_labels(
                    target,
                    float(settings["pure_activity_min"]),
                    float(settings["other_activity_max"]),
                )
                if len(indices) >= 8 and len(torch.unique(labels)) >= 2:
                    eligible.append(start)
            if not eligible:
                continue
            rng = random.Random(seed + epoch * 10007 + position)
            starts = (
                rng.sample(eligible, per_session)
                if len(eligible) >= per_session
                else [eligible[rng.randrange(len(eligible))] for _ in range(per_session)]
            )
        jobs.extend((example, start, min(start + window, length)) for start in starts)
    random.Random(seed + epoch * 7919).shuffle(jobs)
    return jobs


def contextual_step(
    model: DualEncoderContextMetric,
    optimizer: torch.optim.Optimizer,
    example: dict,
    start: int,
    end: int,
    settings: dict,
    device: torch.device,
    generator_seed: int,
    train: bool,
) -> tuple[float, int]:
    primary = example["primary"][start:end]
    secondary = example["secondary"][start:end]
    targets = example["targets"][start:end]
    indices, labels = pure_frame_labels(
        targets,
        float(settings["pure_activity_min"]),
        float(settings["other_activity_max"]),
    )
    generator = torch.Generator().manual_seed(generator_seed)
    left, right, pair_target = sampled_pairs(
        labels, int(settings["pairs_per_class_per_session"]), generator
    )
    if not len(pair_target):
        return 0.0, 0
    primary = primary.to(device)
    secondary = secondary.to(device)
    if train:
        noise = float(settings["feature_noise_std"])
        primary = primary + torch.randn_like(primary) * noise
        secondary = secondary + torch.randn_like(secondary) * noise
        optimizer.zero_grad(set_to_none=True)
    use_bfloat16 = settings.get("precision") == "bfloat16"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
        embeddings, change_logits = model.encode(primary, secondary)
        selected = embeddings.squeeze(0)[indices.to(device)]
        pair_logits = model.pair_logits(selected[left.to(device)], selected[right.to(device)])
        pair_loss = F.binary_cross_entropy_with_logits(pair_logits, pair_target.to(device))
        dominant = targets.argmax(dim=1).to(device)
        speech = (targets.sum(dim=1) > 0.5).to(device)
        change_target = torch.zeros(len(targets), device=device)
        change_target[1:] = (dominant[1:] != dominant[:-1]).float()
        valid = speech.clone()
        valid[1:] &= speech[:-1]
        valid[0] = False
        boundary_loss = (
            F.binary_cross_entropy_with_logits(change_logits.squeeze(0)[valid], change_target[valid])
            if valid.any()
            else pair_loss.new_zeros(())
        )
        loss = pair_loss + float(settings["boundary_loss_weight"]) * boundary_loss
    if train:
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite contextual loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip_norm"]))
        if not torch.isfinite(norm):
            raise RuntimeError("Non-finite contextual gradient")
        optimizer.step()
    return float(loss.detach()), len(pair_target)


def train_contextual(
    external: list[dict], official_train: list[dict], official_val: list[dict], config: dict,
    device: torch.device, seed: int, smoke: bool,
) -> tuple[DualEncoderContextMetric, list[dict], float]:
    settings = config["contextual"]
    model = DualEncoderContextMetric(**settings["model"]).to(device)
    logs = []
    external_epochs = 1 if smoke else int(settings["external_epochs"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    for epoch in range(1, external_epochs + 1):
        model.train()
        losses, pairs = [], 0
        for position, (example, start, end) in enumerate(crop_jobs(external, settings, seed, epoch)):
            loss, count = contextual_step(model, optimizer, example, start, end, settings, device, seed + epoch * 100003 + position, True)
            if count:
                losses.append(loss)
                pairs += count
        row = {"head": "contextual", "stage": "external", "epoch": epoch, "loss": float(np.mean(losses)), "pairs": pairs}
        logs.append(row)
        print(json.dumps(row), flush=True)
    official_epochs = 1 if smoke else int(settings["official_epochs"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=official_epochs)
    for epoch in range(1, official_epochs + 1):
        model.train()
        order = list(range(len(official_train)))
        random.Random(seed + 500000 + epoch).shuffle(order)
        losses, pairs = [], 0
        for position, index in enumerate(order):
            example = official_train[index]
            loss, count = contextual_step(model, optimizer, example, 0, len(example["targets"]), settings, device, seed + 700000 + epoch * 1009 + position, True)
            if count:
                losses.append(loss)
                pairs += count
        scheduler.step()
        row = {"head": "contextual", "stage": "official", "epoch": epoch, "loss": float(np.mean(losses)), "pairs": pairs, "learning_rate": optimizer.param_groups[0]["lr"]}
        logs.append(row)
        print(json.dumps(row), flush=True)
    model.eval()
    targets, scores = [], []
    with torch.inference_mode():
        for position, example in enumerate(official_val):
            target = example["targets"]
            indices, labels = pure_frame_labels(target, float(settings["pure_activity_min"]), float(settings["other_activity_max"]))
            left, right, pair_target = sampled_pairs(labels, int(settings["pairs_per_class_per_session"]), torch.Generator().manual_seed(seed + 900000 + position))
            if not len(pair_target):
                continue
            embeddings, _ = model.encode(example["primary"].to(device), example["secondary"].to(device))
            selected = embeddings.squeeze(0)[indices.to(device)]
            score = model.pair_logits(selected[left.to(device)], selected[right.to(device)])
            targets.append(pair_target.numpy())
            scores.append(score.float().cpu().numpy())
    auc = float(roc_auc_score(np.concatenate(targets), np.concatenate(scores)))
    return model, logs, auc


def purity_step(
    model: DualEncoderPurityNet, optimizer: torch.optim.Optimizer, example: dict,
    start: int, end: int, settings: dict, weights: torch.Tensor, device: torch.device,
) -> tuple[float, int, int]:
    primary = example["primary"][start:end].to(device)
    secondary = example["secondary"][start:end].to(device)
    target = target_classes(example["targets"][start:end], int(settings.get("overlap_dilation_frames", 0))).to(device)
    noise = float(settings["feature_noise_std"])
    primary = primary + torch.randn_like(primary) * noise
    secondary = secondary + torch.randn_like(secondary) * noise
    optimizer.zero_grad(set_to_none=True)
    logits = model(primary.unsqueeze(0), secondary.unsqueeze(0)).squeeze(0)
    loss = F.cross_entropy(logits, target, weight=weights)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite purity loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip_norm"]))
    if not torch.isfinite(norm):
        raise RuntimeError("Non-finite purity gradient")
    optimizer.step()
    return float(loss.detach()), int((logits.detach().argmax(1) == target).sum()), len(target)


def class_weights(examples: list[dict], settings: dict, device: torch.device) -> tuple[torch.Tensor, list[int]]:
    counts = torch.zeros(3, dtype=torch.long)
    for example in examples:
        target = target_classes(example["targets"], int(settings.get("overlap_dilation_frames", 0)))
        counts += torch.bincount(target, minlength=3)
    if not torch.all(counts > 0):
        raise RuntimeError(f"Purity classes are degenerate: {counts.tolist()}")
    return (counts.sum().float() / (3.0 * counts.float())).to(device), counts.tolist()


def train_purity(
    external: list[dict], official_train: list[dict], official_val: list[dict], config: dict,
    device: torch.device, seed: int, smoke: bool,
) -> tuple[DualEncoderPurityNet, list[dict], float, list[int], list[int]]:
    settings = config["purity"]
    model = DualEncoderPurityNet(**settings["model"]).to(device)
    logs = []
    external_weights, external_counts = class_weights(external, settings, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    external_epochs = 1 if smoke else int(settings["external_epochs"])
    for epoch in range(1, external_epochs + 1):
        model.train()
        losses, correct, total = [], 0, 0
        for example, start, end in crop_jobs(external, settings, seed + 31, epoch):
            loss, hit, count = purity_step(model, optimizer, example, start, end, settings, external_weights, device)
            losses.append(loss)
            correct += hit
            total += count
        row = {"head": "purity", "stage": "external", "epoch": epoch, "loss": float(np.mean(losses)), "accuracy": correct / total}
        logs.append(row)
        print(json.dumps(row), flush=True)
    official_weights, official_counts = class_weights(official_train, settings, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    official_epochs = 1 if smoke else int(settings["official_epochs"])
    for epoch in range(1, official_epochs + 1):
        model.train()
        order = list(range(len(official_train)))
        random.Random(seed + 800000 + epoch).shuffle(order)
        losses, correct, total = [], 0, 0
        for index in order:
            example = official_train[index]
            loss, hit, count = purity_step(model, optimizer, example, 0, len(example["targets"]), settings, official_weights, device)
            losses.append(loss)
            correct += hit
            total += count
        row = {"head": "purity", "stage": "official", "epoch": epoch, "loss": float(np.mean(losses)), "accuracy": correct / total}
        logs.append(row)
        print(json.dumps(row), flush=True)
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for example in official_val:
            target = target_classes(example["targets"], int(settings.get("overlap_dilation_frames", 0)))
            logits = model(example["primary"].to(device).unsqueeze(0), example["secondary"].to(device).unsqueeze(0)).squeeze(0)
            correct += int((logits.argmax(1).cpu() == target).sum())
            total += len(target)
    return model, logs, correct / total, external_counts, official_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fold = args.fold
    split_root = root / f"data/splits/fold_{fold}"
    train_sessions = (split_root / "train_sessions.txt").read_text().split()
    validation_sessions = (split_root / "val_sessions.txt").read_text().split()
    if set(train_sessions) & set(validation_sessions):
        raise RuntimeError("Fold isolation failed")
    if args.smoke:
        train_sessions = train_sessions[:2]
        validation_sessions = validation_sessions[:2]
    seed = int(config["seed"]) + fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(config["device"])
    started = time.time()
    external, external_audit = load_external(root, config, args.smoke)
    official_train = load_official(root, config, fold, "train", train_sessions)
    official_val = load_official(root, config, fold, "val", validation_sessions)
    contextual, contextual_log, contextual_auc = train_contextual(external, official_train, official_val, config, device, seed, args.smoke)
    purity, purity_log, purity_accuracy, external_counts, official_counts = train_purity(external, official_train, official_val, config, device, seed, args.smoke)
    output_root = (
        root / f"outputs/diagnostics/{config['name']}_fold_{fold}_smoke"
        if args.smoke
        else root / str(config["output_root"]).format(fold=fold)
    )
    contextual_path = output_root / "contextual_metric.pt"
    purity_path = output_root / "purity_head.pt"
    if (contextual_path.exists() or purity_path.exists()) and not args.overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    common = {
        "fold": fold,
        "train_sessions": [*train_sessions, *(row["session_id"] for row in external)],
        "official_train_sessions": train_sessions,
        "external_train_sessions": [row["session_id"] for row in external],
        "excluded_validation_sessions": validation_sessions,
        "uses_validation_labels": False,
        "uses_test_data": False,
        "uses_test_for_model_selection": False,
        "training_scope": "external_public_train_plus_official_fold_train",
        "external_dataset": config["external_pretraining"].get(
            "dataset_description", "public external data"
        ),
        "external_audit": external_audit,
        "config_sha256": sha256_file(config_path),
    }
    torch.save(
        {
            **common,
            "state_dict": {name: value.detach().cpu() for name, value in contextual.state_dict().items()},
            "model_config": config["contextual"]["model"],
            "model_class": "context_only",
            "fixed_external_epochs": 1 if args.smoke else int(config["contextual"]["external_epochs"]),
            "fixed_training_epochs": 1 if args.smoke else int(config["contextual"]["official_epochs"]),
            "validation_pair_auc": contextual_auc,
        },
        contextual_path,
    )
    torch.save(
        {
            **common,
            "state_dict": {name: value.detach().cpu() for name, value in purity.state_dict().items()},
            "model_config": config["purity"]["model"],
            "fixed_external_epochs": 1 if args.smoke else int(config["purity"]["external_epochs"]),
            "fixed_training_epochs": 1 if args.smoke else int(config["purity"]["official_epochs"]),
            "class_counts": official_counts,
            "external_class_counts": external_counts,
            "validation_frame_accuracy": purity_accuracy,
        },
        purity_path,
    )
    metadata = {
        "experiment": config["name"],
        "fold": fold,
        "external_sessions": len(external),
        "official_train_sessions": len(train_sessions),
        "excluded_validation_sessions": validation_sessions,
        "contextual_validation_pair_auc": contextual_auc,
        "purity_validation_frame_accuracy": purity_accuracy,
        "contextual_checkpoint_sha256": sha256_file(contextual_path),
        "purity_checkpoint_sha256": sha256_file(purity_path),
        "uses_validation_labels_for_training": False,
        "uses_test_data": False,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_root / "train_log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [*contextual_log, *purity_log]), encoding="utf-8")
    (output_root / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
