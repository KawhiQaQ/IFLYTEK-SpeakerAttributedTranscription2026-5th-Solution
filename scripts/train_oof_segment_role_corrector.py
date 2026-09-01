#!/usr/bin/env python3
"""Train a segment role corrector on the opposite fold's genuine OOF outputs."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from oof_segment_role_corrector_model import OOFSegmentRoleCorrector


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def source_segments(payload: dict) -> list[dict]:
    if payload.get("uses_validation_labels") is not False or payload.get("uses_test_data") is not False:
        raise RuntimeError("OOF source provenance failed")
    return payload["segments"]


def segment_features(segments: list[dict], acoustic: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    if acoustic.get("uses_speaker_labels") is not False or acoustic.get("uses_test_data") is not False:
        raise RuntimeError("Acoustic feature provenance failed")
    centers = acoustic["centers"].float()
    frames = F.normalize(acoustic["features"].float(), dim=-1)
    speakers = sorted({str(row["speaker"]) for row in segments})
    speaker_index = {speaker: index for index, speaker in enumerate(speakers)}
    pooled, timing, labels = [], [], []
    duration = max(float(acoustic["duration"]), 0.01)
    for index, row in enumerate(segments):
        start, end = float(row["start_time"]), float(row["end_time"])
        selected = (centers >= start) & (centers <= end)
        if not int(selected.sum()):
            selected[(centers - (start + end) / 2).abs().argmin()] = True
        pooled.append(F.normalize(frames[selected].mean(0), dim=0))
        previous_end = float(segments[index - 1]["end_time"]) if index else start
        next_start = float(segments[index + 1]["start_time"]) if index + 1 < len(segments) else end
        timing.append([
            min(end - start, 10.0) / 10.0,
            ((start + end) / 2) / duration,
            min(max(start - previous_end, 0.0), 5.0) / 5.0,
            min(max(next_start - end, 0.0), 5.0) / 5.0,
            min(len(str(row["words"]).split()), 30) / 30.0,
        ])
        labels.append(speaker_index[str(row["speaker"])])
    embeddings = torch.stack(pooled)
    base_labels = torch.tensor(labels, dtype=torch.long)
    prototypes = []
    for speaker in range(len(speakers)):
        values = embeddings[base_labels == speaker]
        initial = F.normalize(values.mean(0), dim=0)
        keep = max(1, int(np.ceil(0.7 * len(values))))
        selected = torch.topk(values @ initial, k=keep).indices
        prototypes.append(F.normalize(values[selected].mean(0), dim=0))
    return embeddings, torch.stack(prototypes), torch.tensor(timing), speakers


def training_targets(segments: list[dict], reference: list[dict], speakers: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    reference_speakers = sorted({str(row["speaker"]) for row in reference})
    reference_index = {speaker: index for index, speaker in enumerate(reference_speakers)}
    source_index = {speaker: index for index, speaker in enumerate(speakers)}
    aligned = []
    weights = []
    for row in segments:
        interval = (float(row["start_time"]), float(row["end_time"]))
        tokens = str(row["words"]).split()
        scored = []
        for target in reference:
            target_interval = (float(target["start_time"]), float(target["end_time"]))
            common = overlap(interval, target_interval)
            fraction = common / max(min(interval[1] - interval[0], target_interval[1] - target_interval[0]), 0.05)
            lexical = difflib.SequenceMatcher(a=tokens, b=str(target["words"]).split(), autojunk=False).ratio()
            distance = abs(sum(interval) / 2 - sum(target_interval) / 2)
            scored.append((2.5 * fraction + 1.5 * lexical - 0.03 * distance, common, lexical, str(target["speaker"])))
        best = max(scored, key=lambda value: (value[0], value[1], value[2]))
        aligned.append(reference_index[best[3]])
        weights.append(max(1, len(tokens)))
    base = torch.tensor([source_index[str(row["speaker"])] for row in segments])
    aligned_tensor = torch.tensor(aligned)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    contingency = torch.zeros(len(speakers), len(reference_speakers))
    for source, target, weight in zip(base, aligned_tensor, weights_tensor):
        contingency[source, target] += weight
    source_rows, reference_columns = linear_sum_assignment(-contingency.numpy())
    reference_to_source = {int(reference): int(source) for source, reference in zip(source_rows, reference_columns)}
    targets = torch.tensor([reference_to_source.get(int(value), -1) for value in aligned_tensor])
    return targets, weights_tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    fold = int(config.get("fold", 0) if args.fold is None else args.fold)
    source_fold = 1 - fold
    target_train_ids = set((root / f"data/splits/fold_{fold}/train_sessions.txt").read_text().split())
    target_val_ids = set((root / f"data/splits/fold_{fold}/val_sessions.txt").read_text().split())
    source_ids = (root / f"data/splits/fold_{source_fold}/val_sessions.txt").read_text().split()
    if not set(source_ids) <= target_train_ids or set(source_ids) & target_val_ids:
        raise RuntimeError("Opposite-fold OOF training identity audit failed")
    reference = json.loads((root / f"data/splits/fold_{source_fold}/val_ref.seglst.json").read_text())
    by_session: dict[str, list[dict]] = defaultdict(list)
    for row in reference:
        by_session[str(row["session_id"])].append(row)
    source_root = root / f"outputs/{config['source_experiment']}/fold_{source_fold}/sessions"
    feature_root = root / config["feature_root"]
    examples = []
    base_correct = total_supervised = 0
    for session_id in source_ids[:2] if args.smoke else source_ids:
        source_path = source_root / f"{session_id}.json"
        feature_path = feature_root / f"{session_id}.pt"
        segments = source_segments(json.loads(source_path.read_text()))
        acoustic = torch.load(feature_path, map_location="cpu", weights_only=True)
        embeddings, prototypes, timing, speakers = segment_features(segments, acoustic)
        targets, weights = training_targets(segments, by_session[session_id], speakers)
        base = torch.tensor([speakers.index(str(row["speaker"])) for row in segments])
        supervised = targets >= 0
        base_correct += int((base[supervised] == targets[supervised]).sum())
        total_supervised += int(supervised.sum())
        examples.append((session_id, embeddings, prototypes, timing, base, targets, weights, sha256_file(source_path), sha256_file(feature_path)))

    # Add label-free public-MOSS predictions from every target-fold training
    # session.  These provide many realistic ASR segmentation/role corruptions;
    # the current validation fold remains completely excluded.
    auxiliary_experiment = config.get("auxiliary_source_experiment")
    auxiliary_count = 0
    if auxiliary_experiment and not args.smoke:
        auxiliary_root = root / "outputs" / auxiliary_experiment
        auxiliary_metadata_path = auxiliary_root / "run_metadata.json"
        auxiliary_metadata = json.loads(auxiliary_metadata_path.read_text())
        if (
            auxiliary_metadata.get("development_only") is not True
            or auxiliary_metadata.get("uses_reference_for_inference") is not False
            or auxiliary_metadata.get("uses_test_data") is not False
        ):
            raise RuntimeError("Auxiliary public-model source provenance failed")
        full_reference = json.loads((root / "data/dev/ref.seglst.json").read_text())
        full_by_session: dict[str, list[dict]] = defaultdict(list)
        for row in full_reference:
            if str(row["session_id"]) in target_train_ids:
                full_by_session[str(row["session_id"])].append(row)
        for session_id in sorted(target_train_ids):
            source_path = auxiliary_root / "sessions" / f"{session_id}.json"
            feature_path = feature_root / f"{session_id}.pt"
            segments = json.loads(source_path.read_text())["segments"]
            acoustic = torch.load(feature_path, map_location="cpu", weights_only=True)
            embeddings, prototypes, timing, speakers = segment_features(segments, acoustic)
            targets, weights = training_targets(segments, full_by_session[session_id], speakers)
            base = torch.tensor([speakers.index(str(row["speaker"])) for row in segments])
            supervised = targets >= 0
            base_correct += int((base[supervised] == targets[supervised]).sum())
            total_supervised += int(supervised.sum())
            examples.append((f"aux:{session_id}", embeddings, prototypes, timing, base, targets, weights, sha256_file(source_path), sha256_file(feature_path)))
            auxiliary_count += 1

    # Local turn swaps are a different failure mode from whole-cluster public
    # model errors: a very short acknowledgement is often attached to the
    # adjacent role.  Build fold-pure episodes from training references only so
    # the contextual model learns to recover these errors acoustically.
    synthetic_episodes = int(config.get("training", {}).get("synthetic_reference_episodes", 0))
    synthetic_count = 0
    if synthetic_episodes and not args.smoke:
        full_reference = json.loads((root / "data/dev/ref.seglst.json").read_text())
        full_by_session: dict[str, list[dict]] = defaultdict(list)
        for row in full_reference:
            if str(row["session_id"]) in target_train_ids:
                full_by_session[str(row["session_id"])].append(row)
        for session_id in sorted(target_train_ids):
            original = sorted(
                full_by_session[session_id],
                key=lambda row: (float(row["start_time"]), float(row["end_time"]), str(row["speaker"])),
            )
            speakers = sorted({str(row["speaker"]) for row in original})
            candidates = [
                index
                for index, row in enumerate(original)
                if float(row["end_time"]) - float(row["start_time"]) <= 1.5
                or len(str(row["words"]).split()) <= 4
            ]
            if len(speakers) < 2 or not candidates:
                continue
            feature_path = feature_root / f"{session_id}.pt"
            acoustic = torch.load(feature_path, map_location="cpu", weights_only=True)
            for episode in range(synthetic_episodes):
                rng = random.Random(int(config["seed"]) * 100000 + int(session_id) * 100 + episode)
                corrupted = [dict(row) for row in original]
                correction_count = max(1, min(len(candidates), round(0.08 * len(original))))
                selected = rng.sample(candidates, correction_count)
                for index in selected:
                    current = str(corrupted[index]["speaker"])
                    contextual = []
                    if index:
                        contextual.append(str(corrupted[index - 1]["speaker"]))
                    if index + 1 < len(corrupted):
                        contextual.append(str(corrupted[index + 1]["speaker"]))
                    alternatives = [speaker for speaker in contextual if speaker != current]
                    if not alternatives:
                        alternatives = [speaker for speaker in speakers if speaker != current]
                    corrupted[index]["speaker"] = rng.choice(alternatives)
                embeddings, prototypes, timing, predicted_speakers = segment_features(corrupted, acoustic)
                targets, weights = training_targets(corrupted, original, predicted_speakers)
                base = torch.tensor([predicted_speakers.index(str(row["speaker"])) for row in corrupted])
                supervised = targets >= 0
                base_correct += int((base[supervised] == targets[supervised]).sum())
                total_supervised += int(supervised.sum())
                examples.append((
                    f"synthetic:{session_id}:{episode}", embeddings, prototypes, timing, base, targets, weights,
                    sha256_file(root / "data/dev/ref.seglst.json"), sha256_file(feature_path),
                ))
                synthetic_count += 1

    seed = int(config["seed"]) + fold
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device(config["device"])
    model = OOFSegmentRoleCorrector(**config["model"]).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    epochs = 1 if args.smoke else int(training["epochs"])
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        random.Random(seed + epoch).shuffle(examples)
        model.train(); loss_sum = correct = decisions = changed = 0
        for _, embeddings, prototypes, timing, base, targets, weights, _, _ in examples:
            embeddings = embeddings.to(device); prototypes = prototypes.to(device); timing = timing.to(device)
            base = base.to(device); targets = targets.to(device); weights = weights.to(device)
            supervised = targets >= 0
            if not int(supervised.sum()):
                continue
            noise = float(training["feature_noise_std"])
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                F.normalize(embeddings + torch.randn_like(embeddings) * noise, dim=-1),
                F.normalize(prototypes + torch.randn_like(prototypes) * noise, dim=-1),
                timing,
                base,
            )
            losses = F.cross_entropy(logits[supervised], targets[supervised], reduction="none", label_smoothing=float(training["label_smoothing"]))
            correction = (targets[supervised] != base[supervised]).float()
            sample_weight = weights[supervised].sqrt() * (1.0 + correction * (float(training["correction_weight"]) - 1.0))
            loss = (losses * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            predicted = logits[supervised].argmax(-1)
            loss_sum += float(loss.detach()); correct += int((predicted == targets[supervised]).sum())
            decisions += int(supervised.sum()); changed += int((predicted != base[supervised]).sum())
        row = {"epoch": epoch, "loss": loss_sum / len(examples), "accuracy": correct / max(decisions, 1), "changed": changed, "base_prior": float(model.base_prior.detach())}
        history.append(row)
        if epoch == 1 or epoch % int(training["print_frequency"]) == 0 or epoch == epochs:
            print(json.dumps(row), flush=True)

    output_path = root / str(config["checkpoint_path"]).format(fold=fold)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.cpu().state_dict(), "model_config": config["model"], "fold": fold,
        "source_fold": source_fold, "train_sessions": source_ids,
        "auxiliary_train_sessions": sorted(target_train_ids) if auxiliary_experiment else [],
        "auxiliary_source_experiment": auxiliary_experiment,
        "synthetic_reference_episodes": synthetic_episodes,
        "synthetic_training_examples": synthetic_count,
        "excluded_validation_sessions": sorted(target_val_ids), "uses_validation_labels": False,
        "uses_test_data": False, "genuine_oof_training": True, "history": history,
        "training_base_accuracy": base_correct / max(total_supervised, 1),
        "config_sha256": sha256_file(config_path), "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(checkpoint, output_path)
    print(json.dumps({"checkpoint": str(output_path), "sha256": sha256_file(output_path), "training_examples": len(examples), "genuine_oof_sessions": len(source_ids), "auxiliary_sessions": auxiliary_count, "synthetic_examples": synthetic_count, "training_base_accuracy": checkpoint["training_base_accuracy"], "final": history[-1]}), flush=True)


if __name__ == "__main__":
    main()
