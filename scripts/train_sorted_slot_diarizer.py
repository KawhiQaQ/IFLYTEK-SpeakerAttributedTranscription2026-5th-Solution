#!/usr/bin/env python3
"""Train a fold-pure first-appearance speaker-slot sequence decoder."""

from __future__ import annotations

import argparse
import difflib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from sorted_slot_diarizer_model import SortedSpeakerSlotDecoder
from train_oof_segment_role_corrector import segment_features, sha256_file, source_segments


def overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def ordered(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["start_time"]),
            float(row["end_time"]),
            str(row["speaker"]),
        ),
    )


def first_appearance_map(reference: list[dict]) -> dict[str, int]:
    first = {}
    for row in ordered(reference):
        first.setdefault(str(row["speaker"]), float(row["start_time"]))
    speakers = sorted(first, key=lambda speaker: (first[speaker], speaker))
    return {speaker: index for index, speaker in enumerate(speakers)}


def aligned_speaker(row: dict, reference: list[dict]) -> str:
    interval = (float(row["start_time"]), float(row["end_time"]))
    duration = max(interval[1] - interval[0], 0.05)
    tokens = str(row["words"]).split()
    scored = []
    for target in reference:
        target_interval = (
            float(target["start_time"]),
            float(target["end_time"]),
        )
        target_duration = max(target_interval[1] - target_interval[0], 0.05)
        common = overlap(interval, target_interval)
        fraction = common / max(min(duration, target_duration), 0.05)
        lexical = difflib.SequenceMatcher(
            a=tokens,
            b=str(target["words"]).split(),
            autojunk=False,
        ).ratio()
        distance = abs(sum(interval) / 2 - sum(target_interval) / 2)
        scored.append(
            (
                2.5 * fraction + 1.5 * lexical - 0.03 * distance,
                common,
                lexical,
                str(target["speaker"]),
            )
        )
    return max(scored, key=lambda value: (value[0], value[1], value[2]))[3]


def feature_path(
    root: Path, template: str, fold: int, subset: str, session_id: str
) -> Path:
    return root / template.format(
        fold=fold, subset=subset, session_id=session_id
    )


def sequence_features(
    segments: list[dict], payloads: list[dict]
) -> tuple[list[torch.Tensor], torch.Tensor]:
    values = []
    expected_timing = None
    for payload in payloads:
        if (
            payload.get("uses_speaker_labels") is not False
            or payload.get("uses_test_data") is not False
        ):
            raise RuntimeError("Sorted-slot input feature provenance failed")
        embedding, _, timing, _ = segment_features(segments, payload)
        if expected_timing is None:
            expected_timing = timing
        elif not torch.allclose(timing, expected_timing, atol=1e-6):
            raise RuntimeError("Sorted-slot encoder alignment failed")
        values.append(embedding)
    return values, expected_timing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fold = int(config.get("fold", 0) if args.fold is None else args.fold)
    train_sessions = (
        root / f"data/splits/fold_{fold}/train_sessions.txt"
    ).read_text().split()
    validation_sessions = set(
        (root / f"data/splits/fold_{fold}/val_sessions.txt").read_text().split()
    )
    if set(train_sessions) & validation_sessions:
        raise RuntimeError("Sorted-slot fold isolation failed")
    full_reference = json.loads((root / "data/dev/ref.seglst.json").read_text())
    references: dict[str, list[dict]] = defaultdict(list)
    for row in full_reference:
        if str(row["session_id"]) in train_sessions:
            references[str(row["session_id"])].append(row)
    encoder_names = list(config["feature_templates"])
    examples = []

    def add_example(
        session_id: str,
        segments: list[dict],
        targets: torch.Tensor,
        example_type: str,
        source_path: Path,
    ) -> None:
        segments = ordered(segments)
        paths = [
            feature_path(
                root,
                config["feature_templates"][name],
                fold,
                "train",
                session_id,
            )
            for name in encoder_names
        ]
        payloads = [
            torch.load(path, map_location="cpu", weights_only=True) for path in paths
        ]
        values, timing = sequence_features(segments, payloads)
        if len(targets) != len(segments):
            raise RuntimeError(f"Sorted-slot target alignment failed: {session_id}")
        examples.append(
            {
                "session_id": session_id,
                "type": example_type,
                "segments": values,
                "timing": timing,
                "targets": targets,
                "source_sha256": sha256_file(source_path),
                "feature_sha256": [sha256_file(path) for path in paths],
            }
        )

    reference_path = root / "data/dev/ref.seglst.json"
    for session_id in train_sessions:
        rows = ordered(references[session_id])
        mapping = first_appearance_map(rows)
        targets = torch.tensor(
            [mapping[str(row["speaker"])] for row in rows], dtype=torch.long
        )
        add_example(session_id, rows, targets, "reference", reference_path)

    oof_count = 0
    for source_fold in [int(value) for value in config["oof_source_folds"]]:
        if source_fold == fold:
            continue
        source_ids = (
            root / f"data/splits/fold_{source_fold}/val_sessions.txt"
        ).read_text().split()
        for session_id in source_ids:
            if session_id not in references or session_id in validation_sessions:
                continue
            source_path = (
                root
                / f"outputs/{config['oof_source_experiment']}/fold_{source_fold}/sessions/{session_id}.json"
            )
            segments = ordered(source_segments(json.loads(source_path.read_text())))
            mapping = first_appearance_map(references[session_id])
            targets = torch.tensor(
                [mapping[aligned_speaker(row, references[session_id])] for row in segments],
                dtype=torch.long,
            )
            add_example(session_id, segments, targets, "genuine_oof", source_path)
            oof_count += 1
    if args.smoke:
        examples = examples[:1] + [
            example for example in examples if example["type"] == "genuine_oof"
        ][:1]

    maximum_speakers = int(config["model"]["maximum_speakers"])
    if not examples or any(
        int(example["targets"].max()) >= maximum_speakers for example in examples
    ):
        raise RuntimeError("Sorted-slot training examples are invalid")
    seed = int(config["seed"]) + fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(str(config["device"]))
    model = SortedSpeakerSlotDecoder(**config["model"]).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = 1 if args.smoke else int(training["fixed_epochs"])
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        random.Random(seed + epoch).shuffle(examples)
        model.train()
        loss_sum = assignment_correct = segments_seen = count_correct = 0
        for example in examples:
            values = [item.to(device) for item in example["segments"]]
            timing = example["timing"].to(device)
            targets = example["targets"].to(device)
            noise = float(training["feature_noise_std"])
            noisy = [
                F.normalize(item + torch.randn_like(item) * noise, dim=-1)
                for item in values
            ]
            optimizer.zero_grad(set_to_none=True)
            logits, presence = model(noisy, timing)
            speaker_count = int(targets.max()) + 1
            presence_target = torch.zeros(maximum_speakers, device=device)
            presence_target[:speaker_count] = 1
            assignment_loss = F.cross_entropy(
                logits,
                targets,
                label_smoothing=float(training["label_smoothing"]),
            )
            presence_loss = F.binary_cross_entropy_with_logits(
                presence, presence_target
            )
            loss = assignment_loss + float(training["presence_loss_weight"]) * presence_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()
            predicted = logits.argmax(-1)
            loss_sum += float(loss.detach())
            assignment_correct += int((predicted == targets).sum())
            segments_seen += len(targets)
            count_correct += int(int((presence >= 0).sum()) == speaker_count)
        row = {
            "epoch": epoch,
            "loss": loss_sum / max(len(examples), 1),
            "segment_accuracy": assignment_correct / max(segments_seen, 1),
            "count_accuracy": count_correct / max(len(examples), 1),
        }
        history.append(row)
        if epoch == 1 or epoch % int(training["print_frequency"]) == 0 or epoch == epochs:
            print(json.dumps(row), flush=True)

    checkpoint_path = root / str(config["checkpoint_path"]).format(fold=fold)
    if checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.cpu().state_dict(),
        "model_config": config["model"],
        "fold": fold,
        "encoder_names": encoder_names,
        "train_sessions": sorted(set(example["session_id"] for example in examples)),
        "excluded_validation_sessions": sorted(validation_sessions),
        "training_examples": len(examples),
        "genuine_oof_examples": oof_count,
        "uses_training_labels": True,
        "uses_validation_labels": False,
        "uses_test_data": False,
        "history": history,
        "config_sha256": sha256_file(config_path),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(checkpoint, checkpoint_path)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "training_examples": len(examples),
                "genuine_oof_examples": oof_count,
                "final": history[-1],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
