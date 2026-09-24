#!/usr/bin/env python3
"""Build deterministic fold-pure four-speaker overlap mixtures for Sortformer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise RuntimeError(f"Expected mono 16-bit PCM: {path}")
        sample_rate = handle.getframerate()
        audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return audio.astype(np.float32) / 32768.0, sample_rate


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(np.rint(audio * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def intersects_other_speaker(row: dict, session_rows: list[dict]) -> bool:
    start, end = float(row["start_time"]), float(row["end_time"])
    return any(
        other["speaker"] != row["speaker"]
        and min(end, float(other["end_time"]))
        > max(start, float(other["start_time"]))
        for other in session_rows
    )


def choose_distinct_pools(
    rng: random.Random,
    all_pools: list[tuple[str, str]],
    high_pools: list[tuple[str, str]],
    speakers: int,
    minimum_high: int,
) -> list[tuple[str, str]]:
    chosen: list[tuple[str, str]] = []
    used_sessions: set[str] = set()

    def choose_from(candidates: list[tuple[str, str]]) -> tuple[str, str]:
        eligible = [pool for pool in candidates if pool[0] not in used_sessions]
        if not eligible:
            raise RuntimeError("Not enough distinct source sessions for synthetic mix")
        pool = rng.choice(eligible)
        chosen.append(pool)
        used_sessions.add(pool[0])
        return pool

    for _ in range(minimum_high):
        choose_from(high_pools)
    while len(chosen) < speakers:
        choose_from(all_pools)
    rng.shuffle(chosen)
    return chosen


def extract_clips(
    rng: random.Random,
    pool_rows: list[dict],
    audio: np.ndarray,
    sample_rate: int,
    target_seconds: float,
    maximum_clip_seconds: float,
) -> list[np.ndarray]:
    order = list(pool_rows)
    rng.shuffle(order)
    clips: list[np.ndarray] = []
    accumulated = 0.0
    cursor = 0
    while accumulated < target_seconds:
        if cursor and cursor % len(order) == 0:
            rng.shuffle(order)
        row = order[cursor % len(order)]
        cursor += 1
        source_start = float(row["start_time"])
        source_end = float(row["end_time"])
        available = source_end - source_start
        width = min(available, maximum_clip_seconds, target_seconds - accumulated)
        if width < 0.2:
            break
        offset = rng.uniform(0.0, max(0.0, available - width))
        first = max(0, int(round((source_start + offset) * sample_rate)))
        last = min(len(audio), first + int(round(width * sample_rate)))
        clip = audio[first:last].copy()
        if len(clip) < int(0.2 * sample_rate):
            continue
        clips.append(clip)
        accumulated += len(clip) / sample_rate
    if accumulated < 0.8 * target_seconds:
        raise RuntimeError("Speaker pool did not yield enough waveform samples")
    return clips


def order_clips(
    rng: random.Random, clips_by_speaker: dict[int, list[np.ndarray]]
) -> list[tuple[int, np.ndarray]]:
    queues = {speaker: list(clips) for speaker, clips in clips_by_speaker.items()}
    for clips in queues.values():
        rng.shuffle(clips)
    ordered: list[tuple[int, np.ndarray]] = []
    previous: int | None = None
    while any(queues.values()):
        candidates = [speaker for speaker, clips in queues.items() if clips and speaker != previous]
        if not candidates:
            candidates = [speaker for speaker, clips in queues.items() if clips]
        speaker = rng.choice(candidates)
        ordered.append((speaker, queues[speaker].pop()))
        previous = speaker
    return ordered


def active_statistics(segments: list[dict], duration: float) -> dict[str, float]:
    frame_rate = 100
    active = np.zeros(math.ceil(duration * frame_rate), dtype=np.int16)
    for row in segments:
        first = max(0, int(math.floor(float(row["start"]) * frame_rate)))
        last = min(len(active), int(math.ceil(float(row["end"]) * frame_rate)))
        active[first:last] += 1
    speech = active > 0
    return {
        "speech_ratio": float(speech.mean()),
        "overlap_ratio_within_speech": float((active > 1).sum() / max(speech.sum(), 1)),
        "max_simultaneous_speakers": int(active.max(initial=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", help="Fold number, or 'full' for final training")
    parser.add_argument("--max-mixtures", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fold_value = config["fold"] if args.fold is None else args.fold
    fold: int | str = "full" if str(fold_value) == "full" else int(fold_value)
    if fold == "full":
        train_sessions = {
            path.stem for path in (root / "data" / "dev" / "wav").glob("*.wav")
        }
        validation_sessions: set[str] = set()
    else:
        split_dir = (root / "data/splits" / f"fold_{fold}").resolve()
        if split_dir.parent.name != "splits":
            raise RuntimeError("Fold guard failed")
        train_sessions = set(
            (split_dir / "train_sessions.txt").read_text(encoding="utf-8").split()
        )
        validation_sessions = set(
            (split_dir / "val_sessions.txt").read_text(encoding="utf-8").split()
        )
    if train_sessions & validation_sessions:
        raise RuntimeError("Frozen CV leakage")

    settings = config["mixtures"]
    mixture_count = int(settings["count"])
    smoke = args.max_mixtures is not None
    if smoke:
        mixture_count = min(mixture_count, int(args.max_mixtures))
    output_dir = root / str(config["output_dir"]).format(fold=fold)
    if smoke:
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    manifest_path = output_dir / "manifest.jsonl"
    audit_path = output_dir / "audit.json"
    if manifest_path.exists() and not args.overwrite:
        print(json.dumps({"status": "exists", "path": str(manifest_path)}))
        return
    audio_dir, rttm_dir = output_dir / "wav", output_dir / "rttm"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir.mkdir(parents=True, exist_ok=True)

    references = json.loads(
        (root / "data/dev/ref.seglst.json").read_text(encoding="utf-8")
    )
    by_session: dict[str, list[dict]] = defaultdict(list)
    for row in references:
        if row["session_id"] in train_sessions:
            by_session[row["session_id"]].append(row)
    if set(by_session) != train_sessions:
        raise RuntimeError("Incomplete fold training references")
    source_counts = {
        session_id: len({row["speaker"] for row in rows})
        for session_id, rows in by_session.items()
    }

    minimum_clip = float(settings["minimum_clip_seconds"])
    pools: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for session_id, rows in by_session.items():
        for row in rows:
            duration = float(row["end_time"]) - float(row["start_time"])
            if duration >= minimum_clip and not intersects_other_speaker(row, rows):
                pools[(session_id, str(row["speaker"]))].append(row)
    minimum_pool = float(settings["minimum_pool_seconds"])
    pools = {
        key: rows
        for key, rows in pools.items()
        if sum(float(row["end_time"]) - float(row["start_time"]) for row in rows)
        >= minimum_pool
    }
    all_pools = sorted(pools)
    high_threshold = int(settings["high_source_speaker_threshold"])
    high_pools = [pool for pool in all_pools if source_counts[pool[0]] > high_threshold]
    if len(all_pools) < int(settings["speakers"]) or not high_pools:
        raise RuntimeError("Insufficient clean speaker pools")

    fold_seed_offset = 0 if fold == "full" else 1009 * fold
    rng = random.Random(int(config["seed"]) + fold_seed_offset)
    audio_cache: dict[str, np.ndarray] = {}
    sample_rate = 16000
    manifest_rows: list[dict] = []
    mixture_audits: list[dict] = []
    used_source_sessions: set[str] = set()
    duration_seconds = float(settings["duration_seconds"])
    sample_count = int(round(duration_seconds * sample_rate))

    for mixture_index in range(mixture_count):
        selected = choose_distinct_pools(
            rng,
            all_pools,
            high_pools,
            int(settings["speakers"]),
            int(settings["minimum_high_source_speakers_per_mix"]),
        )
        clips_by_speaker: dict[int, list[np.ndarray]] = {}
        for speaker_index, (source_session, _) in enumerate(selected):
            if source_session not in audio_cache:
                audio, rate = read_wav(root / "data/dev/wav" / f"{source_session}.wav")
                if rate != sample_rate:
                    raise RuntimeError("Unexpected development sample rate")
                audio_cache[source_session] = audio
            clips_by_speaker[speaker_index] = extract_clips(
                rng,
                pools[selected[speaker_index]],
                audio_cache[source_session],
                sample_rate,
                float(settings["target_speech_seconds_per_speaker"]),
                float(settings["maximum_clip_seconds"]),
            )
            used_source_sessions.add(source_session)

        mixture = np.zeros(sample_count, dtype=np.float32)
        scheduled: list[dict] = []
        previous_end = 0.25
        fade_samples = int(round(float(settings["fade_seconds"]) * sample_rate))
        for speaker_index, clip in order_clips(rng, clips_by_speaker):
            if scheduled and rng.random() < float(settings["overlap_probability"]):
                overlap_seconds = rng.uniform(
                    float(settings["overlap_seconds_min"]),
                    float(settings["overlap_seconds_max"]),
                )
                start = max(0.0, previous_end - min(overlap_seconds, 0.5 * len(clip) / sample_rate))
            else:
                start = previous_end + rng.uniform(
                    float(settings["gap_seconds_min"]),
                    float(settings["gap_seconds_max"]),
                )
            first = int(round(start * sample_rate))
            if first + len(clip) > sample_count - int(0.2 * sample_rate):
                raise RuntimeError("Synthetic schedule exceeded configured duration")

            rms = float(np.sqrt(np.mean(np.square(clip)) + 1e-12))
            clip = clip * (float(settings["target_clip_rms"]) / max(rms, 1e-4))
            if fade_samples > 0 and len(clip) > 2 * fade_samples:
                ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
                clip[:fade_samples] *= ramp
                clip[-fade_samples:] *= ramp[::-1]
            mixture[first : first + len(clip)] += clip
            end = (first + len(clip)) / sample_rate
            scheduled.append(
                {"speaker": f"spk{speaker_index + 1}", "start": first / sample_rate, "end": end}
            )
            previous_end = end

        peak = float(np.max(np.abs(mixture), initial=0.0))
        if peak > 0.98:
            mixture *= 0.98 / peak
        session_id = f"syn_f{fold}_{mixture_index:03d}"
        wav_path = (audio_dir / f"{session_id}.wav").resolve()
        rttm_path = (rttm_dir / f"{session_id}.rttm").resolve()
        write_wav(wav_path, mixture, sample_rate)
        rttm_lines = [
            f"SPEAKER {session_id} 1 {row['start']:.3f} "
            f"{row['end'] - row['start']:.3f} <NA> <NA> {row['speaker']} <NA> <NA>"
            for row in sorted(scheduled, key=lambda item: (item["start"], item["end"]))
        ]
        rttm_path.write_text("\n".join(rttm_lines) + "\n", encoding="utf-8")
        manifest_rows.append(
            {
                "audio_filepath": str(wav_path),
                "duration": duration_seconds,
                "offset": 0,
                "rttm_filepath": str(rttm_path),
                "uniq_id": session_id,
            }
        )
        stats = active_statistics(scheduled, duration_seconds)
        mixture_audits.append(
            {
                "session_id": session_id,
                "source_speakers": [
                    {"session_id": source, "speaker": speaker, "source_count": source_counts[source]}
                    for source, speaker in selected
                ],
                "segments": len(scheduled),
                **stats,
            }
        )
        print(json.dumps(mixture_audits[-1]), flush=True)

    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    audit = {
        "name": config["name"],
        "fold": fold,
        "smoke_only": smoke,
        "development_only": True,
        "uses_validation_labels": False,
        "uses_test_data": False,
        "source_sessions": sorted(used_source_sessions),
        "validation_sessions": sorted(validation_sessions),
        "source_validation_intersection": sorted(used_source_sessions & validation_sessions),
        "eligible_pool_count": len(all_pools),
        "eligible_high_source_pool_count": len(high_pools),
        "mixture_count": len(manifest_rows),
        "mixtures": mixture_audits,
        "manifest_sha256": sha256_file(manifest_path),
        "config": config,
        "config_sha256": sha256_file(config_path),
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "audit": str(audit_path),
                "mixtures": len(manifest_rows),
                "source_sessions": len(used_source_sessions),
                "uses_validation_labels": False,
                "uses_test_data": False,
            }
        )
    )


if __name__ == "__main__":
    main()
