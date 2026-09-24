# Data and model provenance

## Official competition data

The official development set contains 106 labeled conversations. It is the
only competition split used for supervised fitting and cross-validation. The
394-session competition test set is read only by final-inference entry points.
Official audio and annotations are not redistributed in this package.

## External training data

The context-aware speaker model uses 48 labeled windows from the public
VoxConverse v0.3 corpus. Each window is approximately 42 seconds long. The
exact source paths, byte ranges, clipped RTTM segments, and SHA-256 values are
stored in `manifests/voxconverse_fixed_48.json`.

Running `scripts/prepare_voxconverse_fixed.py` materializes precisely that
manifest and verifies every WAV and RTTM hash.

## Public pretrained models

| Component | Public model | Role |
|---|---|---|
| ASR | Qwen3-ASR-1.7B | transcript hypothesis |
| Alignment | Qwen3-ForcedAligner-0.6B | word timestamps |
| ASR | FireRedASR2-AED | Mandarin/dialect text and timestamps |
| ASR | FireRedASR2-LLM | complementary text hypothesis |
| Joint model | MOSS-Transcribe-Diarize | transcript and speaker partition |
| Diarization | NVIDIA offline/streaming Sortformer | candidate speaker activity |
| Speaker encoder | CAMPPlus | identity and partition coherence |
| Speaker encoder | ERes2NetV2 | complementary identity space |
| Speaker encoder | WeSpeaker ResNet34-LM | novel-speaker verification |
| VAD | FSMN VAD | fallback speech activity |

Exact model identifiers, expected local paths, and recorded hashes are in
`configs/public_models.yaml`.

## Custom checkpoints

The package contains eight competition-trained artifacts under `models/`.
Their exact relative paths and hashes are recorded in `checkpoints.sha256` and
validated by `scripts/verify_release.py`.

## Licenses

VoxConverse v0.3 is distributed under CC BY 4.0; original media rights remain
with their owners. Competition data, public checkpoints, and third-party
source code remain subject to their own terms.
