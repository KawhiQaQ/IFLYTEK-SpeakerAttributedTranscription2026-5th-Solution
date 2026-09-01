# CAST: Context-Aware Speaker-Attributed Transcription

### 5th-Place Solution for the 2026 iFLYTEK Speaker-Attributed Transcription Challenge

[![Rank](https://img.shields.io/badge/Rank-5th-C99700)](#results)
[![Leaderboard tcpWER](https://img.shields.io/badge/LB%20tcpWER-0.14727-2ea44f)](#results)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](#installation)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE)

English | [简体中文](README_zh-CN.md)

This repository contains the **5th-place solution** to the 2026 iFLYTEK *Speaker-Attributed Transcription Challenge* (面向说话人转写内容的角色分离挑战赛). Given a short multi-speaker conversation with possible overlap, the system jointly predicts the transcript, timestamps, and speaker attribution in SegLST format. The official metric is time-constrained minimum-permutation WER (tcpWER).

The proposed system achieves **0.14727 leaderboard tcpWER** by combining multi-ASR consensus, acoustic speaker graphs, a context-aware speaker Transformer, novel-speaker verification, and a conservative overlap-consistency constraint.

## Highlights

- **Complementary transcription models:** Qwen3-ASR, FireRedASR2-AED, FireRedASR2-LLM, and MOSS-Transcribe-Diarize provide diverse lexical and temporal hypotheses.
- **Multi-view speaker representations:** streaming/offline Sortformer tracks are refined with CAMPPlus, ERes2NetV2, and WeSpeaker embeddings.
- **Context-aware role modeling:** a dual-encoder Transformer uses conversation-level context to resolve locally ambiguous speaker assignments.
- **Distribution-matched external data:** a small public VoxConverse subset is selected using aggregate conversation statistics and used without competition test labels or pseudo-labels.
- **Fold-pure evaluation:** every supervised component excludes the evaluated fold, and model selection uses pooled tcpWER error counts.

## Results

Lower is better. Local results are pooled fold-0/fold-1 tcpWER with a 5-second collar.

| System | Local tcpWER | Leaderboard tcpWER |
|---|---:|---:|
| Multi-ASR consensus + acoustic speaker graph | 0.14145 | 0.15396 |
| + WeSpeaker novel-speaker verification | 0.13120 | 0.14820 |
| + context-aware speaker Transformer | 0.13120 equivalent | 0.14743 |
| **+ overlap-consistency constraint (ours)** | **0.12943** | **0.14727** |

The final consistency constraint changes only 11 speaker labels across 10 test sessions. Recognized words, timestamps, segment order, and segment count remain unchanged. See [Results and Ablations](docs/RESULTS.md) for details.

## Method Overview

The system contains four stages:

1. **Multi-ASR lexical consensus.** Qwen3-ASR, two FireRedASR2 decoders, and MOSS-Transcribe-Diarize generate complementary transcripts. Agreement across independent recognizers is used to replace uncertain lexical regions while preserving reliable timestamps.
2. **Acoustic speaker graph.** Streaming and offline Sortformer hypotheses provide candidate speaker activity. Multiscale CAMPPlus and ERes2NetV2 embeddings define boundary-aware speaker affinities and reject acoustically inconsistent merges.
3. **Context-aware speaker refinement.** A three-layer Transformer jointly encodes ERes2NetV2 and CAMPPlus frame sequences over the complete conversation. Its contextual speaker metric is combined with a WeSpeaker-based novel-speaker verifier.
4. **Overlap-consistency constraint.** Physically impossible same-speaker overlaps are repaired only when the source-to-refined role mapping is unambiguous and the edit introduces no new conflict.

The complete formulation, network design, objectives, and inference algorithm are described in [Method](docs/ARCHITECTURE.md).

## Checkpoints

Competition-trained checkpoints will be released separately:

| Checkpoint | Download | Extraction code |
|---|---|---|
| Context-aware speaker transcription models | [Baidu Netdisk](https://pan.baidu.com/s/1TZNc5W9h0CyZPdFnph8RfA?pwd=63pj) | `63pj` |

After downloading `CAST_checkpoints`, merge its `models/` directory into the
repository root. The checkpoint paths are already arranged to match the
released configurations:

```bash
cp -a /path/to/CAST_checkpoints/models/. ./models/
```

Public ASR, diarization, and speaker encoders can be downloaded from their official sources:

```bash
python scripts/download_public_models.py --root .
```

For restricted networks, a Hugging Face mirror can be used without changing local paths:

```bash
HF_ENDPOINT=https://hf-mirror.com \
python scripts/download_public_models.py --root .
```

The exact public model IDs and expected files are listed in [`configs/public_models.yaml`](configs/public_models.yaml).

## Installation

The recorded environment uses Python 3.10, PyTorch 2.11.0+cu128, CUDA 12.8, and one RTX 3090.

```bash
git clone git@github.com:KawhiQaQ/IFLYTEK-SpeakerAttributedTranscription2026-5th-Solution.git
cd IFLYTEK-SpeakerAttributedTranscription2026-5th-Solution

conda env create -f configs/environment.yml
conda activate xunfei-s2
```

## Data Preparation

Download the official data from the competition page and arrange it locally as follows:

```text
data/
├── dev/
│   ├── wav/*.wav
│   └── ref.seglst.json
├── test/
│   └── wav/*.wav
└── sample_submission/*.json
```

The official dataset is not redistributed. External-data and pretrained-model provenance are documented in [Data and Model Provenance](docs/DATA_MODEL_PROVENANCE.md) and [Third-Party Notices](THIRD_PARTY_NOTICES.md).

## Training and Inference

The implementation is organized by model component rather than by leaderboard experiment. The main stages are:

- ASR inference with Qwen3-ASR, FireRedASR2, and MOSS-Transcribe-Diarize;
- Sortformer diarization and multiscale speaker-feature extraction;
- acoustic metric and speaker-purity training;
- context-aware speaker Transformer training;
- novel-speaker verification and final speaker-attributed transcription.

Detailed commands, expected inputs, and stage dependencies are provided in the [Reproducibility Guide](docs/REPRODUCIBILITY.md). The trained checkpoint download will be inserted once the Baidu Netdisk upload is available.

## Evaluation

Install [MeetEval](https://github.com/fgnt/meeteval) and run:

```bash
meeteval-wer tcpwer \
  -r path/to/ref.seglst.json \
  -h path/to/hyp.seglst.json \
  --collar 5
```

The promotion metric is pooled fold-0/fold-1 tcpWER rather than the unweighted average of rounded fold scores. See [Validation Protocol](docs/VALIDATION.md).

## Repository Structure

```text
configs/                 environment, public-model, CV, and model configurations
docs/                    method, results, validation, provenance, and reproduction
scripts/                 training, inference, evaluation, and submission code
```

Runtime directories such as `data/`, `models/`, `outputs/`, `submissions/`, and `third_party/` are created locally and ignored by Git.

## License

The project code is released under the [Apache License 2.0](LICENSE). Datasets, pretrained models, trained weights, and third-party dependencies remain subject to their respective licenses and competition rules.
