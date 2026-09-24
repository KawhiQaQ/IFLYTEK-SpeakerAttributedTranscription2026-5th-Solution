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
- **Fixed public external data:** the contextual speaker model is initialized on 48 reproducible, publicly available VoxConverse windows with RTTM annotations.
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

The eight competition-trained artifacts used by the best submission are available here:

| Checkpoint | Download | Extraction code |
|---|---|---|
| CAST best-system checkpoints (8 artifacts) | [Baidu Netdisk](https://pan.baidu.com/s/1TZNc5W9h0CyZPdFnph8RfA?pwd=63pj) | `63pj` |

After downloading `CAST_checkpoints`, merge its `models/` directory into the
repository root. The checkpoint paths are already arranged to match the
released configurations:

```bash
cp -a /path/to/CAST_checkpoints/models/. ./models/
python scripts/normalize_checkpoint_layout.py .
python scripts/verify_release.py .
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
git clone https://github.com/KawhiQaQ/IFLYTEK-SpeakerAttributedTranscription2026-5th-Solution.git
cd IFLYTEK-SpeakerAttributedTranscription2026-5th-Solution

conda env create -f configs/environment.yml
conda activate cast
bash scripts/install_runtime.sh
conda deactivate && conda activate cast
```

The installer creates a separate `.venv-moss/` environment for
MOSS-Transcribe-Diarize so that its Transformers version does not conflict with
Qwen3-ASR. At least 80 GB of free disk space is recommended for public models
and generated caches.

## Data Preparation

Download the official data from the competition page and arrange it locally as follows:

```text
data/
├── dev/
│   ├── wav/*.wav
│   └── ref.seglst.json
├── test/
│   └── wav/*.wav
└── sample_submission/
    └── submit_sample.json
```

With the original competition archives, the layout can be created directly:

```bash
mkdir -p data/sample_submission
unzip dev.zip -d data
unzip test.zip -d data
unzip -j submit_sample.zip submit_sample.json -d data/sample_submission
python scripts/prepare_official_data.py .
```

The official dataset is not redistributed. External-data and pretrained-model provenance are documented in [Data and Model Provenance](docs/DATA_MODEL_PROVENANCE.md) and [Third-Party Notices](THIRD_PARTY_NOTICES.md).

## Inference with the Best Checkpoints

After installing the environment, arranging the official data, and copying the
released checkpoints, run:

```bash
bash test.sh
```

Missing public pretrained models are downloaded automatically. The complete
frozen inference graph writes the final UTF-8 SegLST prediction to
`submissions/final_solution.seglst.json`.

## Full Training

To rebuild every learned component with the released data manifest, model
architecture, random seeds, and fixed epoch budgets, run:

```bash
bash train.sh
```

The script prepares the fixed public VoxConverse subset, extracts all acoustic
features, trains the speaker-count, metric, purity, contextual, and
novel-speaker models, generates deterministic Sortformer mixtures, and adapts
the streaming Sortformer. After training, run:

```bash
CAST_SKIP_PUBLIC_MODEL_DOWNLOAD=1 \
bash test.sh --allow-retrained-checkpoints
```

Exact commands, stage order, resumable execution, inputs, outputs, and frozen
configuration paths are provided in the
[Reproducibility Guide](docs/REPRODUCIBILITY.md).

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
docs/                    method, validation, provenance, and reproduction
manifests/               fixed public external-data manifest
reference/               released reference prediction and checksum target
scripts/                 training, inference, evaluation, and submission code
third_party/             exact source snapshots required by the pipeline
train.sh                 complete training entry point
test.sh                  complete inference entry point
```

Runtime directories such as `data/`, `models/`, `outputs/`, and `submissions/`
are created locally and ignored by Git.

## License

The project code is released under the [Apache License 2.0](LICENSE). Datasets, pretrained models, trained weights, and third-party dependencies remain subject to their respective licenses and competition rules.
