# Reproducibility Guide

This guide reproduces the released CAST system in two supported modes:

1. download the released competition-trained checkpoints and run inference;
2. retrain every learned component with the frozen configuration, then run inference.

All commands below are executed from the repository root. The recorded platform
is Ubuntu 22.04 with one NVIDIA RTX 3090 (24 GB), CUDA 12.8, and at least 80 GB
of free disk space.

## 1. Clone and install the environment

```bash
git clone https://github.com/KawhiQaQ/IFLYTEK-SpeakerAttributedTranscription2026-5th-Solution.git
cd IFLYTEK-SpeakerAttributedTranscription2026-5th-Solution

conda env create -f configs/environment.yml
conda activate cast
bash scripts/install_runtime.sh
conda deactivate && conda activate cast
```

The installer pins the CUDA build of PyTorch, installs the main runtime, and
creates `.venv-moss/` for MOSS-Transcribe-Diarize. The isolated environment is
required because the released Qwen-ASR and MOSS code use different Transformers
versions.

## 2. Prepare the official data

Download the official development set, test set, and submission example from
the competition page, then use this exact layout:

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

Validate the audio format, session IDs, reference, sample submission, and the
released five-fold split:

```bash
python scripts/prepare_official_data.py .
```

The script requires 16 kHz, 16-bit, mono WAV files and never modifies the
official inputs.

The three original competition archives can be unpacked without manual file
renaming:

```bash
mkdir -p data/sample_submission
unzip dev.zip -d data
unzip test.zip -d data
unzip -j submit_sample.zip submit_sample.json -d data/sample_submission
python scripts/prepare_official_data.py .
```

## 3. Install the released best checkpoints

Download `CAST_checkpoints` from
[Baidu Netdisk](https://pan.baidu.com/s/1TZNc5W9h0CyZPdFnph8RfA?pwd=63pj)
(extraction code: `63pj`). Merge the downloaded model tree into the clone and
normalize the two legacy contextual-checkpoint paths:

```bash
mkdir -p models
cp -a /path/to/CAST_checkpoints/models/. ./models/
python scripts/normalize_checkpoint_layout.py .
python scripts/verify_release.py .
```

`checkpoints.sha256` records all eight learned artifacts used by the final
system. `verify_release.py` accepts the original Baidu archive metadata as well
as the path-normalized release metadata.

Public ASR, diarization, and speaker-encoder checkpoints are not duplicated in
the Baidu archive. `test.sh` and `train.sh` download them automatically according
to `configs/public_models.yaml`. In regions where Hugging Face is unavailable,
set the mirror before running either entry point:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

To download them separately:

```bash
python scripts/download_public_models.py --root . --skip-existing
```

## 4. Inference with the released best checkpoints

Run the complete frozen inference graph:

```bash
bash test.sh
```

The script validates the official inputs and custom checkpoints, downloads any
missing public checkpoints, and executes the following stages in order:

1. speaker-count features;
2. Qwen3-ASR, FireRedASR2-AED, FireRedASR2-LLM, and MOSS transcription;
3. offline and adapted streaming Sortformer candidates;
4. label-free diarization routing;
5. CAMPPlus, ERes2NetV2, and WeSpeaker multiscale speaker features;
6. boundary-aware acoustic graph and multi-ASR consensus;
7. context-aware speaker Transformer refinement;
8. novel-speaker verification and overlap consistency.

The final UTF-8 SegLST prediction is written to:

```text
submissions/final_solution.seglst.json
```

If the public models have already been downloaded, disable network access:

```bash
CAST_SKIP_PUBLIC_MODEL_DOWNLOAD=1 bash test.sh
```

Long inference jobs can be resumed by stage number. For example:

```bash
CAST_SKIP_PUBLIC_MODEL_DOWNLOAD=1 \
bash test.sh --from-stage 9 --to-stage 17
```

Inspect all 17 expanded commands without loading data or models:

```bash
bash test.sh --dry-run
```

The released reference prediction is `reference/final_submission.seglst.json`;
its SHA-256 is
`033b699bfcfbcb9da3a6ae9aa0188d3d31b2de1bc8e82fca8a422d20415d6630`.

## 5. Retrain the complete system

Run:

```bash
bash train.sh
```

This command performs the complete frozen training recipe. It downloads and
verifies the fixed 48-window VoxConverse v0.3 subset described by
`manifests/voxconverse_fixed_48.json`, extracts the required acoustic features,
trains every learned speaker component, deterministically generates Sortformer
mixtures, and adapts the streaming Sortformer.

The orchestrator executes these 16 stages:

| Stage | Learned component or preparation | Frozen configuration |
|---:|---|---|
| 1-2 | speaker-count features and random forest | code defaults and `configs/cv/folds_v1.csv` |
| 3 | CAMPPlus development features | `configs/experiments/v25_multiscale_speaker_graph.yaml` |
| 4 | ERes2NetV2 development features | `configs/experiments/v134_eres2netv2_multiscale_metric_features.yaml` |
| 5-6 | VoxConverse ERes2NetV2 and CAMPPlus features | the same two feature configurations |
| 7 | external initialization of the contextual speaker Transformer | `configs/experiments/contextual_speaker_transformer_external_init.yaml` |
| 8 | CAMPPlus boundary metric | `configs/experiments/v92_boundary_metric_full.yaml` |
| 9 | ERes2NetV2 boundary metric | `configs/deployment/v156_eres2netv2_boundary_metric_full.yaml` |
| 10 | dual-encoder speaker-purity head | `configs/deployment/v156_speaker_purity_full.yaml` |
| 11 | full-development contextual speaker Transformer | `configs/deployment/v534_voxconverse_context_full.yaml` |
| 12-13 | deterministic mixtures and streaming Sortformer adaptation | `configs/experiments/v14_sortformer_synthetic.yaml` and `v14_sortformer_v2_1_synthetic_finetune.yaml` |
| 14-16 | WeSpeaker features, two-fold OOF cache, and novel-speaker verifier | `configs/experiments/v171_wespeaker_resnet34_multiscale_features.yaml` and `configs/deployment/v174_wespeaker_novel_existence_energy_full.yaml` |

All model dimensions, learning rates, batch sizes, fixed epoch counts, random
seeds, and checkpoint paths are stored in the listed configuration files.
Training outputs are written under `models/`; features, deterministic mixtures,
and logs are written under `data/` and `outputs/`.

Inspect the complete training command graph without starting a GPU job:

```bash
bash train.sh --dry-run
```

Training can be resumed at a stage boundary:

```bash
CAST_SKIP_PUBLIC_MODEL_DOWNLOAD=1 \
bash train.sh --from-stage 8 --to-stage 16
```

After training, infer with the newly generated checkpoints instead of enforcing
the released byte hashes:

```bash
CAST_SKIP_PUBLIC_MODEL_DOWNLOAD=1 \
bash test.sh --allow-retrained-checkpoints
```

Floating-point checkpoints may not be byte-identical across GPU architectures,
but the architecture, data, seeds, and optimization schedule are fixed.

## 6. Evaluation

```bash
meeteval-wer tcpwer \
  -r data/dev/ref.seglst.json \
  -h path/to/development_prediction.seglst.json \
  --collar 5
```

For cross-validation, aggregate total errors and total reference tokens before
computing pooled tcpWER; do not average rounded fold-level scores.

## 7. Repository self-check

```bash
make check
make dry-run
```

`make check` compiles all released Python files and verifies the source tree,
reference prediction, and installed custom checkpoints. `make dry-run` expands
the complete training and inference graphs without starting a model job.
