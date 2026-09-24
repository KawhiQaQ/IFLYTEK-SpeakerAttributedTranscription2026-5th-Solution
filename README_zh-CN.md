# CAST：上下文感知的说话人归属转写

### 2026 科大讯飞面向说话人转写内容的角色分离挑战赛第 5 名方案

[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0%2Bcu128-EE4C2C?logo=pytorch&logoColor=white)](#环境安装)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](#环境安装)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](#环境安装)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE)

[English](README.md) | 简体中文

本仓库开源 2026 科大讯飞**面向说话人转写内容的角色分离挑战赛第 5 名方案**。任务输入为可能包含重叠语音的多人短对话，系统需要以 SegLST 格式联合预测转写文本、时间戳和说话人归属，官方指标为带时间约束的最小排列词错误率 tcpWER。

本方案融合多 ASR 文本共识、声学说话人图、上下文感知说话人 Transformer、新说话人验证和保守的重叠一致性约束，最终取得 **0.14727** 的线上 tcpWER。

## 方案特点

- **互补转写模型：** Qwen3-ASR、FireRedASR2-AED、FireRedASR2-LLM 和 MOSS-Transcribe-Diarize 提供不同的文本与时间假设。
- **多视角说话人表征：** 通过 CAMPPlus、ERes2NetV2 和 WeSpeaker 表征共同优化流式与离线 Sortformer 说话人轨道。
- **上下文角色建模：** 双编码器 Transformer 利用整段对话上下文处理局部说话人归属歧义。
- **固定公开外部数据：** 上下文说话人模型使用 48 个可重现的 VoxConverse 公开音频窗及 RTTM 标注进行初始化。
- **严格交叉验证：** 所有监督组件均排除当前验证折，选模指标采用 pooled tcpWER 错误数。

## 比赛结果

tcpWER 越低越好。本地成绩为 5 秒 collar 下 fold 0/1 汇总错误数计算的 pooled tcpWER。

| 系统 | 本地 tcpWER | 线上 tcpWER |
|---|---:|---:|
| 多 ASR 文本共识 + 声学说话人图 | 0.14145 | 0.15396 |
| + WeSpeaker 新说话人验证 | 0.13120 | 0.14820 |
| + 上下文感知说话人 Transformer | 等价于 0.13120 | 0.14743 |
| **+ 重叠一致性约束（最终方案）** | **0.12943** | **0.14727** |

最终一致性约束仅修改测试集 10 个会话中的 11 个说话人标签，识别文字、时间戳、片段顺序和片段数量均保持不变。详细分析见[结果与消融](docs/RESULTS.md)。

## 方法概览

系统由四个阶段组成：

1. **多 ASR 文本共识。** Qwen3-ASR、两个 FireRedASR2 解码器和 MOSS-Transcribe-Diarize 生成互补转写；利用独立识别器之间的一致性修正不确定文本，同时保留可靠时间戳。
2. **声学说话人图。** 流式与离线 Sortformer 提供候选说话人活动轨道；CAMPPlus 和 ERes2NetV2 多尺度表征用于计算边界感知的说话人亲和度，并排除声学不一致的合并。
3. **上下文说话人优化。** 三层 Transformer 对整段对话的 ERes2NetV2 与 CAMPPlus 序列进行联合编码，再结合基于 WeSpeaker 的新说话人验证器进行角色仲裁。
4. **重叠一致性约束。** 仅当来源角色映射无歧义，且修改不会引入新冲突时，才修复物理上不可能的同说话人重叠。

完整的问题定义、网络结构、训练目标和推理算法见[方法说明](docs/ARCHITECTURE.md)。

## 模型权重

最佳提交所需的 8 个比赛训练产物可从以下链接下载：

| 权重 | 下载地址 | 提取码 |
|---|---|---|
| CAST 最佳系统权重（8 个产物） | [百度网盘](https://pan.baidu.com/s/1TZNc5W9h0CyZPdFnph8RfA?pwd=63pj) | `63pj` |

下载 `CAST_checkpoints` 后，将其中的 `models/` 目录合并到仓库根目录。
包内目录结构已与公开配置中的权重路径对齐：

```bash
cp -a /path/to/CAST_checkpoints/models/. ./models/
python scripts/normalize_checkpoint_layout.py .
python scripts/verify_release.py .
```

公开 ASR、说话人分离和说话人编码模型可以从官方源下载：

```bash
python scripts/download_public_models.py --root .
```

如果当前网络不能直连 Hugging Face，可以使用镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com \
python scripts/download_public_models.py --root .
```

公开模型的准确 ID 和预期文件见 [`configs/public_models.yaml`](configs/public_models.yaml)。

## 环境安装

最终研发环境为 Python 3.10、PyTorch 2.11.0+cu128、CUDA 12.8 和单张 RTX 3090。

```bash
git clone https://github.com/KawhiQaQ/IFLYTEK-SpeakerAttributedTranscription2026-5th-Solution.git
cd IFLYTEK-SpeakerAttributedTranscription2026-5th-Solution

conda env create -f configs/environment.yml
conda activate cast
bash scripts/install_runtime.sh
conda deactivate && conda activate cast
```

安装脚本会额外创建 `.venv-moss/` 隔离环境，避免
MOSS-Transcribe-Diarize 与 Qwen3-ASR 所需 Transformers 版本冲突。
公开预训练模型及中间缓存建议预留至少 80 GB 磁盘空间。

## 数据准备

请从比赛页面下载官方数据，并按以下结构放置：

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

使用比赛原始压缩包时，可直接执行：

```bash
mkdir -p data/sample_submission
unzip dev.zip -d data
unzip test.zip -d data
unzip -j submit_sample.zip submit_sample.json -d data/sample_submission
python scripts/prepare_official_data.py .
```

本仓库不重新分发官方数据。外部数据和预训练模型来源见[数据与模型来源](docs/DATA_MODEL_PROVENANCE.md)及[第三方声明](THIRD_PARTY_NOTICES.md)。

## 使用最佳权重推理

完成环境安装、官方数据放置和发布权重复制后，执行：

```bash
bash test.sh
```

脚本会自动下载缺失的公开预训练模型，完整执行固定推理流程，
并将 UTF-8 SegLST 结果写入 `submissions/final_solution.seglst.json`。

## 完整重新训练

使用发布的数据清单、模型结构、随机种子和固定 epoch 设置重建全部学习组件：

```bash
bash train.sh
```

该脚本会准备固定的 VoxConverse 公开子集，提取所有声学特征，
训练说话人数、度量、纯度、上下文和新说话人模型，确定性生成
Sortformer 混合语音并适配流式 Sortformer。训练结束后执行：

```bash
CAST_SKIP_PUBLIC_MODEL_DOWNLOAD=1 \
bash test.sh --allow-retrained-checkpoints
```

完整执行顺序、断点续跑、输入输出和每个模型对应的固定配置路径见
[复现指南](docs/REPRODUCIBILITY.md)。

## 本地评测

安装 [MeetEval](https://github.com/fgnt/meeteval) 后运行：

```bash
meeteval-wer tcpwer \
  -r path/to/ref.seglst.json \
  -h path/to/hyp.seglst.json \
  --collar 5
```

选模指标为 fold 0/1 的 pooled tcpWER，而不是两折四舍五入分数的算术平均。具体约束见[验证规范](docs/VALIDATION.md)。

## 仓库结构

```text
configs/                 环境、公开模型、交叉验证和模型配置
docs/                    方法、验证、数据来源和复现文档
manifests/               固定的公开外部数据清单
reference/               发布参考预测及校验目标
scripts/                 训练、推理、评测和提交代码
third_party/             运行所需的精确第三方源码快照
train.sh                 完整训练入口
test.sh                  完整推理入口
```

`data/`、`models/`、`outputs/` 和 `submissions/` 为本地运行时目录，
由程序按需创建并被 Git 忽略。

## 开源协议

本项目代码采用 [Apache License 2.0](LICENSE) 开源。数据集、公开预训练模型、训练权重和第三方依赖仍遵循各自许可证与比赛规则。
