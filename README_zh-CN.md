# CAST：上下文感知的说话人归属转写

### 2026 科大讯飞面向说话人转写内容的角色分离挑战赛第 5 名方案

[![排名](https://img.shields.io/badge/Rank-5th-C99700)](#比赛结果)
[![榜单 tcpWER](https://img.shields.io/badge/LB%20tcpWER-0.14727-2ea44f)](#比赛结果)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](#环境安装)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE)

[English](README.md) | 简体中文

本仓库开源 2026 科大讯飞**面向说话人转写内容的角色分离挑战赛第 5 名方案**。任务输入为可能包含重叠语音的多人短对话，系统需要以 SegLST 格式联合预测转写文本、时间戳和说话人归属，官方指标为带时间约束的最小排列词错误率 tcpWER。

本方案融合多 ASR 文本共识、声学说话人图、上下文感知说话人 Transformer、新说话人验证和保守的重叠一致性约束，最终取得 **0.14727** 的线上 tcpWER。

## 方案特点

- **互补转写模型：** Qwen3-ASR、FireRedASR2-AED、FireRedASR2-LLM 和 MOSS-Transcribe-Diarize 提供不同的文本与时间假设。
- **多视角说话人表征：** 通过 CAMPPlus、ERes2NetV2 和 WeSpeaker 表征共同优化流式与离线 Sortformer 说话人轨道。
- **上下文角色建模：** 双编码器 Transformer 利用整段对话上下文处理局部说话人归属歧义。
- **分布匹配的外部数据：** 仅依据汇总对话结构统计挑选少量公开 VoxConverse 数据，不使用比赛测试集标签或伪标签。
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

比赛训练权重后续通过百度网盘发布：

| 权重 | 下载地址 | 提取码 |
|---|---|---|
| 上下文感知说话人转写模型 | [百度网盘](https://pan.baidu.com/s/1TZNc5W9h0CyZPdFnph8RfA?pwd=63pj) | `63pj` |

下载 `CAST_checkpoints` 后，将其中的 `models/` 目录合并到仓库根目录。
包内目录结构已与公开配置中的权重路径对齐：

```bash
cp -a /path/to/CAST_checkpoints/models/. ./models/
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
git clone git@github.com:KawhiQaQ/IFLYTEK-SpeakerAttributedTranscription2026-5th-Solution.git
cd IFLYTEK-SpeakerAttributedTranscription2026-5th-Solution

conda env create -f configs/environment.yml
conda activate xunfei-s2
```

## 数据准备

请从比赛页面下载官方数据，并按以下结构放置：

```text
data/
├── dev/
│   ├── wav/*.wav
│   └── ref.seglst.json
├── test/
│   └── wav/*.wav
└── sample_submission/*.json
```

本仓库不重新分发官方数据。外部数据和预训练模型来源见[数据与模型来源](docs/DATA_MODEL_PROVENANCE.md)及[第三方声明](THIRD_PARTY_NOTICES.md)。

## 训练与推理

开源实现按照模型组件组织，而不是按照内部实验编号组织。主要阶段包括：

- Qwen3-ASR、FireRedASR2 和 MOSS-Transcribe-Diarize 推理；
- Sortformer 说话人分离与多尺度说话人特征提取；
- 声学度量模型和说话人纯度模型训练；
- 上下文感知说话人 Transformer 训练；
- 新说话人验证与最终说话人归属转写。

各阶段输入、依赖关系和运行方式见[复现指南](docs/REPRODUCIBILITY.md)。百度网盘权重上传完成后会补充准确下载地址。

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
docs/                    方法、结果、验证、数据来源和复现文档
scripts/                 训练、推理、评测和提交代码
```

`data/`、`models/`、`outputs/`、`submissions/` 和 `third_party/` 均为本地运行时目录，由程序按需创建并被 Git 忽略。

## 开源协议

本项目代码采用 [Apache License 2.0](LICENSE) 开源。数据集、公开预训练模型、训练权重和第三方依赖仍遵循各自许可证与比赛规则。
