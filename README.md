# DDCUnet — DFEFT-CNN 双时相遥感变化检测

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于 DFEFT-CNN 的遥感影像双时相变化检测项目。使用孪生 ResNet34 编码器 + FE/DC/FF 多尺度融合模块 + CBAM 注意力机制，在 LEVIR-CD 数据集上取得了优于论文基线的表现。

---

## 📑 目录

- [简介](#-简介)
- [模型架构](#-模型架构)
- [实验结果](#-实验结果)
- [项目结构](#-项目结构)
- [环境配置](#-环境配置)
- [数据准备](#-数据准备)
- [快速开始](#-快速开始)
- [核心模块说明](#-核心模块说明)
- [命令行参数](#-命令行参数)
- [致谢](#-致谢)

---

## 📖 简介

变化检测（Change Detection）是遥感影像分析中的核心任务之一，目标是识别同一地理区域在不同时间拍摄的两张图像之间的变化区域（如新建建筑、道路变化、植被变迁等）。

本项目实现了 **DFEFT-CNN**（Dual Feature Extraction and Fusion Transform CNN），一种端到端的双时相变化检测网络，主要特点：

- 🧠 **孪生共享编码器**：A/B 两个时相共享同一个 ResNet34 编码器，确保特征空间一致
- 🔀 **多尺度特征融合**：通过 FE-Module、FF-Module、DC-Module 实现深层语义与浅层细节的有效融合
- 🎯 **CBAM 注意力机制**：通道 + 空间双维度注意力，让模型聚焦于更有判别力的特征
- 📝 **详细的代码注释**：每个文件、每个函数都有中文注释，适合深度学习初学者学习

---

## 🏗 模型架构

```
输入图像 A (T1)                    输入图像 B (T2)
      │                                  │
      ▼                                  ▼
┌─────────────┐                  ┌─────────────┐
│  ResNet34   │   ◄── 共享权重 ──▶│  ResNet34   │
│  Encoder    │                  │  Encoder    │
└──────┬──────┘                  └──────┬──────┘
       │ 5 尺度特征                     │ 5 尺度特征
       ▼                                ▼
┌─────────────┐                  ┌─────────────┐
│  Branch     │                  │  Branch     │
│  Decoder    │                  │  Decoder    │
│  (FE + DC)  │                  │  (FE + DC)  │
└──────┬──────┘                  └──────┬──────┘
       │ 4 尺度解码特征                  │
       └──────────┬──────────────────────┘
                  │
                  ▼
    ┌─────────────────────────┐
    │  FF-Module × 4          │  ← 同尺度双时相融合
    │  (CBAM + Cross-Fusion)  │
    └─────────────┬───────────┘
                  │
                  ▼
    ┌─────────────────────────┐
    │  自顶向下解码器          │  ← 逐步上采样 + 拼接融合
    └─────────────┬───────────┘
                  │
                  ▼
    ┌─────────────────────────┐
    │  1×1 Conv → 变化图 Logits│
    └─────────────────────────┘
```

### 核心子模块

| 模块 | 功能 | 位置 |
|------|------|------|
| **FE-Module** | 多尺度特征提取：融合深层语义与浅层细节，通过 WeightBlock 自适应加权 | `models/modules.py` |
| **FF-Module** | 双时相特征融合：分别做 CBAM 增强后拼接融合 | `models/modules.py` |
| **DC-Module** | 空洞卷积上下文增强：ASPP 多膨胀率 → CBAM 注意力 | `models/modules.py` |
| **CBAM** | 通道注意力 + 空间注意力 | `models/cbam.py` |
| **ASPP** | 空洞空间金字塔池化，多尺度上下文感知 | `models/aspp.py` |
| **WeightBlock** | 自适应权重预测（GAP → FC → Sigmoid） | `models/modules.py` |

---

## 📊 实验结果

### LEVIR-CD 数据集

| 指标 | 论文 (DFEFT-CNN) | 本复现 | 提升 |
|------|:----------------:|:------:|:----:|
| **Precision** | 0.8973 | **0.9126** | +0.0153 |
| **Recall** | 0.8387 | **0.8416** | +0.0029 |
| **F1-Score** | 0.8591 | **0.8757** | +0.0166 |
| **IoU** | 0.7766 | **0.7788** | +0.0022 |
| **Accuracy** | — | **0.9878** | — |

> 🔒 锁定最佳 checkpoint：epoch 59，阈值 0.51，详细数据见 `reproduction_report_levircd.txt`

### 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| 初始学习率 | 5e-4 |
| 学习率策略 | MultiStepLR (milestones=[50,100,150], gamma=0.1) |
| 损失函数 | BCEWithLogitsLoss |
| Batch Size | 8 |
| 训练轮数 | 200 |
| 裁剪尺寸 | 256×256 |
| 数据增强 | 随机裁剪 + 水平/垂直翻转 + 90°旋转 |

---

## 📁 项目结构

```
DDCUnet/
├── train.py                  # 训练脚本（含验证 + checkpoint 保存）
├── test.py                   # 测试/评估脚本
├── config.py                 # 全局训练配置（dataclass）
├── requirements.txt          # Python 依赖
├── .gitignore                # Git 忽略规则（排除权重/数据）
│
├── models/                   # 模型定义
│   ├── dfeft_cnn.py          # DFEFT-CNN 主网络
│   ├── modules.py            # 子模块（FE/FF/DC/WeightBlock）
│   ├── cbam.py               # CBAM 注意力机制
│   └── aspp.py               # ASPP 空洞空间金字塔池化
│
├── datasets/                 # 数据集
│   └── levir_cd.py           # LEVIR-CD 数据加载与增强
│
├── utils/                    # 工具
│   └── metrics.py            # 评价指标（P/R/F1/IoU/Acc + 累计器）
│
├── study_guide/              # 学习资料
│   └── 新手学习稿_DFEFT-CNN_LEVIR-CD.md
│
├── checkpoints_*/            # 模型权重（已 gitignore）
├── data/                     # 数据集（已 gitignore）
│
├── repro_report_step1.txt          # 训练阶段复现报告
└── reproduction_report_levircd.txt # 最终测试复现报告
```

---

## ⚙️ 环境配置

**推荐环境：** Python 3.8+，CUDA 11.8+

```bash
# 克隆仓库
git clone git@github.com:mokeing2/DDCUnet.git
cd DDCUnet

# 安装依赖
pip install -r requirements.txt
```

**主要依赖：**

| 包 | 版本 | 用途 |
|---|------|------|
| torch | 2.8.0 | 深度学习框架 |
| torchvision | 0.23.0 | 预训练模型 + 图像变换 |
| numpy | 2.3.2 | 数值计算 |
| Pillow | 11.3.0 | 图像读取 |
| tqdm | 4.66.2 | 进度条显示 |

---

## 📦 数据准备

本项目使用 **LEVIR-CD** 数据集，请按以下结构组织数据：

```
data/LEVIR-CD/
├── train/
│   ├── A/          # 时相 A 图像（前时相）
│   ├── B/          # 时相 B 图像（后时相）
│   └── label/      # 变化标签图（二值掩码）
├── val/
│   ├── A/
│   ├── B/
│   └── label/
└── test/
    ├── A/
    ├── B/
    └── label/
```

> 📌 **要求**：A、B、label 三个目录中同名文件必须一一对应（均为 PNG 格式）。数据集默认路径可在 `config.py` 中修改，或通过 `--data_root` 命令行参数覆盖。

**数据集统计：**

| 划分 | 样本数 | 图像尺寸 |
|------|:------:|:--------:|
| Train | 445 | 1024×1024 |
| Val | 64 | 1024×1024 |
| Test | 128 | 1024×1024 |

---

## 🚀 快速开始

### 1. 训练

```bash
# 使用默认配置训练（修改 config.py 中的 data_root 和 save_dir）
python train.py

# 或通过命令行指定参数
python train.py \
    --data_root /path/to/LEVIR-CD \
    --save_dir /path/to/checkpoints \
    --batch_size 8 \
    --epochs 200 \
    --lr 0.0005 \
    --crop_size 256 \
    --pretrained
```

训练过程中会自动：
- 在每个 epoch 后验证并打印指标
- 保存 `last_model.pt`（最新）和 `best_model.pt`（验证损失最低）
- 在 tqdm 进度条上实时显示 batch loss 和当前学习率

### 2. 测试

```bash
# 在测试集上评估
python test.py \
    --ckpt checkpoints/best_model.pt \
    --split test \
    --threshold 0.51
```

### 3. 阈值调优

不同阈值会影响 Precision/Recall 的权衡。建议在验证集上扫描阈值（0.30~0.80, step=0.01），选择使 F1 最高的阈值。详细扫描结果见 `reproduction_report_levircd.txt`。

---

## 🔬 核心模块说明

### 5 阶段处理流程

```
阶段 1: 双时相编码
  img_a → ResNet34 → [x0, x1, x2, x3, x4]
  img_b → ResNet34 → [x0, x1, x2, x3, x4]  ← 共享权重

阶段 2: 单分支解码
  [x0~x4] → DC-Module(x4) → FE-Module × 4 → [f1, f2, f3, f4]

阶段 3: 同尺度融合
  (fa_i, fb_i) → FF-Module(CBAM + Concat + Conv) → f_i

阶段 4: 自顶向下恢复
  f4 → Conv → Upsample → Concat(f3) → Conv → ... → x1

阶段 5: 输出
  x1 → ConvBNReLU(64→32) → Conv2d(32→1) → Upsample → Logits
```

### 边界反射问题

模型训练时使用 256×256 随机裁剪，推理时对 1024×1024 大图逐像素滑动窗口预测。需要注意 256 不能被 1024 整除时可能存在边界重叠/间隙，可通过调整裁剪尺寸或重叠推理策略优化。

---

## 🔧 命令行参数

### train.py

| 参数 | 类型 | 默认值 | 说明 |
|--------|------|---------|------|
| `--data_root` | str | config 默认值 | 数据集根目录 |
| `--save_dir` | str | config 默认值 | 权重保存目录 |
| `--batch_size` | int | 8 | 每批样本数 |
| `--epochs` | int | 200 | 训练总轮数 |
| `--lr` | float | 5e-4 | 初始学习率 |
| `--crop_size` | int | 256 | 训练随机裁剪尺寸 |
| `--num_workers` | int | 4 | 数据加载子进程数 |
| `--seed` | int | 42 | 随机种子 |
| `--pretrained` / `--no-pretrained` | bool | True | 是否使用 ImageNet 预训练权重 |

### test.py

| 参数 | 类型 | 默认值 | 说明 |
|--------|------|---------|------|
| `--ckpt` | str | **必填** | Checkpoint 文件路径 |
| `--split` | str | test | 数据划分（train/val/test） |
| `--threshold` | float | 0.5 | 二值化阈值 |

---

## 📝 许可证

本项目采用 [MIT License](LICENSE)。

---

## 🙏 致谢

- [LEVIR-CD 数据集](https://justchenhao.github.io/LEVIR/) — 大规模遥感变化检测基准数据集
- DFEFT-CNN 论文作者
- PyTorch 和 torchvision 团队
