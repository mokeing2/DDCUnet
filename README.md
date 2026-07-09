# DDCUnet — DFEFT-CNN 双时相变化检测

简体中文说明（项目概要、使用方法、依赖与数据组织）。

## 简介
- 本项目实现 DFEFT-CNN（基于 ResNet34 的双时相变化检测网络），用于遥感影像的变化检测任务（以 LEVIR-CD 数据集为主）。
- 包含训练脚本 `train.py`、评估脚本 `test.py`、数据读取器 `datasets/levir_cd.py`、模型实现在 `models/` 下。

## 目录结构（关键部分）
- `train.py`：训练入口。
- `test.py`：评估/测试入口。
- `config.py`：训练配置（数据路径、超参等）。
- `datasets/levir_cd.py`：LEVIR-CD 数据集读取与增强。
- `models/`：模型实现（`dfeft_cnn.py` 为主网络）。
- `checkpoints_*`、`data/`：示例中用于保存权重与数据的目录（已在 `.gitignore` 中忽略以避免提交大文件）。

## 环境与依赖
建议使用 Python 3.8+。依赖在 `requirements.txt`：

```bash
pip install -r requirements.txt
```

主要依赖：`torch`, `torchvision`, `numpy`, `Pillow`, `tqdm`。

## 数据准备
项目默认使用 LEVIR-CD 数据集，期望的数据目录结构：

```
data/LEVIR-CD/
  ├─ train/
  │   ├─ A/
  │   ├─ B/
  │   └─ label/
  ├─ val/
  │   ├─ A/
  │   ├─ B/
  │   └─ label/
  └─ test/
      ├─ A/
      ├─ B/
      └─ label/
```

可通过 `config.py` 修改 `data_root`，或在运行时通过命令行参数覆盖（例如 `--data_root`）。

## 快速开始

1. 训练：

```bash
python train.py --data_root path/to/data --save_dir path/to/checkpoints
```

常用可选参数：`--batch_size`, `--epochs`, `--lr`, `--crop_size`, `--num_workers`, `--pretrained/--no-pretrained`。

2. 测试/评估：

```bash
python test.py --ckpt path/to/checkpoints/best_model.pt --split test --threshold 0.5
```

## 模型与指标
- 模型输出 logits（未做 sigmoid），训练使用 `BCEWithLogitsLoss`。
- 评估指标包含：precision、recall、f1、iou、accuracy（实现见 `utils/metrics.py`）。

## 检查点与大文件
- 本仓库已添加 `.gitignore`，常见检查点、数据和本地虚拟环境已被忽略。
- 如果仓库历史中已包含大文件，需手动从索引移除（项目已示例执行过 `git rm --cached`）。

## 许可与致谢
- 请根据需要补充许可信息（当前仓库未添加 LICENSE 文件）。

---

如需我将 README 推送到远程仓库（执行 `git push`），或者把 README 英文版补上，请告诉我要执行的操作。 
