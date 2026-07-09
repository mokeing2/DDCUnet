"""
全局训练配置文件。

这个文件的作用可以理解为：
“把训练时最常改的参数集中放到一个地方统一管理”。

对于初学者来说，建议优先熟悉这里的配置项，因为它们决定了：
- 数据从哪里读取
- 权重保存到哪里
- 每次喂多少张图
- 一共训练多少轮
- 学习率是多少
- 用 CPU 还是 GPU

本项目默认配置尽量对齐论文实验设置：
- 数据集：LEVIR-CD
- 训练轮次：200
- 学习率：0.0005
- 裁剪大小：256x256
- 优化器：Adam（在 train.py 中定义）
- 学习率调度：MultiStepLR（在 train.py 中定义）
"""

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TrainConfig:
    """
    训练配置类。

    为什么这里使用 dataclass？
    - 因为它非常适合“存一组配置参数”。
    - 写起来简洁，不需要手动写大量 __init__。
    - 创建对象后，可以直接通过 `cfg.xxx` 的方式访问配置。

    例如：
    - `cfg.batch_size` 表示 batch size
    - `cfg.lr` 表示学习率
    - `cfg.data_root` 表示数据根目录

    这些参数在 `train.py` 中会被读取，
    也可以通过命令行参数进行覆盖。
    """

    # -------------------- 数据相关 --------------------
    # LEVIR-CD 数据集根目录。
    # 期望目录结构大致如下：
    # data_root/
    #   ├─ train/
    #   │   ├─ A/
    #   │   ├─ B/
    #   │   └─ label/
    #   ├─ val/
    #   │   ├─ A/
    #   │   ├─ B/
    #   │   └─ label/
    #   └─ test/
    #       ├─ A/
    #       ├─ B/
    #       └─ label/
    data_root: str = '/root/autodl-tmp/DDCUnet/data/LEVIR-CD'

    # 模型权重保存目录。
    # 训练过程中会在这里保存：
    # - best_model.pt
    # - last_model.pt
    save_dir: str = '/root/autodl-tmp/DDCUnet/checkpoints'

    # -------------------- 训练超参数 --------------------
    # 每个 batch 中包含的样本数量。
    # batch 越大，一次训练看到的数据越多，但显存占用也越高。
    batch_size: int = 8

    # 训练总轮数。
    # 1 个 epoch 表示把整个训练集完整训练一遍。
    epochs: int = 200

    # 初始学习率。
    # 学习率过大可能训练不稳定，过小则收敛太慢。
    lr: float = 5e-4

    # 训练时随机裁剪尺寸。
    # 例如 256 表示把原图裁成 256x256 小块来训练。
    crop_size: int = 256

    # DataLoader 的子进程数量。
    # 通常适当增大可以提高数据读取速度，但也会增加 CPU/内存占用。
    num_workers: int = 4

    # 随机种子。
    # 用来尽量保证每次实验更可复现。
    seed: int = 42

    # 推理时二值化阈值。
    # 当模型输出概率 >= threshold 时，判定为“变化”。
    threshold: float = 0.5

    # 是否使用 ImageNet 预训练的 ResNet34 编码器。
    # 对大多数视觉任务来说，预训练通常能帮助更快收敛。
    pretrained: bool = True

    # 自动选择设备：
    # - 如果检测到 CUDA，就使用 GPU
    # - 否则回退到 CPU
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed: int):
    """
    固定随机种子，提升实验可复现性。

    为什么深度学习需要固定随机种子？
    - 因为训练中很多步骤都带随机性：
      1) 参数初始化
      2) 数据打乱顺序
      3) 随机裁剪 / 翻转等增强
    - 如果不固定种子，每次运行结果可能差异更大。

    这里分别固定了：
    - Python 自带 random
    - NumPy
    - PyTorch CPU
    - PyTorch GPU

    注意：
    即便固定种子，不同的硬件、驱动、CUDA 版本、
    或某些底层算子的实现差异，仍然可能带来少量结果波动。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设置 cuDNN 为确定性模式：
    # - deterministic=True：尽量固定算法路径，结果更稳定
    # - benchmark=False：关闭自动寻找最快算法，减少随机性
    # 代价是：某些情况下训练速度可能略慢。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
