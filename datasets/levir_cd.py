"""
LEVIR-CD 数据集读取与增强文件。

这个文件的职责非常重要，你可以把它理解成：
“负责把硬盘上的图片，整理成模型能直接吃进去的数据格式”。

本文件主要做 3 件事：
1. 读取双时相图像 A / B 和标签 label。
2. 在训练阶段执行随机裁剪、翻转、旋转等增强。
3. 把 PIL 图像转换为 PyTorch Tensor，并完成归一化与标签二值化。

对应论文设置：
- LEVIR-CD 原图通常较大（约 1024x1024）。
- 训练时随机裁剪成 256x256 小块。

为什么变化检测要同时读取 A 和 B？
- 因为 A 表示“前时相图像”，B 表示“后时相图像”。
- 模型要通过比较这两个时相，判断哪里发生了变化。

为什么标签要单独处理？
- 因为标签不是普通 RGB 图像。
- 它本质上是一张 0/1 掩码图：
  - 0 表示没变化
  - 1 表示发生变化
"""

from pathlib import Path
import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


# ImageNet 的均值和标准差。
# 因为本项目编码器使用的是 ImageNet 预训练的 ResNet34，
# 所以输入图像最好也按 ImageNet 的统计量做归一化。
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LEVIRCDDataset(Dataset):
    """
    LEVIR-CD 双时相变化检测数据集类。

    继承自 PyTorch 的 Dataset 后，
    只要你实现了 `__len__` 和 `__getitem__`，
    就可以配合 DataLoader 按 batch 自动读取数据。

    数据目录约定如下：
    root/
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

    其中：
    - A: 时相 A 图像
    - B: 时相 B 图像
    - label: 变化标签图
    """

    def __init__(
        self,
        root: str,
        split: str,
        crop_size: int = 256,
        train: bool = False,
    ):
        """
        初始化数据集。

        参数：
        - root: 数据集根目录。
        - split: 数据划分，通常是 train / val / test。
        - crop_size: 训练时随机裁剪大小。
        - train: 是否处于训练模式。
          - True：启用数据增强。
          - False：不做随机增强。
        """
        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size
        self.train = train

        # 根据 split 组装出三个子目录路径。
        self.dir_a = self.root / split / 'A'
        self.dir_b = self.root / split / 'B'
        self.dir_label = self.root / split / 'label'

        # 默认以 A 目录中的文件名作为主样本列表。
        # 假设 A / B / label 三个目录中的同名文件是互相对应的。
        self.names = sorted(p.name for p in self.dir_a.glob('*.png'))

        # 如果一个样本都没找到，直接报错，提醒数据路径可能不对。
        if not self.names:
            raise FileNotFoundError(f'No PNG files found in {self.dir_a}')

        # 严格检查配对完整性。
        # 这样可以尽早发现：A 有、但 B 或 label 缺失的情况。
        missing = [
            name
            for name in self.names
            if not (self.dir_b / name).exists() or not (self.dir_label / name).exists()
        ]
        if missing:
            raise FileNotFoundError(f'Missing paired files, first few: {missing[:5]}')

    def __len__(self) -> int:
        """
        返回数据集中样本总数。

        DataLoader 会依赖这个函数来知道：
        “这个数据集一共有多少条数据”。
        """
        return len(self.names)

    def _load_triplet(self, name: str):
        """
        读取单个样本三元组：A、B、label。

        为什么 A/B 用 RGB，而 label 用 L？
        - A/B 是普通彩色图像，所以用 RGB 三通道。
        - label 是二值/灰度掩码图，所以用 L（单通道）即可。

        返回：
        - img_a: PIL.Image，RGB
        - img_b: PIL.Image，RGB
        - label: PIL.Image，L
        """
        img_a = Image.open(self.dir_a / name).convert('RGB')
        img_b = Image.open(self.dir_b / name).convert('RGB')
        label = Image.open(self.dir_label / name).convert('L')
        return img_a, img_b, label

    def _random_crop_params(self, w: int, h: int) -> Tuple[int, int, int, int]:
        """
        生成随机裁剪参数。

        参数：
        - w: 图像宽度
        - h: 图像高度

        返回：
        - (top, left, height, width)

        含义解释：
        - top: 裁剪框左上角的纵坐标
        - left: 裁剪框左上角的横坐标
        - height: 裁剪高度
        - width: 裁剪宽度
        """
        # 如果原图尺寸刚好就等于 crop_size，
        # 那就不用随机裁了，直接整张取走。
        if h == self.crop_size and w == self.crop_size:
            return 0, 0, h, w

        # randint(a, b) 会在 [a, b] 范围内随机取整数。
        # 这样裁剪框始终不会超出原图边界。
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        return top, left, self.crop_size, self.crop_size

    def _apply_train_transforms(self, img_a, img_b, label):
        """
        对训练样本执行数据增强。

        这里非常关键的一点是：
        A / B / label 必须做“完全同步”的空间变换。

        为什么必须同步？
        - 因为三者在像素位置上必须一一对应。
        - 如果只翻转 A，不翻转 B 或 label，数据就错位了。
        - 一旦错位，模型就学不到正确的变化关系。

        当前增强流程：
        1) 随机裁剪
        2) 随机水平翻转
        3) 随机垂直翻转
        4) 随机旋转 90 / 180 / 270 度
        """
        # PIL.Image.size 返回顺序是 (width, height)。
        w, h = img_a.size
        top, left, height, width = self._random_crop_params(w, h)

        # 对 A / B / label 同步裁剪。
        img_a = TF.crop(img_a, top, left, height, width)
        img_b = TF.crop(img_b, top, left, height, width)
        label = TF.crop(label, top, left, height, width)

        # 50% 概率做水平翻转。
        if random.random() < 0.5:
            img_a = TF.hflip(img_a)
            img_b = TF.hflip(img_b)
            label = TF.hflip(label)

        # 50% 概率做垂直翻转。
        if random.random() < 0.5:
            img_a = TF.vflip(img_a)
            img_b = TF.vflip(img_b)
            label = TF.vflip(label)

        # 50% 概率做 90 / 180 / 270 度旋转。
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            img_a = TF.rotate(img_a, angle)
            img_b = TF.rotate(img_b, angle)
            label = TF.rotate(label, angle)

        return img_a, img_b, label

    def __getitem__(self, index: int):
        """
        读取并返回第 index 个样本。

        这是 Dataset 中最核心的函数。
        DataLoader 每次取数据时，本质上都会调用它。

        返回值是一个字典，结构如下：
        {
          'name': 文件名,
          'image_a': [3, H, W],
          'image_b': [3, H, W],
          'label': [1, H, W]
        }

        维度解释：
        - 图像是 [通道数, 高, 宽]
        - A/B 为 3 通道
        - label 为 1 通道
        """
        name = self.names[index]

        # 先从磁盘中读取这一组三元数据。
        img_a, img_b, label = self._load_triplet(name)

        # 只有训练集才做随机增强。
        # 验证/测试集通常保持稳定，避免评估结果受随机性干扰。
        if self.train:
            img_a, img_b, label = self._apply_train_transforms(img_a, img_b, label)

        # -------------------- 图像转 Tensor 并归一化 --------------------
        # to_tensor 会把 PIL 图像转换成 float Tensor，
        # 同时把像素值从 [0,255] 缩放到 [0,1]。
        img_a = TF.to_tensor(img_a)
        img_b = TF.to_tensor(img_b)

        # 再按 ImageNet 统计量归一化。
        # 这样更适合送入使用 ImageNet 预训练的 ResNet 编码器。
        img_a = TF.normalize(img_a, IMAGENET_MEAN, IMAGENET_STD)
        img_b = TF.normalize(img_b, IMAGENET_MEAN, IMAGENET_STD)

        # -------------------- 标签处理 --------------------
        # 先把标签转成 numpy 数组，类型设为 float32。
        label = np.array(label, dtype=np.float32)

        # 二值化：像素值 > 127 认为是变化区域，否则是未变化区域。
        # 最终标签只保留 0.0 和 1.0。
        label = (label > 127).astype(np.float32)

        # 转成 Tensor，并在最前面补一个通道维度，
        # 让它从 [H, W] 变成 [1, H, W]。
        label = torch.from_numpy(label).unsqueeze(0)

        return {
            'name': name,
            'image_a': img_a,
            'image_b': img_b,
            'label': label,
        }
