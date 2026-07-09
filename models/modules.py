"""
DFEFT-CNN 的核心子模块实现文件。

如果说 `dfeft_cnn.py` 是整座模型的大框架，
那么这个文件就是框架里面的“零部件工厂”。

这里实现了论文复现中最关键的几个模块：
1. ConvBNReLU：最基础的卷积块
2. ResidualBottleneck：残差瓶颈块
3. WeightBlock：权重提取模块
4. FEModule：多尺度特征提取 / 融合模块
5. FFModule：双时相特征融合模块
6. DCModule：空洞卷积上下文增强模块

理解建议：
- 先看最简单的基础块（ConvBNReLU、ResidualBottleneck）
- 再看注意力和权重（WeightBlock、CBAM）
- 最后看三个核心模块（FE / FF / DC）

这样会更容易看懂整个模型的“拼装逻辑”。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.aspp import ASPP
from models.cbam import CBAM


class ConvBNReLU(nn.Module):
    """
    基础卷积块：Conv -> BatchNorm -> ReLU。

    这是视觉模型里最常见的基本单元之一。

    三部分作用分别是：
    1. Conv：提取局部特征
    2. BatchNorm：稳定训练过程，让特征分布更平稳
    3. ReLU：加入非线性能力，让模型能拟合更复杂的关系

    为什么要封装成一个类？
    - 因为这个结构在很多地方都会重复出现。
    - 封装后代码更简洁，也更统一。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBottleneck(nn.Module):
    """
    残差瓶颈块。

    结构流程：
    1) 1x1 卷积：先降维（减少通道）
    2) 3x3 卷积：在较低通道空间中做主要特征处理
    3) 1x1 卷积：再升维回原通道数
    4) 与输入做残差相加

    为什么叫 bottleneck（瓶颈）？
    - 因为中间通道数被压缩了，像瓶颈一样变窄。
    - 这样能减少参数量和计算量。

    为什么要做残差连接？
    - 可以缓解深层网络训练困难问题。
    - 有利于梯度传播。
    - 让模块更容易学习“在原特征基础上做增量修正”。
    """

    def __init__(self, channels: int):
        super().__init__()

        # 中间通道数取原来的一半，但至少不低于 32。
        # 这样既能压缩计算，又不会压得过小。
        inner = max(channels // 2, 32)

        self.conv1 = nn.Conv2d(channels, inner, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inner)

        self.conv2 = nn.Conv2d(inner, inner, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inner)

        self.conv3 = nn.Conv2d(inner, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保存输入，用于最后做残差相加。
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # 残差连接：输出 = 变换后的特征 + 原始输入
        out = self.relu(out + identity)
        return out


class WeightBlock(nn.Module):
    """
    权重提取模块（对应论文中的 Weight-Block）。

    它的目标是：
    “根据输入特征，自适应地生成一个权重值”。

    这里的结构是：
    1) 全局平均池化
    2) 1x1 卷积降维
    3) ReLU
    4) 1x1 卷积映射到 1 个通道
    5) Sigmoid 归一化到 0~1

    输出是一个形状接近 [N, 1, 1, 1] 的权重图。
    你可以把它理解成：
    “这个特征整体上应该被保留多少”。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        # 隐藏层通道数，类似 SE 模块里的压缩比设计。
        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先通过全局池化把空间维度压缩到 1x1，
        # 再预测一个 0~1 的融合权重。
        return self.net(self.avg_pool(x))


class FEModule(nn.Module):
    """
    FE-Module（Feature Extraction / 多尺度特征提取融合模块）。

    这是单分支解码器中的核心模块之一。

    输入：
    - low_feat: 当前层较浅的编码特征（分辨率更高，细节更多）
    - high_feat: 上一层较深的解码特征（语义更强，分辨率更低）

    目标：
    - 把“高层语义信息”和“低层细节信息”有效结合起来。

    处理流程：
    1) 先把 high_feat 上采样到与 low_feat 同样大小
    2) 通过卷积把两个特征投影到同样通道数
    3) 用 WeightBlock 估计融合权重
    4) 对融合结果施加 CBAM 注意力增强
    5) 再通过卷积 + 残差瓶颈块做细化

    直观理解：
    - low_feat 告诉模型“哪里有边缘、纹理、位置细节”
    - high_feat 告诉模型“这里大概是什么语义区域”
    - FE 模块就是想办法把这两种信息结合起来
    """

    def __init__(self, in_low: int, in_high: int, out_channels: int):
        super().__init__()

        # 先把低层、高层特征都投影到统一通道数，方便后续融合。
        self.low_proj = ConvBNReLU(in_low, out_channels)
        self.high_proj = ConvBNReLU(in_high, out_channels)

        # 分别为低层和高层特征提取权重。
        self.low_weight = WeightBlock(out_channels)
        self.high_weight = WeightBlock(out_channels)

        # 使用 CBAM 对融合结果进一步增强。
        self.attention = CBAM(out_channels)

        # 最后用卷积 + 残差块做细化。
        self.refine = nn.Sequential(
            ConvBNReLU(out_channels, out_channels),
            ResidualBottleneck(out_channels),
        )

    def forward(self, low_feat: torch.Tensor, high_feat: torch.Tensor) -> torch.Tensor:
        # -------------------- 1) 尺寸对齐 --------------------
        # 深层特征通常分辨率更低，因此先上采样到浅层特征大小。
        high_feat = F.interpolate(high_feat, size=low_feat.shape[-2:], mode='bilinear', align_corners=False)

        # -------------------- 2) 通道对齐 --------------------
        low_feat = self.low_proj(low_feat)
        high_feat = self.high_proj(high_feat)

        # -------------------- 3) 权重估计与融合 --------------------
        # low_weight 和 high_weight 各自产生一个 0~1 权重。
        # 这里将两者相加，得到最终融合权重。
        weight = self.low_weight(low_feat) + self.high_weight(high_feat)

        # 用这个权重在低层和高层特征之间做线性融合。
        # 当 weight 更大时，更偏向 low_feat；
        # 当 weight 更小时，更偏向 high_feat。
        fused = weight * low_feat + (1.0 - weight) * high_feat

        # -------------------- 4) 注意力增强 --------------------
        fused = self.attention(fused)

        # -------------------- 5) 细化输出 --------------------
        return self.refine(fused)


class FFModule(nn.Module):
    """
    FF-Module（Feature Fusion / 双时相特征融合模块）。

    这个模块专门负责：
    “把时相 A 和时相 B 在同一尺度上的特征进行融合”。

    输入：
    - feat_a: 时相 A 的某一层特征
    - feat_b: 时相 B 的同层特征

    流程：
    1) feat_a 先做 CBAM 注意力增强
    2) feat_b 也做 CBAM 注意力增强
    3) 两个特征在通道维上拼接
    4) 再通过卷积 + 残差块融合成一个输出特征

    为什么先分别增强，再融合？
    - 因为两个时相的特征可能都包含噪声、冗余信息、无关变化。
    - 先做注意力筛选，有利于后续比较和融合。
    """

    def __init__(self, channels: int):
        super().__init__()

        # A、B 两个分支各自一套 CBAM。
        self.attn_a = CBAM(channels)
        self.attn_b = CBAM(channels)

        # 拼接后通道数翻倍，因此输入是 channels * 2。
        self.fuse = nn.Sequential(
            ConvBNReLU(channels * 2, channels),
            ResidualBottleneck(channels),
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        feat_a = self.attn_a(feat_a)
        feat_b = self.attn_b(feat_b)

        # dim=1 表示在通道维进行拼接。
        return self.fuse(torch.cat([feat_a, feat_b], dim=1))


class DCModule(nn.Module):
    """
    DC-Module（Dilated Convolution Module / 空洞卷积模块）。

    这个模块通常放在最深层特征上使用。

    为什么是最深层？
    - 因为最深层分辨率已经较低，更适合聚合大范围上下文信息。
    - 这时做多尺度空洞卷积，能更有效扩大感受野。

    当前结构流程：
    1) ASPP：并行多膨胀率卷积，提取多尺度上下文
    2) ConvBNReLU
    3) ConvBNReLU
    4) CBAM 注意力增强

    直观理解：
    - ASPP 负责“看更大范围”
    - 后续卷积负责“整理和细化这些上下文信息”
    - CBAM 负责“强调更有用的通道和空间位置”
    """

    def __init__(self, channels: int):
        super().__init__()
        self.aspp = ASPP(channels, channels)
        self.conv1 = ConvBNReLU(channels, channels)
        self.conv2 = ConvBNReLU(channels, channels)
        self.attn = CBAM(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aspp(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attn(x)
        return x
