"""
CBAM 注意力模块实现。

CBAM 的全称是：Convolutional Block Attention Module。
它是一种经典的轻量级注意力机制。

CBAM 的核心思想非常直观：
1. 先判断“哪些通道更重要” —— Channel Attention
2. 再判断“哪些空间位置更重要” —— Spatial Attention

也就是说，它会从两个角度筛选特征：
- 通道角度：什么类型的特征更重要？
- 空间角度：图像的哪些位置更值得关注？

在本项目中，CBAM 被用于：
- FE-Module
- FF-Module
- DC-Module

目的都是类似的：
让模型把注意力更多放在“有助于变化检测”的特征上。
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    通道注意力模块。

    它主要回答一个问题：
    “当前特征图中，哪些通道更重要？”

    常见理解方式：
    - 一个通道可以看成一种特征响应
    - 有的通道更关注边缘
    - 有的通道更关注纹理
    - 有的通道更关注某些语义模式

    这里的做法是：
    1) 对输入做全局平均池化
    2) 对输入做全局最大池化
    3) 两条分支共享同一个 MLP（这里用 1x1 卷积实现）
    4) 相加后过 sigmoid，得到每个通道的权重
    5) 用该权重重新缩放输入特征
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)

        # 全局平均池化：保留整体平均响应。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 全局最大池化：保留最强响应。
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享 MLP，这里用两个 1x1 卷积近似全连接层。
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 平均池化分支和最大池化分支分别提取通道描述。
        attn = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))

        # 通过 sigmoid 压到 0~1，再乘回原输入特征。
        return x * self.sigmoid(attn)


class SpatialAttention(nn.Module):
    """
    空间注意力模块。

    它主要回答的问题是：
    “在整张特征图中，哪些空间位置更重要？”

    做法：
    1) 沿通道维做平均池化，得到一张单通道图
    2) 沿通道维做最大池化，得到另一张单通道图
    3) 把两张图拼接后，用卷积生成空间注意力图
    4) 再用 sigmoid 得到 0~1 的空间权重
    5) 把这个权重乘回输入特征
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道平均池化：得到每个位置的平均响应。
        avg_out = torch.mean(x, dim=1, keepdim=True)

        # 通道最大池化：得到每个位置的最强响应。
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接两张单通道图。
        attn = torch.cat([avg_out, max_out], dim=1)

        # 用卷积整合空间信息。
        attn = self.conv(attn)

        # 用 sigmoid 生成空间权重，并乘回输入特征。
        return x * self.sigmoid(attn)


class CBAM(nn.Module):
    """
    CBAM 主模块：先做通道注意力，再做空间注意力。

    顺序是：
    1) Channel Attention
    2) Spatial Attention

    为什么先通道后空间？
    - 这是 CBAM 经典论文中的设计。
    - 先筛选“什么特征重要”，再筛选“这些重要特征出现在哪些位置”。
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
