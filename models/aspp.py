"""
ASPP（Atrous Spatial Pyramid Pooling，空洞空间金字塔池化）模块。

这个模块的核心思想是：
“用多种不同膨胀率（dilation）的卷积，同时去看不同尺度范围的上下文信息”。

为什么需要 ASPP？
- 普通卷积的感受野有限。
- 只看小范围邻域时，模型可能无法理解更大的场景关系。
- ASPP 通过多分支空洞卷积，在不过度降低分辨率的情况下扩大感受野。

在这个项目里，ASPP 被放在 DCModule 中，
主要用于增强最深层特征的上下文表达能力。

常见膨胀率设置：
- 1：相当于普通 1x1 卷积
- 12 / 24 / 36：感受野逐步变大
"""

import torch
import torch.nn as nn


class ASPP(nn.Module):
    """
    简化版 ASPP 模块。

    参数说明：
    - in_channels: 输入特征通道数
    - out_channels: 每个分支输出通道数
    - dilations: 每个分支使用的膨胀率列表

    处理流程：
    1) 建立多个并行卷积分支
    2) 每个分支使用不同膨胀率提取特征
    3) 将所有分支输出在通道维拼接
    4) 再通过 1x1 卷积把拼接后的通道压回 out_channels
    """

    def __init__(self, in_channels: int, out_channels: int, dilations=(1, 12, 24, 36)):
        super().__init__()

        branches = []
        for dilation in dilations:
            # 当 dilation=1 时，这里直接使用 1x1 卷积，
            # 相当于一个局部线性映射分支。
            # 当 dilation>1 时，使用 3x3 空洞卷积。
            kernel_size = 1 if dilation == 1 else 3
            padding = 0 if dilation == 1 else dilation

            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # ModuleList 用于存放多个并行子模块。
        self.branches = nn.ModuleList(branches)

        # 所有分支输出拼接后，通道数会变成：
        # out_channels * 分支数
        # 因此再用 1x1 卷积压缩回 out_channels。
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(dilations), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 让输入特征同时经过所有分支。
        feats = [branch(x) for branch in self.branches]

        # 在通道维拼接。
        x = torch.cat(feats, dim=1)

        # 再压缩通道并输出。
        return self.project(x)
