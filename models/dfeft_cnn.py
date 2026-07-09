"""
DFEFT-CNN 主网络实现文件。

如果把整个项目比作一条流水线，那么这个文件就是“模型结构蓝图”。
它定义了：输入两张图像后，特征是怎样一步步被提取、融合、再恢复成变化图的。

总体结构可以理解为 5 个阶段：
1. 共享权重的孪生编码器（ResNet34）
   - 图像 A 走一遍编码器
   - 图像 B 也走一遍同一个编码器
   - “同一个编码器”意味着参数共享
2. 每个时相分别进入单分支解码器
   - 利用 DC 模块扩大感受野
   - 利用 FE 模块逐级融合深层语义与浅层细节
3. 同尺度双时相特征融合
   - 通过 FF 模块，把 A 与 B 在同一尺度的特征进行融合
4. 多尺度自顶向下解码
   - 逐步上采样，把深层语义信息传递回高分辨率空间
5. 输出 1 通道变化图 logits
   - 注意这里输出的是 logits，不是已经过 sigmoid 的概率

对于初学者，你可以这样理解这类网络：
- 编码器：负责“看懂图像内容”
- 解码器：负责“把抽象特征还原回像素空间”
- 双时相融合：负责“比较前后两张图哪里不一样”
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34

from models.modules import DCModule, FEModule, FFModule, ConvBNReLU


class ResNet34Encoder(nn.Module):
    """
    ResNet34 编码器（单分支）。

    这个编码器会把输入图像逐步下采样，提取出多个尺度的特征图。

    在 DFEFT-CNN 中：
    - 图像 A 会用它编码一次
    - 图像 B 也会用它编码一次
    - 但两次用的是同一套参数

    这就叫“孪生共享权重（Siamese shared weights）”。

    为什么共享权重？
    - 因为 A 和 B 本质上都是同类遥感图像。
    - 我们希望它们被映射到同一种特征空间中，便于比较差异。
    - 共享参数还能减少模型参数量。
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # 根据开关决定是否加载 ImageNet 预训练权重。
        # 使用预训练通常能让编码器更快学到有用的视觉特征。
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)

        # ResNet 的最前面一段通常称为 stem。
        # 这里包含：conv1 -> bn1 -> relu
        # 注意：这里暂时不包含 maxpool，maxpool 单独拿出来放后面。
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )

        # ResNet 后续的主干层。
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor):
        """
        前向传播：输入一张图，输出 5 个尺度特征。

        返回：
        - x0: stem 输出，分辨率较高，细节较多
        - x1: layer1 输出
        - x2: layer2 输出
        - x3: layer3 输出
        - x4: layer4 输出，语义最强、分辨率最低

        为什么要返回多尺度特征？
        - 深层特征语义强，但空间细节少。
        - 浅层特征细节多，但语义抽象能力弱。
        - 解码阶段通常需要把它们结合起来。
        """
        x0 = self.stem(x)

        # maxpool 会进一步降低分辨率，扩大感受野。
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4


class BranchDecoder(nn.Module):
    """
    单时相解码分支。

    这个模块只处理“一个时相”的多尺度特征。
    也就是说：
    - 图像 A 编码后，会进入一次 BranchDecoder
    - 图像 B 编码后，也会进入一次 BranchDecoder

    它的核心思路是：
    1. 最深层特征先通过 DCModule，扩大感受野、增强上下文。
    2. 再从深到浅逐层通过 FEModule，融合高层语义和低层细节。

    最终输出四个解码特征：
    - f1: 最浅层解码特征，分辨率较高
    - f2
    - f3
    - f4: 最深层解码特征，语义较强

    这些特征后续会被送入 FFModule，与另一个时相做同尺度融合。
    """

    def __init__(self, channels=(64, 64, 128, 256, 512), decoder_channels=(64, 128, 256, 512)):
        super().__init__()

        # channels 对应编码器 5 个尺度的通道数。
        c0, c1, c2, c3, c4 = channels

        # decoder_channels 对应解码阶段每层输出通道数。
        d1, d2, d3, d4 = decoder_channels

        # 最深层先过 DC 模块。
        self.dc = DCModule(c4)

        # 然后从深到浅逐级做 FE 融合。
        # fe4: 用 c3 和 deep 融合，输出 d4
        # fe3: 用 c2 和 f4 融合，输出 d3
        # fe2: 用 c1 和 f3 融合，输出 d2
        # fe1: 用 c0 和 f2 融合，输出 d1
        self.fe4 = FEModule(c3, c4, d4)
        self.fe3 = FEModule(c2, d4, d3)
        self.fe2 = FEModule(c1, d3, d2)
        self.fe1 = FEModule(c0, d2, d1)

    def forward(self, feats):
        """
        参数 feats 是编码器输出的五个尺度特征：
        (c0, c1, c2, c3, c4)
        """
        c0, c1, c2, c3, c4 = feats

        # 最深层特征先通过 DC 模块增强。
        deep = self.dc(c4)

        # 再逐层进行特征提取与融合。
        f4 = self.fe4(c3, deep)
        f3 = self.fe3(c2, f4)
        f2 = self.fe2(c1, f3)
        f1 = self.fe1(c0, f2)

        return f1, f2, f3, f4


class DFEFTCNN(nn.Module):
    """
    DFEFT-CNN 主模型。

    前向过程可以概括成下面这条主线：

    第 1 步：A/B 双时相编码
    - 图像 A 经过共享编码器，得到多尺度特征
    - 图像 B 经过同一个共享编码器，得到多尺度特征

    第 2 步：A/B 双分支解码
    - A 的多尺度特征进入单分支解码器
    - B 的多尺度特征进入单分支解码器

    第 3 步：同尺度双时相融合
    - A 与 B 在相同尺度上的特征分别配对
    - 再通过 FFModule 做融合

    第 4 步：自顶向下逐级恢复空间信息
    - 从最深层开始，一边上采样，一边与更浅层特征拼接融合

    第 5 步：输出变化图 logits
    - 最终输出为 1 通道，表示每个像素的变化倾向
    - 数值越大，越偏向“变化”
    - 数值越小，越偏向“未变化”
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # -------------------- 1) 共享孪生编码器 --------------------
        # A 和 B 都使用同一个 encoder 对象，因此参数天然共享。
        self.encoder = ResNet34Encoder(pretrained=pretrained)

        # -------------------- 2) 单分支解码器 --------------------
        # A / B 两个时相都复用同一个解码结构。
        self.decoder = BranchDecoder()

        # -------------------- 3) 同尺度双时相融合模块 --------------------
        # 四个尺度分别配置一个 FF 模块。
        self.ff1 = FFModule(64)
        self.ff2 = FFModule(128)
        self.ff3 = FFModule(256)
        self.ff4 = FFModule(512)

        # -------------------- 4) 最终多尺度解码头 --------------------
        # 这里是一个常见的“自顶向下逐级融合”设计：
        # - 先处理最深层 f4
        # - 再与 f3 融合
        # - 再与 f2 融合
        # - 再与 f1 融合
        self.out4 = ConvBNReLU(512, 256)
        self.out3 = ConvBNReLU(256 + 256, 128)
        self.out2 = ConvBNReLU(128 + 128, 64)
        self.out1 = ConvBNReLU(64 + 64, 64)

        # -------------------- 5) 输出头 --------------------
        # 先把通道压到 32，再用 1x1 卷积输出 1 个通道。
        # 这个 1 通道就是最终的变化图 logits。
        self.head = nn.Sequential(
            ConvBNReLU(64, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """
        输入：
        - img_a: 时相 A 图像，形状通常为 [N, 3, H, W]
        - img_b: 时相 B 图像，形状通常为 [N, 3, H, W]

        输出：
        - logits: 变化图原始输出，形状通常为 [N, 1, H, W]

        注意：
        - 这里输出的是 logits，不是经过 sigmoid 的概率。
        - 训练时交给 BCEWithLogitsLoss。
        - 推理/评估时再做 sigmoid。
        """
        # -------------------- 第 1 步：双时相编码 --------------------
        feats_a = self.encoder(img_a)
        feats_b = self.encoder(img_b)

        # -------------------- 第 2 步：双分支解码 --------------------
        fa1, fa2, fa3, fa4 = self.decoder(feats_a)
        fb1, fb2, fb3, fb4 = self.decoder(feats_b)

        # -------------------- 第 3 步：同尺度双时相融合 --------------------
        f4 = self.ff4(fa4, fb4)
        f3 = self.ff3(fa3, fb3)
        f2 = self.ff2(fa2, fb2)
        f1 = self.ff1(fa1, fb1)

        # -------------------- 第 4 步：顶层到浅层逐级融合 --------------------
        # 最深层先降通道。
        x4 = self.out4(f4)

        # 上采样到与 f3 相同空间尺寸，方便后续拼接。
        x4 = F.interpolate(x4, size=f3.shape[-2:], mode='bilinear', align_corners=False)

        # 与 f3 拼接后融合。
        x3 = self.out3(torch.cat([x4, f3], dim=1))
        x3 = F.interpolate(x3, size=f2.shape[-2:], mode='bilinear', align_corners=False)

        # 与 f2 拼接后融合。
        x2 = self.out2(torch.cat([x3, f2], dim=1))
        x2 = F.interpolate(x2, size=f1.shape[-2:], mode='bilinear', align_corners=False)

        # 与 f1 拼接后融合。
        x1 = self.out1(torch.cat([x2, f1], dim=1))

        # -------------------- 第 5 步：输出变化 logits --------------------
        logits = self.head(x1)

        # 最终再插值回输入原始分辨率，
        # 保证输出大小与输入图像大小一致。
        logits = F.interpolate(logits, size=img_a.shape[-2:], mode='bilinear', align_corners=False)
        return logits
