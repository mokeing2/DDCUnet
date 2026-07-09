"""
变化检测评价指标工具文件。

这个文件的作用是：
“把模型输出的变化图，转换成可量化的评估指标”。

对于初学者，训练损失（loss）和评估指标（metrics）要区分开：
- loss：主要服务于训练优化
- metrics：主要服务于结果评价

本文件提供两种用法：
1. `compute_metrics_from_logits`
   - 适合单个 batch 直接计算指标
2. `MetricAccumulator`
   - 适合整套 val/test 数据集逐 batch 累加，最后统一计算指标

核心指标包括：
- Precision（精确率）
- Recall（召回率）
- F1
- IoU
- Accuracy（准确率）

这些指标都依赖四个基础统计量：
- TP: 真正例（预测变化，实际也变化）
- FP: 假正例（预测变化，实际没变化）
- FN: 假负例（预测没变化，实际变化）
- TN: 真负例（预测没变化，实际也没变化）
"""

import torch


@torch.no_grad()
def compute_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """
    从 logits 和真值标签中直接计算一组指标（单次计算版）。

    参数：
    - logits: 模型原始输出，形状通常为 [N, 1, H, W]
    - targets: 标签，形状通常为 [N, 1, H, W]，取值应为 0/1
    - threshold: 二值化阈值，默认 0.5

    处理步骤：
    1) 对 logits 做 sigmoid，得到概率
    2) 根据 threshold 把概率转成 0/1 预测图
    3) 与 targets 对比，计算 TP / FP / FN / TN
    4) 再根据公式计算各项指标
    """
    # logits -> 概率
    probs = torch.sigmoid(logits)

    # 按阈值转为二值预测图。
    preds = (probs >= threshold).float()

    # 再次确保标签是 0/1。
    targets = (targets >= 0.5).float()

    # -------------------- 计算混淆矩阵四项 --------------------
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()

    # eps 是一个很小的数，用来避免分母为 0。
    eps = 1e-7

    # -------------------- 根据公式计算指标 --------------------
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp + eps)

    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn + eps)

    # F1 = 2PR / (P + R)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # IoU = TP / (TP + FP + FN)
    iou = tp / (tp + fp + fn + eps)

    # Accuracy = (TP + TN) / 全部像素
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'iou': iou.item(),
        'accuracy': accuracy.item(),
        'tp': tp.item(),
        'fp': fp.item(),
        'fn': fn.item(),
        'tn': tn.item(),
    }


class MetricAccumulator:
    """
    指标累计器。

    这个类更适合在验证集 / 测试集上使用。

    为什么需要累计器？
    - 因为验证集通常有很多 batch。
    - 如果每个 batch 单独算一个 F1，再简单求平均，结果可能不够严谨。
    - 更稳妥的方式是：
      先把所有 batch 的 TP / FP / FN / TN 累加起来，
      再统一算整套数据集的最终指标。

    使用流程：
    1) 创建对象：meter = MetricAccumulator()
    2) 每个 batch 调用一次 update(...)
    3) 全部结束后调用 compute()
    """

    def __init__(self):
        # 这里用 float 保存累计值即可。
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
        """
        用一个 batch 的预测结果更新累计统计量。
        """
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        targets = (targets >= 0.5).float()

        self.tp += (preds * targets).sum().item()
        self.fp += (preds * (1 - targets)).sum().item()
        self.fn += ((1 - preds) * targets).sum().item()
        self.tn += ((1 - preds) * (1 - targets)).sum().item()

    def compute(self):
        """
        根据累计的 TP / FP / FN / TN 计算最终指标。
        """
        eps = 1e-7
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'accuracy': accuracy,
        }
