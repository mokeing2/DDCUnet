"""
测试脚本：加载已经训练好的 checkpoint，在指定数据集划分上评估模型指标。

你可以把它理解成“考试脚本”：
- `train.py` 负责训练模型，相当于“上课 + 做题”。
- `test.py` 负责拿训练好的模型去正式评估，相当于“考试打分”。

默认情况下：
- 评估的是 `test` 集。
- 阈值是 0.5。

你也可以通过命令行切换：
- `--split train`：看训练集表现
- `--split val`：看验证集表现
- `--split test`：看测试集表现

注意：
这个脚本只负责评估，不会更新模型参数。
因此这里不会出现 optimizer、loss.backward()、optimizer.step() 等训练步骤。
"""

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig
from datasets.levir_cd import LEVIRCDDataset
from models.dfeft_cnn import DFEFTCNN
from utils.metrics import MetricAccumulator


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """
    在给定数据集上评估二值变化检测指标。

    参数：
    - model: 已加载权重的模型。
    - loader: 测试/验证数据加载器。
    - device: 运行设备（CPU 或 GPU）。
    - threshold: 概率转二值图时使用的阈值。

    返回：
    - 一个字典，包含 precision / recall / f1 / iou / accuracy。

    为什么这里不计算 loss？
    - 因为测试阶段更关注最终检测效果指标。
    - loss 更常用于训练过程中的优化和监控。
    """
    # 切换模型到评估模式。
    model.eval()

    # 用于累计全数据集的 TP/FP/FN/TN。
    meter = MetricAccumulator()

    # 遍历整个数据集。
    for batch in tqdm(loader, desc='test'):
        image_a = batch['image_a'].to(device)
        image_b = batch['image_b'].to(device)
        label = batch['label'].to(device)

        # 前向传播得到变化图 logits。
        logits = model(image_a, image_b)

        # 把当前 batch 的统计量累加到总指标中。
        meter.update(logits, label, threshold)

    # 返回整套数据集的最终指标。
    return meter.compute()


def main():
    """
    测试脚本入口。

    主要流程：
    1) 读取命令行参数
    2) 构建数据集和 DataLoader
    3) 创建模型并加载 checkpoint
    4) 在指定 split 上运行评估
    5) 打印最终指标
    """
    parser = argparse.ArgumentParser()

    # 必填参数：checkpoint 路径。
    # 例如：--ckpt checkpoints/best_model.pt
    parser.add_argument('--ckpt', type=str, required=True)

    # 选择评估数据划分。
    # choices 的意思是：只允许这三个值，防止手误输入非法内容。
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])

    # 二值化阈值。
    # 如果你怀疑 0.5 不是最佳阈值，可以尝试改成 0.4 / 0.6 等重新测试。
    parser.add_argument('--threshold', type=float, default=0.5)

    args = parser.parse_args()

    # 读取默认配置。
    # 这里主要会用到：数据路径、设备、crop_size、num_workers。
    cfg = TrainConfig()
    device = torch.device(cfg.device)

    # -------------------- 构建数据集与加载器 --------------------
    # train=False 表示不使用随机增强，保持评估公平性。
    dataset = LEVIRCDDataset(cfg.data_root, split=args.split, crop_size=cfg.crop_size, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # -------------------- 构建模型并加载权重 --------------------
    # 评估时这里设为 pretrained=False 没问题，
    # 因为真正的模型参数会被 checkpoint 中的权重覆盖掉。
    model = DFEFTCNN(pretrained=False).to(device)

    # map_location=device 表示：
    # 无论 checkpoint 当初是在 CPU 还是 GPU 上保存的，
    # 现在都按当前 device 加载。
    checkpoint = torch.load(args.ckpt, map_location=device)

    # checkpoint['model'] 中保存的是模型参数字典。
    model.load_state_dict(checkpoint['model'])

    # -------------------- 开始评估 --------------------
    metrics = evaluate(model, loader, device, args.threshold)

    # -------------------- 打印结果 --------------------
    print('split:', args.split)
    print(f"precision={metrics['precision']:.4f}")
    print(f"recall={metrics['recall']:.4f}")
    print(f"f1={metrics['f1']:.4f}")
    print(f"iou={metrics['iou']:.4f}")
    print(f"accuracy={metrics['accuracy']:.4f}")


# 当直接运行 `python test.py ...` 时，从这里进入。
if __name__ == '__main__':
    main()
