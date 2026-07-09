"""
训练脚本：负责把整个变化检测项目“串起来”运行。

你可以把这个文件理解成项目的“总调度中心”，它主要做 5 件事：
1. 读取训练配置（例如数据路径、学习率、训练轮数）。
2. 构建训练集 / 验证集，以及对应的 DataLoader。
3. 创建模型、损失函数、优化器、学习率调度器。
4. 执行每一轮训练，并在每轮结束后做验证。
5. 保存最近一次模型（last）和验证效果最好的模型（best）。

本项目默认配置参考论文实验设置：
- 损失函数：BCEWithLogitsLoss
- 优化器：Adam
- 学习率策略：MultiStepLR
- 训练轮数：200
- 裁剪尺寸：256
- batch size：8

为什么输出是 logits 而不是概率？
- 因为模型最后一层没有主动做 sigmoid。
- 训练时直接配合 BCEWithLogitsLoss 使用，这样数值更稳定。
- 评估时再通过 sigmoid 把 logits 转成 0~1 概率。
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig, set_seed
from datasets.levir_cd import LEVIRCDDataset
from models.dfeft_cnn import DFEFTCNN
from utils.metrics import MetricAccumulator


@torch.no_grad()
def evaluate(model, loader, device, criterion, threshold=0.5):
    """
    在验证集上评估模型表现。

    参数说明：
    - model: 已经创建好的变化检测模型。
    - loader: 验证集数据加载器，里面会不断产出 batch。
    - device: 运行设备，通常是 'cuda' 或 'cpu'。
    - criterion: 损失函数，这里通常是 BCEWithLogitsLoss。
    - threshold: 将概率图二值化的阈值，默认 0.5。

    返回值：
    - 一个字典，包含 loss / precision / recall / f1 / iou / accuracy。

    这里为什么要写 @torch.no_grad()？
    - 因为验证阶段不需要反向传播。
    - 这样可以节省显存、加快速度。
    - 同时避免误把验证过程也记录进计算图。
    """
    # 切换到评估模式。
    # 这会让 BatchNorm、Dropout 等层进入“推理状态”。
    model.eval()

    # 指标累计器：不是每个 batch 单独给结果，
    # 而是把整套验证集的 TP/FP/FN/TN 累起来，最后统一算指标。
    meter = MetricAccumulator()

    # 用于统计整套验证集的平均 loss。
    loss_sum = 0.0
    count = 0

    # tqdm 用来显示进度条，方便你看到验证跑到了哪里。
    for batch in tqdm(loader, desc='eval', leave=False):
        # 从数据字典中取出双时相图像和标签，并搬到对应设备上。
        image_a = batch['image_a'].to(device)
        image_b = batch['image_b'].to(device)
        label = batch['label'].to(device)

        # 前向传播：输入时相 A 和时相 B，输出变化图 logits。
        logits = model(image_a, image_b)

        # 计算当前 batch 的损失。
        loss = criterion(logits, label)

        # 为了求整个验证集的平均损失，
        # 这里用“batch 平均损失 × batch 样本数”累加。
        loss_sum += loss.item() * image_a.size(0)
        count += image_a.size(0)

        # 更新整套验证集的统计量（TP/FP/FN/TN）。
        meter.update(logits, label, threshold)

    # 根据累计结果计算 precision / recall / f1 / iou / accuracy。
    metrics = meter.compute()

    # 再补充上平均验证损失。
    metrics['loss'] = loss_sum / max(count, 1)
    return metrics


def main():
    """
    训练主流程入口。

    建议你把这个函数当成“训练全过程时间线”来看：
    1) 解析命令行参数
    2) 构建并覆盖配置
    3) 固定随机种子
    4) 准备数据集和 DataLoader
    5) 创建模型和训练组件
    6) 进入 epoch 循环：训练 -> 验证 -> 保存权重
    7) 全部完成后打印最终结果
    """
    parser = argparse.ArgumentParser()

    # -------------------- 可选命令行参数 --------------------
    # 这些参数默认都写成 None，表示：
    # 如果命令行没传，就沿用 config.py 里的默认配置；
    # 如果命令行传了，就覆盖默认配置。
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--crop_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)

    # 是否使用 ImageNet 预训练权重。
    # --pretrained     => True
    # --no-pretrained  => False
    # 如果两个都不写，则保持 config.py 中的默认值。
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.set_defaults(pretrained=None)

    args = parser.parse_args()

    # -------------------- 读取配置并按命令行覆盖 --------------------
    # 先创建默认配置对象。
    cfg = TrainConfig()

    # 如果用户在命令行里传了参数，就覆盖掉默认值。
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.crop_size is not None:
        cfg.crop_size = args.crop_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.seed is not None:
        cfg.seed = args.seed
    if args.pretrained is not None:
        cfg.pretrained = args.pretrained

    # 固定随机种子，让实验尽可能可复现。
    set_seed(cfg.seed)

    # 创建权重保存目录。
    # parents=True 表示如果上级目录不存在也一并创建。
    # exist_ok=True 表示目录已存在时不报错。
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- 构建数据集 --------------------
    # train=True 会启用随机裁剪、翻转、旋转等训练增强。
    # train=False 则尽量保持原始样本，用于客观评估。
    train_set = LEVIRCDDataset(cfg.data_root, split='train', crop_size=cfg.crop_size, train=True)
    val_set = LEVIRCDDataset(cfg.data_root, split='val', crop_size=cfg.crop_size, train=False)

    # -------------------- 构建 DataLoader --------------------
    # 训练集：
    # - shuffle=True：每轮打乱样本顺序，有助于训练稳定。
    # 验证集：
    # - shuffle=False：验证不需要打乱，保证评估顺序稳定。
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # -------------------- 创建模型与训练组件 --------------------
    # 根据配置自动选择 CPU 或 GPU。
    device = torch.device(cfg.device)

    # 创建 DFEFT-CNN 模型。
    # pretrained=True 表示编码器 ResNet34 会加载 ImageNet 预训练权重。
    model = DFEFTCNN(pretrained=cfg.pretrained).to(device)

    # BCEWithLogitsLoss = sigmoid + BCE 的稳定实现。
    # 对于二分类变化检测非常常用。
    criterion = nn.BCEWithLogitsLoss()

    # 优化器负责根据梯度更新参数。
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    # MultiStepLR：在指定 epoch 把学习率乘以 gamma。
    # 例如默认设置中：50 / 100 / 150 轮后学习率衰减到原来的 0.1 倍。
    scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

    # best_val_loss 用来记录“当前见过的最优验证损失”。
    best_val_loss = float('inf')

    # 两个常用检查点路径：
    # - best_model.pt：验证集表现最好时保存
    # - last_model.pt：每轮都覆盖保存一次最新结果
    best_path = save_dir / 'best_model.pt'
    last_path = save_dir / 'last_model.pt'

    # ==================== 进入 epoch 训练循环 ====================
    for epoch in range(1, cfg.epochs + 1):
        # -------------------- train 阶段 --------------------
        # 切换为训练模式。
        model.train()

        # 用于统计这一整轮训练集的平均损失。
        epoch_loss = 0.0
        sample_count = 0

        # 进度条标题示例：train 3/200
        progress = tqdm(train_loader, desc=f'train {epoch}/{cfg.epochs}')
        for batch in progress:
            image_a = batch['image_a'].to(device)
            image_b = batch['image_b'].to(device)
            label = batch['label'].to(device)

            # 清空上一轮残留梯度。
            # set_to_none=True 通常更省内存，也更高效。
            optimizer.zero_grad(set_to_none=True)

            # 前向传播：得到预测 logits。
            logits = model(image_a, image_b)

            # 计算当前 batch 的损失。
            loss = criterion(logits, label)

            # 反向传播：根据损失自动计算每个参数的梯度。
            loss.backward()

            # 根据梯度更新模型参数。
            optimizer.step()

            # 统计本轮平均训练损失。
            epoch_loss += loss.item() * image_a.size(0)
            sample_count += image_a.size(0)

            # 在进度条上实时显示当前 batch loss 和当前学习率。
            progress.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        # 一个 epoch 结束后再更新学习率。
        scheduler.step()

        # 当前 epoch 的平均训练损失。
        train_loss = epoch_loss / max(sample_count, 1)

        # -------------------- eval 阶段 --------------------
        val_metrics = evaluate(model, val_loader, device, criterion, cfg.threshold)

        # 打印这一轮的核心结果，方便观察训练趋势。
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_metrics['loss']:.6f} "
            f"precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f} "
            f"f1={val_metrics['f1']:.4f} iou={val_metrics['iou']:.4f}"
        )

        # -------------------- 保存 last checkpoint --------------------
        # state 是一个完整训练快照，不只是模型参数，
        # 还包含优化器、学习率调度器、当前 epoch 和配置等信息。
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'cfg': cfg.__dict__,
            'val_metrics': val_metrics,
        }
        torch.save(state, last_path)

        # -------------------- 保存 best checkpoint --------------------
        # 当前验证损失更低，说明模型目前更优，就更新 best。
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(state, best_path)
            print(f'saved best checkpoint to {best_path}')

    # 所有训练轮次结束后的提示。
    print(f'training done, best checkpoint: {best_path}')


# Python 文件最常见的入口写法。
# 只有当你直接执行 `python train.py` 时，main() 才会运行。
if __name__ == '__main__':
    main()
