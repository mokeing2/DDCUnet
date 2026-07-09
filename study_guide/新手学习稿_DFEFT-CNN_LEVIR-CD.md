# DFEFT-CNN 复现项目新手学习稿（逐步上手版）

> 适用对象：第一次接触遥感变化检测、第一次完整读 PyTorch 工程的同学。  
> 目标：你不仅能跑通项目，还能解释“为什么这样设计”。

---

## 0. 先明确：你要学会什么？

这个项目对应的是 **DFEFT-CNN 在 LEVIR-CD 数据集上的建筑物变化检测复现**。

你最终需要能独立回答 4 个问题：

1. 数据是怎么读进来的？（A/B 两时相 + label）
2. 模型结构是怎样分模块工作的？（Encoder + FE/FF/DC + Head）
3. 训练是怎么优化的？（loss、optimizer、scheduler）
4. 指标是怎么算出来的？（Precision/Recall/F1/IoU）

---

## 1. 推荐学习顺序（非常重要）

请按这个顺序读，不要跳：

1. `train.py`（先看主流程）
2. `datasets/levir_cd.py`（理解输入数据）
3. `utils/metrics.py`（理解输出指标）
4. `models/dfeft_cnn.py`（看整体骨架）
5. `models/modules.py`（看 FE/FF/DC 细节）
6. `models/cbam.py` + `models/aspp.py`（看注意力与多尺度）
7. `test.py`（看评测流程）
8. `config.py`（回头统一看配置）

---

## 2. train.py 学习稿（主线）

`train.py` 是总控制器，你应该先把它看懂。

### 2.1 入口做了什么

- 解析命令行参数（你可以动态改 batch size / epochs 等）
- 读取 `TrainConfig`
- 用命令行覆盖默认配置
- `set_seed()` 固定随机性

### 2.2 数据部分

- 构建 `train_set`、`val_set`
- 构建 `DataLoader`
  - train: `shuffle=True`
  - val: `shuffle=False`

### 2.3 模型与优化

- 模型：`DFEFTCNN`
- 损失：`BCEWithLogitsLoss`
  - 适合二分类像素分割（变化/不变化）
- 优化器：`Adam`
- 学习率调度：`MultiStepLR(milestones=[50,100,150], gamma=0.1)`

### 2.4 每个 epoch 的逻辑

1. `model.train()`，循环训练集：
   - 前向：`logits = model(image_a, image_b)`
   - 损失：`loss = criterion(logits, label)`
   - 反向：`loss.backward()`
   - 更新：`optimizer.step()`
2. `scheduler.step()`
3. 调用 `evaluate()` 在 val 上算 loss + 指标
4. 保存 `last_model.pt`
5. 如果 val loss 更好，保存 `best_model.pt`

---

## 3. datasets/levir_cd.py 学习稿（数据）

这个文件决定“模型吃的是什么”。

### 3.1 数据结构

默认按 LEVIR-CD 标准目录：

- `train/A`, `train/B`, `train/label`
- `val/A`, `val/B`, `val/label`
- `test/A`, `test/B`, `test/label`

同名文件一一配对，例如 `A/0001.png` 对应 `B/0001.png` 与 `label/0001.png`。

### 3.2 训练增强（必须同步）

训练时会对 A/B/label **同步**做：

- 随机裁剪到 `256x256`
- 水平翻转
- 垂直翻转
- 90/180/270 旋转

> 为什么必须同步？
> 因为 label 是像素级监督，只要 A 和 label 错位 1 个像素，训练就会被破坏。

### 3.3 张量化

- A/B 转 tensor 后做 ImageNet 标准化
- label 做二值化 `(label > 127)`，并变成 `[1, H, W]`

---

## 4. utils/metrics.py 学习稿（指标）

### 4.1 从 logits 到二值图

1. `sigmoid(logits)` 得概率
2. `>= threshold` 得预测掩码

### 4.2 混淆矩阵四元素

- TP：预测变化且真实变化
- FP：预测变化但真实不变
- FN：预测不变但真实变化
- TN：预测不变且真实不变

### 4.3 指标公式

- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2PR / (P + R)
- IoU = TP / (TP + FP + FN)

项目里 `MetricAccumulator` 是“整数据集累计后再算指标”，这是标准做法。

---

## 5. dfeft_cnn.py 学习稿（主网络骨架）

你可以把它理解为“先分别提特征，再比较融合”。

### 5.1 三段式结构

1. **共享编码器（ResNet34）**：A 和 B 走同一套权重
2. **单分支解码器**：每个时相独立做 DC + FE 逐级解码
3. **双时相融合与输出**：同尺度用 FF 融合，再逐层上采样输出变化图

### 5.2 为什么共享编码器？

为了让 A/B 在同一个特征空间中可比较，减少“编码器偏差”。

---

## 6. modules.py 学习稿（论文核心）

这是整篇论文最重要的工程对应。

### 6.1 FE-Module（特征提取）

作用：融合“高层语义 + 低层细节”。

流程：

- 高层特征上采样到低层大小
- 两侧通道对齐
- WeightBlock 生成融合权重
- CBAM 强化
- 卷积与残差细化

### 6.2 FF-Module（双时相融合）

作用：融合同尺度 A/B 特征，突出变化区域，压制伪变化。

流程：

- A、B 各自 CBAM
- 通道拼接
- Conv + Residual 融合

### 6.3 DC-Module（空洞卷积）

作用：扩大感受野，增强上下文。

流程：

- ASPP 多膨胀率并行
- 两层 Conv
- CBAM

---

## 7. cbam.py + aspp.py 学习稿（基础模块）

### 7.1 CBAM

- Channel Attention：学“哪些通道重要”
- Spatial Attention：学“哪些空间位置重要”

顺序是通道后空间。

### 7.2 ASPP

并行空洞卷积（膨胀率 1/12/24/36）提多尺度特征，再拼接压缩。

---

## 8. test.py 学习稿（评测）

`test.py` 做的事很直接：

1. 加载 checkpoint
2. 构建指定 split 的数据集
3. 前向推理
4. 用 `MetricAccumulator` 汇总指标

你可以用它做阈值扫描（比如 0.45、0.50、0.55）观察 P/R 的变化趋势。

---

## 9. 新手实操练习（建议你按顺序做）

### 练习 1：最小训练闭环

```bash
python train.py --epochs 1 --batch_size 2 --num_workers 2 --save_dir ./checkpoints_quick
```

目标：确认你能完整跑通 train + val + 保存模型。

### 练习 2：快速测试

```bash
python test.py --ckpt ./checkpoints_quick/best_model.pt --split val --threshold 0.5
```

目标：确认你知道如何加载模型评测。

### 练习 3：阈值敏感性

分别用 `0.4 / 0.5 / 0.6` 评测并记录 Precision、Recall。

你会观察到：
- 阈值升高：Precision 往往上升，Recall 往往下降。

### 练习 4：消融感知（只做理解，不建议长期保留）

临时去掉 `FFModule` 的 CBAM，比较指标变化。

目标：理解“为什么论文要加注意力”。

---

## 10. 常见新手问题速查

### Q1: 为什么输出是 logits，不是概率？
因为训练用 `BCEWithLogitsLoss`，内部已包含 sigmoid，数值更稳定。

### Q2: 为什么 label 不是 0/255 而是 0/1？
损失函数和指标都按二值概率处理，0/1 更标准。

### Q3: 为什么验证集不用随机增强？
为了客观可比，验证/测试应尽量保持原始分布。

### Q4: 为什么每轮都存 last，但 best 只在变好时存？
- `last` 方便断点续训
- `best` 用于最终报告与测试

---

## 11. 你下一步该做什么

如果你已经按这份学习稿看完第一轮，建议马上做两件事：

1. 把 `train.py` 每个步骤写成你自己的“伪代码摘要”（10 行以内）
2. 画一张 DFEFT-CNN 结构草图（Encoder -> Decoder -> FF -> Head）

当你能画出来并讲给别人听，你就已经不是“只会跑代码”的阶段了。

---

## 12. 给你的鼓励

你现在走的是最难但最扎实的路：**自己学会复现，而不是只拿结果**。  
只要按这个学习稿走一遍，你会建立非常稳定的工程理解能力。继续保持！
