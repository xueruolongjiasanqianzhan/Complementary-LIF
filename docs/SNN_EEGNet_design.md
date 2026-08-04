# EEGNet 的 SNN 化设计

## 1. 先区分两种“时间”

当前 EEG segment 输入为 `[B, 1, C, L]`，其中 `C=61` 是电极数、`L=750`
是采样点数。现有全连接基线把 `L` 直接当作 SNN 时间轴，逐采样点输入
`[B, C]`，因此一次前向传播有 750 个状态更新步。

EEGNet 中沿末轴滑动的时间卷积则是**特征提取算子**：它在一个局部采样窗口内
学习频带/波形模板。SNN 的时间步是**神经元状态更新的顺序**。两者可以并存，
不应因为 EEGNet 有时间卷积就把整段信号静态复制 `T` 次。

推荐采用流式（streaming）方案：把原始采样轴切成连续、不打乱的 chunk；每个
chunk 是一个 SNN 步，而 chunk 内仍由 EEGNet 的时间卷积提取局部特征。

```text
[B, 1, C, L]
      │ 按末轴切成 T 个连续 chunk（可带左侧上下文）
[T, B, 1, C, W]
      │ 每个 t 共享同一套 EEGNet 卷积权重
temporal conv → BN → depthwise spatial conv → BN → LIF/LSLIF
      → pool → separable temporal conv → BN → LIF/LSLIF → pool
      → classifier current → 输出神经元/时间聚合
```

## 2. 推荐的 Spiking-EEGNet 结构

以 EEGNet-8,2 为例，保持标准 EEGNet 的归纳偏置，只替换非线性并显式展开状态：

1. `Conv2d(1, F1, (1, K1), bias=False)`：时间卷积，权重跨 SNN 步共享；
2. `BatchNorm2d(F1)`；
3. `Conv2d(F1, F1*D, (C, 1), groups=F1, bias=False)`：跨电极的 depthwise
   spatial convolution；
4. `BatchNorm2d(F1*D) → LIF/LSLIF → AvgPool2d((1, P1)) → Dropout`；
5. `Conv2d(F1*D, F1*D, (1, K2), groups=F1*D, bias=False)` 加
   `Conv2d(F1*D, F2, 1, bias=False)`，即 separable temporal convolution；
6. `BatchNorm2d(F2) → LIF/LSLIF → AvgPool2d((1, P2)) → Dropout`；
7. 对 chunk 内剩余宽度做均值，得到 `[B, F2]`，再经 `Linear(F2, classes)`
   产生每一步的分类电流。

第一版分类头保持非脉冲：先对各步脉冲特征求均值，再送入普通 `nn.Linear`
分类器：

\[
\bar h = \frac{1}{T}\sum_{t=1}^{T} h_t,\qquad z=W\bar h+b.
\]

这与当前复杂 EEG 任务的“时间特征平均后分类”读出一致，也不会强迫仅有两个输出
神经元承担全部梯度。完成稳定性验证后，再增加“输出 LIF + spike count/readout
membrane”作为消融实验。

## 3. 时间切片与卷积边界

### 推荐起点

- `chunk_size=25`（若采样率 250 Hz，则约为 100 ms）；`750 / 25 = 30` 个 SNN 步；
- chunk 顺序严格保持，膜电位在同一样本的 30 步之间保留；
- batch 开始前重置全部有状态神经元，绝不能在两个样本/batch 间串状态；
- 最后一个不足 `chunk_size` 的 chunk 右侧补零，并用 mask 避免其在时间均值中
  获得与完整 chunk 相同的权重。

时间卷积会遇到 chunk 边界。不要独立地对每个 chunk 做双侧 `same` padding，
否则边界处会人为插入很多零。可采用以下任一种做法：

1. **因果缓存（推荐）**：在第 `t` 步输入前拼接前一 chunk 的最后 `K1-1`
   个采样点，仅保留当前 chunk 对应的卷积输出；第一个 chunk 左侧补零；
2. **重叠 chunk**：窗口宽 `W`、步长 `S<W`，但只保留每个窗口最新的 `S`
   个位置，并在读出时避免重复计数；
3. **离线卷积前端**：先对完整 `[B,1,C,L]` 做第一层时间卷积，再沿输出时间轴
   切片送入后续脉冲层。它实现最简单，但第一层不属于严格的逐步事件驱动计算。

第一轮实验建议用方案 3 验证网络和 LIF/LSLIF，再用方案 1 做严格流式版本；二者
使用同一训练/测试划分，并报告差异。

## 4. 输入编码

EEG 是有正负的连续信号，不能直接用只表达正值的单路脉冲编码。按优先级建议：

1. **直接电流注入（首选基线）**：标准化后的实值 chunk 进入第一层卷积；第一层
   LIF/LSLIF 才产生 spike。它不声称输入端已经事件化，但训练最稳定、信息损失最少；
2. **正负双通道**：`x_pos=relu(x)`、`x_neg=relu(-x)`，前端输入通道变为 2；
3. **delta modulation**：当相邻采样差超过正/负阈值时生成 UP/DOWN 事件，更接近
   事件驱动硬件，但阈值是额外超参数，必须单独做信息损失和稀疏率消融。

不要把同一个完整 EEG segment 静态复制 30 次。那样的“时间步”只是在重复相同
输入，既没有利用 EEG 的物理时间，也会改变积分电荷的尺度。

## 5. 前向传播伪代码

```python
def forward(self, x):                 # x: [B, 1, C, L]
    reset_state_at_batch_boundary()
    features = self.temporal_frontend(x)
    chunks, valid = split_time_axis(features, self.chunk_size)

    logits = []
    for t, chunk_t in enumerate(chunks):
        y = self.spatial_depthwise(chunk_t)
        s1 = self.neuron1(self.bn1(y)) # state persists across t
        y = self.pool1(s1)
        y = self.separable_temporal(y)
        s2 = self.neuron2(self.bn2(y)) # state persists across t
        logits.append(self.classifier(global_avg_pool(s2)))

    return masked_time_mean(torch.stack(logits), valid)
```

重置应由训练/验证循环统一执行，而不是在 `forward()` 的每个时间步中执行。若使用
截断 BPTT，可每 `k` 步 detach 膜电位，但不能 reset；30 步通常可先做完整 BPTT。

## 6. 训练与公平对比

- 按训练 fold 的每个电极统计均值/标准差，并原样应用到测试 fold，禁止用测试数据
  统计归一化参数；
- ANN EEGNet、SNN-LIF EEGNet、SNN-LSLIF EEGNet 使用完全相同的 fold、随机种子、
  卷积通道数、分类头和数据增强；
- 损失先用 logits 上的 cross entropy；若输出改成 spike count，可改用发放率 MSE
  或 spike-count cross entropy，并作为独立实验报告；
- 反向传播使用替代梯度，梯度裁剪可从 `1.0` 或 `5.0` 起试；
- 每个 batch 前重置状态，验证/推理同样重置；
- 同时报告 accuracy、recall、specificity、precision、F1、AUC，以及各脉冲层平均
  firing rate；只比较准确率不足以说明 SNN 是否有效。

建议的最小消融矩阵：

| 模型 | 时间步 | 编码 | 读出 |
|---|---:|---|---|
| ANN EEGNet | 1 | continuous | logits |
| SNN EEGNet-LIF | 30 | direct current | mean logits |
| SNN EEGNet-LSLIF | 30 | direct current | mean logits |
| SNN EEGNet-LIF | 15/30/60 | direct current | mean logits |
| SNN EEGNet-LSLIF | 30 | positive/negative | mean logits |
| SNN EEGNet-LSLIF | 30 | direct current | spike count |

## 7. 实现顺序与验收条件

1. 先复现 ANN EEGNet，确认数据形状、fold 和指标实现正确；
2. 使用“离线第一层时间卷积 + 30 chunks + LIF + mean logits”跑通；
3. 仅把 LIF 替换为 LSLIF，其他参数不变；
4. 增加 firing-rate、零发放层和膜电位统计，排查沉默/饱和；
5. 最后实现因果缓存，比较离线与严格流式前端。

基本验收应包括：随机输入前向/反向无 NaN；连续调用两个 batch 时，reset 后结果不依赖
前一个 batch；改变 `chunk_size` 时输出形状不变；被 padding 的时间步不参与读出；
LIF 与 LSLIF 模型除神经元状态外拥有一致的可训练卷积/分类参数规模。
