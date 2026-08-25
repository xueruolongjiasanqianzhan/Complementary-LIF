# LS 与非 LS 的时间梯度热力图

`analysis/analyze_temporal_gradient.py` 从两个最佳 checkpoint 各执行一次相同测试批次的
前向和反向传播，比较指定神经元层预脉冲膜电位的时间梯度。它不会训练模型，也不会修改
checkpoint 参数。

## 安装

先使用项目训练环境（PyTorch、torchvision 和 SpikingJelly），再安装绘图依赖：

```bash
python -m pip install -r analysis/requirements.txt
```

## 推荐命令

```bash
python analysis/analyze_temporal_gradient.py \
  --ls-run /path/to/LS/experiment \
  --baseline-run /path/to/LIF/experiment \
  --data-dir /path/to/CIFAR10DVS \
  --layer layer3.6 \
  --output-dir gradient_analysis
```

两个实验目录必须包含 `args.txt`（或 `run_summary.json`）和
`checkpoint_max.pth`。当前版本针对本项目的 `DVSCIFAR10` 实验，并支持 LIF 与
LSLIF checkpoint。默认分析 VGG11 的中部神经元层 `layer3.6`；如果 checkpoint
使用其他网络，可通过 `--layer` 指定两个模型中名称相同的神经元层。若名称不存在，
程序会列出可选层。

## 常用参数

- `--checkpoint-name`：checkpoint 文件名，默认 `checkpoint_max.pth`。
- `--batch-index` / `--sample-index`：测试批次及批次内样本，默认均为 0。
- `--batch-size`：仅影响诊断反向传播，默认 16。
- `--max-neurons`：沿展平后的通道和空间维均匀采样的最大神经元数，默认 512。
- `--gradient-percentile`：两种方法共同颜色范围的绝对值百分位，默认 99；两幅图
  始终使用同一个对称色标，不能分别归一化。
- `--gradient-target`：默认 `final`，只从最后时间步的分类损失反向传播，使较早时间步
  的梯度只能通过神经元状态沿时间反传，这才对应 BPTT 时间梯度传播。`all` 可复现把
  每个时间步分类损失相加的旧行为。
- `--gradient-source`：默认 `state`，LIF 使用阈值判断前的膜电位，LSLIF 使用主膜电位
  与历史支路融合后的膜电位；`input` 可切回旧版的神经元层输入梯度。
- `--aggregation`：默认 `batch-mean-abs`，对固定测试 batch 中各样本的绝对梯度取均值，
  减少单个样本脉冲稀疏造成的大片空白；`sample-signed` 可查看指定样本的有符号梯度。
- `--normalization`：默认 `per-neuron`，每个神经元按其跨时间步最大绝对梯度归一化，
  用于直观展示梯度能传播到哪些时间步。`none` 保留绝对梯度尺度。
- `--color-scale`：默认 `symlog`，用两幅图共享的对称对数色标显示跨多个数量级的梯度，
  避免较大的第一个时间步把其他列全部压成白色；`linear` 可切回线性色标。该选项不会
  对每个时间步单独归一化，因此 LS 和非 LS 的绝对大小仍可公平比较。
- `--device`：例如 `cuda:0` 或 `cpu`；默认自动选择。
- `--fig-width`、`--fig-height`、`--dpi`：默认 `16`、`8`、`300`，可直接调整图片
  画布和清晰度。

## 输出

- `temporal_gradient_comparison.png`：LS 与非 LS 并排的高分辨率时间梯度热力图。
- `temporal_gradient_comparison.svg`：可无限缩放的矢量版本。
- `temporal_gradients.npz`：同时保存 `*_gradient_raw` 原始/批平均绝对梯度和
  `*_gradient_display` 绘图矩阵，以及真实神经元索引和诊断元数据。

横轴是时间步，纵轴是在目标层展平后均匀采样的神经元。默认颜色是固定 batch 的
膜电位绝对梯度均值经过逐神经元时间归一化后的数值。默认展示最后时间步损失向所有
时间步反传得到的梯度。两个
checkpoint 使用同一测试批次、样本、层、神经元索引、损失
定义和归一化规则。归一化热力图用于观察时间传播形态，不能证明绝对梯度更大；绝对
量级比较应使用 NPZ 中的 `*_gradient_raw`。由于旧日志没有保存逐神经元、
逐时间步梯度，仅靠日志无法生成该图；本工具会读取 checkpoint 并补做一次反向传播，
但不需要重新训练。
