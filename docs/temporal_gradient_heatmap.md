# LS 与非 LS 的时间梯度热力图

`analysis/analyze_temporal_gradient.py` 从两个最佳 checkpoint 各执行一次相同测试批次的
前向和反向传播，比较指定神经元层输入端的时间梯度。它不会训练模型，也不会修改
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
- `--device`：例如 `cuda:0` 或 `cpu`；默认自动选择。
- `--fig-width`、`--fig-height`、`--dpi`：默认 `16`、`8`、`300`，可直接调整图片
  画布和清晰度。

## 输出

- `temporal_gradient_comparison.png`：LS 与非 LS 并排的高分辨率时间梯度热力图。
- `temporal_gradient_comparison.svg`：可无限缩放的矢量版本。
- `temporal_gradients.npz`：两幅图的原始梯度矩阵、真实神经元索引和诊断元数据。

横轴是时间步，纵轴是在目标层展平后均匀采样的神经元，颜色为训练损失对该层
输入的有符号梯度。两个 checkpoint 使用同一测试批次、样本、层、神经元索引、损失
定义和共享色标，以避免因为单独缩放而夸大某一方法。由于旧日志没有保存逐神经元、
逐时间步梯度，仅靠日志无法生成该图；本工具会读取 checkpoint 并补做一次反向传播，
但不需要重新训练。
