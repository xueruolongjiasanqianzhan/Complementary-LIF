# LS 与非 LS 的时间梯度热力图

`analysis/analyze_temporal_gradient.py` 从两个最佳 checkpoint 各执行一次相同测试批次的
前向和反向传播，默认比较指定神经元层输入的时间梯度。它不会训练模型，也不会修改
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

若两个目录记录的 seed、优化器、学习率、scheduler、weight decay、epoch、batch size 或
dropout 不一致，工具会输出显式警告。它仍允许生成探索性图片，但这些差异意味着结果不能
只归因于 LS；论文用比较应换成训练协议对齐的 checkpoint。

## 常用参数

- `--checkpoint-name`：checkpoint 文件名，默认 `checkpoint_max.pth`。
- `--batch-index` / `--sample-index`：测试批次及批次内样本，默认均为 0。
- `--batch-size`：仅影响诊断反向传播，默认 16。
- `--max-neurons`：沿展平后的通道和空间维均匀采样的最大神经元数，默认 512。
- `--cross-layer-count`：跨层热力图自动均匀抽取的神经元层数，默认 5。指定的
  `--layer` 无论是否被抽中都会包含在图中。每个额外层需要为两个 checkpoint 各补做
  一次诊断前向/反向，因此命令不变，但运行时间会增加；显存峰值仍接近单层分析。
- `--horizon-threshold`：有效梯度时间跨度使用的保持率阈值，默认 `1e-2`。
- `--gradient-percentile`：两种方法共同颜色范围的绝对值百分位，默认 99；两幅图
  始终使用同一个从白到深蓝的顺序色标，不能分别归一化。
- `--gradient-target`：默认 `final`，只从最后时间步的分类损失反向传播，使较早时间步
  的梯度只能通过神经元状态沿时间反传，这才对应 BPTT 时间梯度传播。`all` 可复现把
  每个时间步分类损失相加的旧行为。
- `--gradient-source`：默认 `input`，比较神经元层充电输入的梯度。它是 LIF/LSLIF
  共同、语义一致的计算节点，而且 LSLIF 的输入同时进入主膜和不重置 history 状态，
  因而能包含 LS 新增的平滑时间路径。`state` 是局部阈值判断节点：LIF 取 reset 前膜
  电位，LSLIF 取主膜与 history 融合后的 `total_mem`。后者不是传给下一时间步的完整
  recurrent state，故 `state` 适合观察阈值附近的局部代理梯度，但不能单独用于证明
  LS history 路径增强了跨时间梯度。LSLIF 只在 `state` 诊断时临时保留中间张量。
- `--aggregation`：默认 `batch-mean-abs`，对固定测试 batch 中各样本的绝对梯度取均值，
  减少单个样本脉冲稀疏造成的大片空白；`sample-signed` 可查看指定样本的有符号梯度。
- `--normalization`：默认 `final-step`，每个神经元用自身最后时间步绝对梯度归一化，
  因而最后一步为 1；热力图显示该占比的 `log10`，相邻颜色间隔代表相同数量级差异，
  所以 `0.1→0.01` 与 `0.01→0.001` 的视觉距离相同。`per-neuron` 保留旧版的跨时间
  最大值归一化，仅用于复现旧图；`none` 保留绝对梯度。
- `--normalized-color-gamma`、`--difference-linthresh` 和 `--color-scale`：仅影响
  `per-neuron`/`none` 兼容模式。前两个面板始终画梯度绝对强度，并用幂次颜色映射压缩
  动态范围，使较弱时间步仍然可见；`sample-signed` 的原始符号仍完整保存在 NPZ 中。
  默认 `final-step` 已直接使用 log10 数量级色标。
- `--device`：例如 `cuda:0` 或 `cpu`；默认自动选择。
- `--fig-width`、`--fig-height`、`--dpi`：默认 `21`、`8`、`300`，可直接调整图片
  画布和清晰度。

## 输出

- `temporal_gradient_comparison.png`：非 LS、LS 和二者差值并排的高分辨率时间梯度热力图。
- `temporal_gradient_comparison.svg`：可无限缩放的矢量版本。
- `temporal_gradient_profile.{png,svg}`：指定层每个时间步的平均绝对梯度，使用对数纵轴。
- `temporal_gradient_retention.{png,svg}`：指定层相对最后时间步的梯度保持率。
- `temporal_gradient_summary.{png,svg}`：指定层的 log10 梯度衰减斜率和有效梯度时间跨度。
- `cross_layer_gradient_ratio.{png,svg}`：多个均匀采样层的
  `log10(LS mean-|gradient| / Non-LS mean-|gradient|)` 时间热力图。
- `temporal_gradients.npz`：同时保存 `*_gradient_raw` 原始/批平均绝对梯度和
  `*_gradient_display` 非负绘图矩阵，以及真实神经元索引和诊断元数据。即使选择
  `sample-signed`，`*_gradient_raw` 仍保留符号，而 `*_gradient_display` 保存主图所用绝对值。
  `retention_log10_ratio` 保存第三个面板使用的 LS/Non-LS 保持率 log10 比值。

原推荐命令无需增加参数，只生成这一组 PNG/SVG 对比图。每个 checkpoint 各执行一次
诊断前向/反向，不会训练或修改 checkpoint。若输出目录含旧版生成的 retention、summary、
profile 或 cross-layer 图片，程序会删除这些旧文件，避免与当前结果混淆。

## 如何判断 LS 的梯度传播更好

- 前两个面板中，LS 在早期时间步保持的数量级更高，说明它相对最终时间步衰减更慢。
- 第三个面板中，正值（红色）表示 LS 保持率更高，`+1` 表示约高 10 倍。若多个神经元的
  早期时间步持续为正，比只在单神经元或单时间点出现红色更能支持 LS 的优势。

这些图必须使用训练协议对齐的 checkpoint，并优先保持默认的 `--gradient-target final`
和 `--gradient-source input`。单个 checkpoint 的图属于诊断证据；正式结论应汇总多个固定
batch 和 seed，并报告不确定性。

### 为什么改用最后时间步归一化与数量级颜色

不同训练模型的 loss 尺度、置信度和局部 Jacobian 都会改变绝对梯度，因此主图不再用绝对
梯度判断 LS。默认对每个神经元计算 `|g_t|/|g_T|`，直接回答早期梯度相对最终监督保留了
多少。图中实际显示 `log10` 占比：`1、0.1、0.01、0.001` 分别对应 `0、-1、-2、-3`，
每下降一个数量级使用相同颜色跨度。第三幅差值图显示两种保持率的 log10 比，即正值表示
LS 相对自身末端保留得更多。

旧版默认的 `state` 还比较了两个不同语义的局部节点：LIF 的 reset 前膜会进入后续状态，
而 LSLIF 的融合膜主要用于当前阈值判断，history 的平滑递归通过独立的 `n_t` 延续。于是在
融合膜上看到较小局部梯度，并不能推出 LS 的输入/history 路径更弱。为避免该混淆，当前
默认改为共同的层输入；如需复现旧图，可显式传 `--gradient-source state`。

横轴是时间步，纵轴是在目标层展平后均匀采样的神经元。默认颜色是固定 batch 的
梯度相对各神经元最后时间步梯度的 `log10` 占比。默认展示最后时间步损失向所有
时间步反传得到的梯度。两个
checkpoint 使用同一测试批次、样本、层、神经元索引、损失
定义和归一化规则。热力图用于观察相对最终时间步的梯度保持，不能证明绝对梯度更大；
绝对值仍保存在 NPZ 的 `*_gradient_raw` 中。由于旧日志没有保存逐神经元、
逐时间步梯度，仅靠日志无法生成该图；本工具会读取 checkpoint 并补做一次反向传播，
但不需要重新训练。

第三个面板显示 `log10(R_LS/R_Non-LS)`：红色正值表示该神经元/时间步的 LS 梯度
保持率更高，蓝色负值表示 Non-LS 保持率更高，白色零值表示二者接近。色标以 0 为中心
且正负范围对称，不会用不对称色标夸大任一方法。
