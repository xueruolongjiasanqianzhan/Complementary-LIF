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
- `--layer all`：按照模型注册顺序拼接所有 LIF/LSLIF 层的神经元，生成类似论文中“所有
  隐含神经元”的总览图。该模式只支持语义一致的 `--gradient-source input`，会保留所有层
  的时间梯度并明显增加显存；显存不足时应继续逐层分析，而不是降低两个模型中任一方的采样。
- `--batch-index` / `--sample-index`：测试批次及批次内样本，默认均为 0。
- `--batch-size`：仅影响诊断反向传播，默认 16。
- `--max-neurons`：沿展平后的通道和空间维均匀采样的最大神经元数，默认 512。
- `--cross-layer-count`：跨层热力图自动均匀抽取的神经元层数，默认 5。指定的
  `--layer` 无论是否被抽中都会包含在图中。每个额外层需要为两个 checkpoint 各补做
  一次诊断前向/反向，因此命令不变，但运行时间会增加；显存峰值仍接近单层分析。
- `--horizon-threshold`：有效梯度时间跨度使用的保持率阈值，默认 `1e-2`。
- `--gradient-percentile`：两种方法共同颜色范围的绝对值百分位，默认 99；两幅图
  始终使用同一个从白到深蓝的顺序色标，不能分别归一化。
- `--gradient-vmax`：仅与 `--normalization none` 一起使用，固定绝对梯度主图的颜色上限；
  跨 checkpoint、batch 或 seed 比较绝对值时应先确定一个全局上限，再为所有运行传入同一值。
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
- `temporal_gradients.npz`：同时保存 `*_gradient_raw` 原始/批聚合梯度和
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

## 与 Rhythm-SNN 论文图的区别

常被拿来参照的 Rhythm-SNN 图使用 PS-MNIST 的一个小批量样本，分别对 FFSNN、ASRNN
及其 Rhythm 版本做一次前向传播和 BPTT，再把**所有隐含神经元**的归一化时间梯度画成
矩阵：横轴是时间，纵轴是神经元，左右两个面板是基础模型和加入节律掩码后的模型。它不是
本工具的 LIF/LSLIF、指定单层、DVS-CIFAR10 对比实验，不能把本工具的具体处理方式倒推给
论文原图。

该实验要检验的是长序列上的 credit assignment。PS-MNIST 把一张 MNIST 图像按固定像素
置换展开成序列；分类监督在序列末端产生。对较早的隐藏状态而言，梯度必须穿过更长的时间
链，所以基础模型面板常呈现“早期接近白色、后期颜色较深”。Rhythm 模型给不同神经元
分配错开的周期性开/关节律；关闭期减少无效的状态更新，使反向梯度沿较短或较少干扰的
路径传递。因此它的早期列中有更多非白色条带，作者据此主张更多梯度被有效分配到较早
时间步。横向条带还反映不同神经元的节律相位和周期不同，并不是把整层统一地缩短到同一
时间长度。

“归一化时间梯度”只说明绘图前做过尺度处理。仅凭截图和图注，不能确定它究竟是按神经元、
按面板还是全图归一化，也不能断言使用了 log 色标；色条仍有正负刻度，说明还不应擅自把
它解释成绝对梯度。严谨复现需要查论文正文或代码中的归一化公式、求导对象（隐藏状态、
膜电位或层输入）、batch 聚合方式、损失取最后一步还是所有步，以及各面板是否共用
`vmin/vmax`。如果每个面板单独归一化，那么只能比较同一面板内梯度在时间上的分布，不能
凭相同深浅比较两个模型的绝对梯度。

本仓库工具回答的是一个相关但不同的问题：默认计算指定层输入的
`R_t = |g_t| / max(|g_T|, eps)` 并显示 `log10(R_t)`。这里 `1、10^-1、10^-2、10^-3`
分别显示为 `0、-1、-2、-3`，所以相差多个数量级仍然清晰。减少时间步数 `T` 后分别出图
时，颜色范围还会重新由数据百分位数确定；图片都“很好看”并不代表绝对梯度相同。绝对量级
应查看 `*_gradient_raw`，或使用 `--normalization none --gradient-vmax VALUE` 为所有实验固定
相同颜色上限（下限固定为 0）。

## 生成类似论文图的最简命令

如果只是想先看看 LS 在这个实验里的表现，运行：

```bash
python analysis/analyze_temporal_gradient.py \
  --ls-run /path/to/LS/experiment \
  --baseline-run /path/to/LIF/experiment \
  --data-dir /path/to/CIFAR10DVS \
  --paper-style \
  --output-dir gradient_analysis/paper_style
```

`--paper-style` 自动使用所有 LIF/LSLIF 层、最终时间步损失和共同的层输入节点；它先在
batch 内对**有符号梯度**求均值，再用两个模型共同的最大绝对值归一化，并只画 LIF 与 LS
两个面板。横轴是时间，纵轴是按模型注册顺序拼接后均匀采样的隐含神经元；两幅图共用
以 0 为中心的对称色标。0 是白色，正负梯度离 0 越远颜色都越深。若 LS 面板有更多颜色延伸到早期时间步，
说明在这个 batch 中 LS 的相对梯度保持得更久。这是与 Rhythm-SNN 图相同的观察思路，适合
快速诊断，但单个 batch 的图只能作为现象展示。

这个初步实验**没有必要先比较绝对梯度，也不应人为把两个模型的初始梯度设成一样**。
分类损失产生的末端梯度本来就受两个已训练模型的输出与置信度影响；强行改成相同数值会把
实验改成“固定外部反传信号下比较 Jacobian”，不再是模型在真实任务损失下的自然梯度。
这里用同一个全局因子缩放 LIF 和 LS，不会分别把两张图调到同样深；因此既保留时间上的
自然衰减，也保证 0 对应白色和两个模型的颜色可比。

只有当论文结论要进一步写成“LS 的绝对梯度更大”时，才需要补充绝对梯度实验。此时不要
修改梯度，而应保留真实 loss，使用 `--normalization none --gradient-vmax VALUE`，并让所有
模型和运行共享同一个 `VALUE`。更严格的机制实验可以另外向两个模型的最终输出或最终状态
注入相同的单位向量，再比较 vector-Jacobian product；它应作为控制实验单独报告，不能替代
上面的真实损失实验。
