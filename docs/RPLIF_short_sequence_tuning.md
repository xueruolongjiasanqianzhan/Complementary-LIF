# RPLIF / LSRPLIF 短时间步调参说明

## 结论

这里讨论的不是 `-T` 与 `-T_max` 的区别，也不假设现有命令把两者混淆。需要验证的核心假设是：现有命令对不同序列长度统一使用 `history_weight=1.0`、`history_power=1.0`、`history_mode=all` 和 `rplif_alpha=1.5`，而 LS/history 分支的有效尺度与序列长度有关，因此同一组参数可能偏向长序列。

短时间步数据集上 LSRPLIF 不如 RPLIF，并不能直接说明 LS 分支无效；但在完成消融前，也不能断言一定是参数造成的。应在每个数据集自己的固定 `T` 下比较 RPLIF 与 LSRPLIF，并把“LS 强度”和“不应期强度”分开搜索。

## 为什么短序列对参数更敏感

LSRPLIF 的发放判据使用

```text
total_mem(t) = main_mem(t) + history_weight * history_mem(t) / t^history_power
```

默认 `history_power=1` 时，LS 项在前几个时间步最大：`t=1` 时没有任何时间衰减，`t=2` 时也仍有一半权重。短序列的输出主要由这些早期时间步构成，因此 `history_weight=1.0` 可能过强；长序列则有更多后期时间步利用累计历史。这是待实验验证的解释，而不是已经确定的原因。

与此同时，当前 RPLIF 更新在某个时间步发放后，将该位置下一步阈值乘以 `rplif_alpha`；若下一步不发放，阈值便回到初始值。因此 `alpha` 主要控制连续发放。在只有 4--6 步的序列中，一个受抑制时间步占整个决策窗口的比例很高。LS 分支提高早期膜电位、RPLIF 又抑制紧接着的连续发放，两者可能产生“提前发放一次、随后抑制关键帧”的组合效应。

## 推荐的最小消融矩阵

不要一开始同时搜索所有训练参数。每个设置至少运行 3 个随机种子，并同时记录验证准确率、全局发放率和分层发放率。

### 阶段 A：确认问题来自 LS 分支

固定模型、数据划分、增广、优化器、学习率、epoch 和 `T`，依次运行：

| 实验 | 神经元 | `history_weight` | `rplif_alpha` | 目的 |
|---|---|---:|---:|---|
| A0 | RPLIF | 不适用 | 1.5 | 公平基线 |
| A1 | LSRPLIF | 0.0 | 1.5 | 应接近 RPLIF，用于检查实现/训练噪声 |
| A2 | LSRPLIF | 0.25 | 1.5 | 弱 LS |
| A3 | LSRPLIF | 0.5 | 1.5 | 中等 LS |
| A4 | LSRPLIF | 1.0 | 1.5 | 当前默认 LS |

若 A1 与 A0 差距依然很大，应先检查随机种子、初始化、数据顺序和 checkpoint，而不是继续调 LS 权重。

### 阶段 B：降低短序列中的不应期强度

取阶段 A 最好的 `history_weight`，搜索：

```text
rplif_alpha ∈ {1.0, 1.1, 1.25, 1.5}
```

`alpha=1.0` 是必要的机制消融；它保留 LSRPLIF 的 LS 分支，但关闭阈值抬升。如果 `1.0/1.1` 明显优于 `1.5`，说明短序列主要受强不应期影响。反之，如果不同 `alpha` 的差距很小，就不应继续把主要搜索预算放在 RPLIF 参数上。

### 阶段 C：只在仍有必要时搜索 LS 形状

建议测试：

```text
history_power ∈ {0.5, 1.0, 1.5}
history_mode  ∈ {all, post_spike}
```

较大的 `history_power` 会更快削弱后续 LS 项；`post_spike` 会在神经元第一次发放前关闭 LS 项，可用于验证“LS 导致过早首发放”的假设。不要把 `history_weight_lo/hi` 当作当前固定权重的裁剪参数：只有加上 `-history_learn_weight` 时，边界才有意义。

## 可直接运行的短序列示例

下面以 DVS-CIFAR10、`T=4` 为例。比较时只改变行末列出的神经元参数：

```bash
# RPLIF 基线
python train.py -data_dir ./data_dir -dataset DVSCIFAR10 -T 4 \
  -drop_rate 0.3 -model spiking_vgg11_bn -lr 0.05 -mse_n_reg \
  -neuron_model RPLIF -rplif_alpha 1.5 -rplif_v_init_th 1.0 -refractory_step 1 \
  -seed 2022 -name dvs_t4_rplif_a15

# LSRPLIF：建议的短序列起点
python train.py -data_dir ./data_dir -dataset DVSCIFAR10 -T 4 \
  -drop_rate 0.3 -model spiking_vgg11_bn -lr 0.05 -mse_n_reg \
  -neuron_model LSRPLIF -history_weight 0.25 -history_power 1.0 \
  -history_mode all -rplif_alpha 1.25 -rplif_v_init_th 1.0 -refractory_step 1 \
  -seed 2022 -name dvs_t4_lsrplif_hw025_a125
```

## CIFAR-10/100 命令的实际修改

仓库中的 CIFAR-10 和 CIFAR-100 LSRPLIF 命令已从共同的长序列基线

```text
history_weight=1.0, rplif_alpha=1.5
```

改为短序列 `T=4` 的首选候选配置：

```text
history_weight=0.25, history_power=1.0, rplif_alpha=1.25
```

这样修改有两个目的：`history_weight` 从 1.0 降到 0.25，直接限制前四步中 LS 分支对主膜电位的扰动；`alpha` 从 1.5 降到 1.25，保留相对不应期机制，同时减轻一次发放对仅剩几个决策步的影响。`history_mode=all` 和 `history_power=1.0` 暂时不变，避免一次同时改变过多因素。命令显式写出 `-T 4`，用于固定该短序列配置的适用条件。

这是一组有机制依据的首选实验配置，并不是已经由结果证明的最优值。若它仍低于 RPLIF，下一轮只需要围绕它测试 `history_weight={0, 0.25, 0.5}` 与 `rplif_alpha={1.0, 1.25, 1.5}`；不要同时改动 `history_power`。

## 公平比较时的控制变量

- `T`、epoch 和学习率调度必须在 RPLIF/LSRPLIF 配对实验中完全相同，避免把训练预算差异归因于 LS。
- 静态图像数据在各时间步重复输入，而 DVS 数据的各时间步是不同事件帧；即使 `T` 相同，最优 LS 强度也不一定相同。
- 不要只报告单次最高准确率。至少比较 3 个种子的均值和标准差，并检查发放率，避免把随机波动或近乎静默/饱和发放误判为机制收益。
