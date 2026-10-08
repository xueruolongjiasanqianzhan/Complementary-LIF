# MSF / LSMSF

本次集成沿用仓库的单时间步 stateful 神经元接口和现有 ResNet/VGG、训练时间循环。
参考 [官方 CIFAR10-DVS/MHSANet.py](https://github.com/fanliangwei/Multisynaptic-spiking-neurons/blob/main/CIFAR10-DVS/MHSANet.py)
中的 `ActFun_rectangular` 和 `mem_update_MSF`，复用其固定多阈值、矩形替代梯度和 hard reset。
官方该文件的阈值求和写死为四项；本实现推广到任意正整数 D，不增加卷积或全连接权重。
这不是官方整套网络、训练配方或论文精度的复现。

## 动力学

默认 `msf_D=4`、`msf_threshold=1.0`、`msf_decay=0.25`、`msf_alpha=0.5`。
固定阈值为 `msf_threshold + arange(msf_D)`，阈值不是可学习参数。

MSF 每步：

```text
main_mem = msf_decay * previous_main_state + input
spike = sum(main_mem >= thresholds[d])
main_state = 0 if spike > 0 else main_mem
```

LSMSF 每步：

```text
main_mem = msf_decay * previous_main_state + input
history_mem = msf_decay * previous_history_state + input
step += 1
history_term = history_weight * history_mem / (step + history_eps)^history_power
total_mem = main_mem + history_term
spike = sum(total_mem >= thresholds[d])
main_state = 0 if spike > 0 else main_mem
history_state = history_mem
```

所有阈值共享历史项。主膜使用不可微的 `spike > 0` 布尔掩码清零；历史膜不因发放清零。
未发放时保存 `main_mem`，而不是融合后的 `total_mem`。
输出为浮点类型的整数脉冲数 0～D，不二值化、不除以 D。
每个阈值的替代梯度为 `1 / (2 * msf_alpha)`，仅在
`abs(membrane - threshold) < msf_alpha` 时有效，边界处为零；多个窗口重叠时求和。

默认输入 `[0.8, 1.6, 0.4, 0]` 对应 MSF `[0, 1, 0, 0]`、LSMSF `[1, 2, 0, 0]`。
固定 `history_weight=0` 时，LSMSF 与同配置 MSF 的输出和输入梯度一致。

## LS 参数与状态

LSMSF 继承标准 `LSLIFNeuron` 的参数管理：

- 默认固定 `history_weight=1`、`history_power=1`、`history_eps=1e-6`。
- `all` 从第一步使用包含当前输入的历史膜。
- `post_spike` 在该位置此前已发放后启用历史贡献，首次发放的当步不因此启用；历史膜始终积累。
- `half` 使用现有 backbone 层编号：前 `total_layers // 2` 层为 `post_spike`，其余为 `all`；无层编号时按现有行为退化为 `all`。
- 可学习权重无上下界时通过 softplus 保持正值，有上下界时使用 sigmoid 映射。
  `history_weight_lo/hi` 不裁剪固定权重，两者必须成对提供。
- `history_weight_per_step` 配合 `history_learn_weight` 为每步建立权重；超出
  `history_max_steps` 后复用最后一项。训练、推理入口沿用现有做法，将该长度设为 T。
- 可学习 power 通过 sigmoid 映射到 `(0, 2)`。

膜计算和历史状态使用 FP32，输出转换为输入浮点类型。支持标量、全连接和卷积输入形状。
形状或设备变化会重新初始化序列；同形状的下一个 batch 仍必须调用
`functional.reset_net(net)`，现有训练循环已执行此操作。
`reset()` 清除主膜、历史膜、发放标记、步数和运行时诊断状态，保留阈值和学习参数。

通用 `tau`、`v_threshold`、`decay_input`、`v_reset`、`detach_reset` 和
`surrogate_function` 不改变 MSF 专属动力学、矩形梯度或布尔 hard reset。
纯 MSF/LSMSF 暂不组合 ASN、成功调制、突触释放、RPLIF 的 LIF head 或
`multiple_step` 接口；启用会明确报错。普通外层训练时间循环正常使用。

## 训练、推理与重建

训练和推理均支持 `-neuron_model MSF`、`-neuron_model LSMSF`，以及：

```text
-msf_D 4 -msf_threshold 1.0 -msf_decay 0.25 -msf_alpha 0.5
```

五个数据集的成对命令见 `neuron_training_commands_by_dataset.txt`。
MSF/LSMSF 沿用各数据集已有网络、时间步和训练配方。
LSMSF 默认命令显式列出 LS 的固定权重、幂、epsilon 和 mode；不自动开展训练或参数搜索。

`args.txt` 保存完整 CLI，JSON summary 保存有效 `neuron_config`，新神经元的
checkpoint 同时保存 `neuron_config` 和 `experiment_args`。
运行名称包含 D、首阈值、decay、alpha；LSMSF 还包含权重、幂和 mode。
配置摘要区分其余 LS 参数，避免名称过长；完整参数以日志和 checkpoint 为准。

可使用 checkpoint 的 `experiment_args` 恢复网络、数据集和 T，使用
`neuron_config['kwargs']` 构造神经元，并严格加载 `net` state dict。
state dict 不包含运行中的膜状态。`-resume` 会拒绝与保存的 MSF/LSMSF
神经元配置不同的 CLI 设置；不会自动替用户选择参数。
逐步可学习权重的重建需要保持训练时的长度，不能直接以不同 T 改变参数形状。

## 正确性验证与统计

在仓库根目录执行：

```bash
python -m unittest analysis.test_msf_neuron -v
python -m unittest discover -s analysis -p 'test_*.py' -v
```

测试覆盖多阈值输出和等号边界、严格矩形窗口及重叠梯度、泄漏及各级输出的
hard reset、D=1、LS 状态及零权重等价、LS 参数学习及 mode、输入形状和
CPU 混合精度、ResNet/VGG 多步前向/反向/reset、训练推理 CLI 和配置重建。
CUDA 测试在无 GPU 环境明确跳过，CPU 验证不代表完整 GPU 训练已验证。
现有 `test_rplif_neuron.py` 的 tau_eps 精度断言失败与本次接入无关，不修改或禁用它。
以后修改相关代码或依赖时重跑受影响检查；仅更换数据集、种子或实验参数通常不必重跑整套检查。

现有统计器会采集新神经元输出，但 `spike_rate` 字段对 MSF/LSMSF 表示
**平均脉冲数 `mean(spike)`**，可能超过 1。它不等于
**发生发放的比例 `mean(spike > 0)`**。本次不增加统计字段，不套用二值神经元能耗解释。
测试通过说明实现符合定义，不说明 LSMSF 性能优于 MSF。
独立 EEG/患者入口和专用分析脚本不在本次扩展范围内。
