# Direct reset-bypass experiment for LS

## What the experiment should claim

LS should be described as a **plug-in, non-reset branch**. It does not make the
main membrane's reset subtraction smaller. Instead, it leaves an additional
state available after the main membrane resets, thereby reducing the *relative
loss of effective state available to the next firing decision*.

This distinction is important: an input-prefix masking experiment shows that a
model uses earlier inputs, but it does not isolate reset as the cause. It is
therefore useful supporting evidence, but it is not a direct demonstration of
the new reset-bypass claim.

## Causal protocol

1. Feed one identical pulse sequence to a resettable main membrane and an LS
   auxiliary membrane with matched leakage.
2. At fixed, pre-registered time steps at which the main membrane is above
   threshold, force a soft reset of exactly one threshold in the main membrane.
   Fixed interventions avoid the confound that LIF and LS can naturally spike
   at different times; the script rejects a scheduled reset that is not
   spike-valid.
3. Keep the input and all dynamics identical; the only intervention is whether
   the main membrane resets. The auxiliary membrane is never reset.
4. Plot the no-reset counterfactual, reset LIF state, preserved LS branch, and
   their sum. At every reset, also plot
   `post-reset effective state / pre-reset effective state`.

The primary quantity is the LS branch magnitude that remains at a reset. The
secondary quantity is the gain in the relative retention ratio over LIF. The
absolute main-membrane reset loss must also be reported as a sanity check; it
should remain exactly one threshold in this soft-reset simulation. This avoids
the stronger and incorrect claim that LS changes the reset operator itself.

## Quick start / 快速运行

### 先说明：toy 结果是不是“填数据”

`reset_bypass_toy.py` **没有人工填写输出数据，也没有使用随机生成的结果**；
CSV、JSON 和图片都由脚本中的标量 LIF/LS 差分方程逐时间步计算得到。但它是
为了做受控因果干预而手写的简化方程，**并没有实例化仓库中的
`VanillaLIFNeuron` 和 `LSLIFNeuron` 类**。因此它适合解释机制，不应被表述为
“生产实现之间的实测对比”。

如果需要确认真实代码实现，请优先运行 `reset_bypass_real_neurons.py`。它直接
实例化训练代码使用的 `modules.neuron.VanillaLIFNeuron` 和
`modules.neuron.LSLIFNeuron`，给两者输入完全相同的脉冲，并从真实 forward
过程读取放电、reset 后主膜电位、LS 历史支路和融合状态：

```bash
cd /workspace/Complementary-LIF
# 使用训练环境，其中必须已经安装 torch 和 spikingjelly
python analysis/reset_bypass_real_neurons.py
```

真实神经元实验输出到 `analysis_results/reset_bypass_real_neurons/`：

- `actual_neurons_trace.png`：真实类的状态轨迹和各自的自然放电/reset 时刻；
- `actual_neurons_trace.csv`：直接从真实对象读取的逐时间步状态；
- `actual_neurons_summary.json`：记录类名、放电数和 reset 时保留的支路状态。

这里 LIF 和 LSLIF 可能在不同时间自然放电，因为 LS 支路本来就会改变阈值
判定。因此，真实类实验负责验证“代码确实这样运行”，而后面的强制 reset toy
负责固定 reset 时刻、排除放电时刻不同这一混杂因素。论文中最好把两张图并列，
而不是用其中一个替代另一个。

### 受控 toy 实验

从仓库根目录运行。这个 toy experiment 不需要数据集、checkpoint 或
PyTorch，只依赖 NumPy 和 Matplotlib：

```bash
cd /workspace/Complementary-LIF
python -m pip install -r analysis/requirements.txt
python analysis/reset_bypass_toy.py
```

默认结果保存在 `analysis_results/reset_bypass_toy/`。其中：

- `reset_bypass_trace.png`：可以直接放大查看的轨迹图和 reset 保留率柱状图；
- `reset_bypass_trace.csv`：每个时间步的膜电位、LS 支路和 reset 指标；
- `reset_bypass_summary.json`：用于汇报的汇总指标。

运行完成后，终端也会打印 JSON 指标。重点比较
`mean_lif_retention_ratio` 和 `mean_ls_retention_ratio`；同时确认
`mean_absolute_main_reset_loss` 仍等于阈值。这表示 LS 保留了 reset 之外
的可用状态，而不是改变了 reset 本身。

如需指定输出目录和 reset 时刻：

```bash
python analysis/reset_bypass_toy.py \
  --out-dir analysis_results/reset_bypass_toy \
  --reset-steps 7 15
```

要查看全部可调参数（包括时间步、时间常数、阈值、输入脉冲和 LS 权重），运行：

```bash
python analysis/reset_bypass_toy.py --help
```

注意：指定的 reset 时刻必须满足主膜电位已经达到阈值。如果组合参数不满足
这一条件，脚本会提示 `not spike-valid`；此时应提高 `--pulse-amplitude`、
延后 `--reset-steps`，而不是绕过检查。例如：

```bash
python analysis/reset_bypass_toy.py \
  --steps 32 \
  --pulse-steps 32 \
  --pulse-amplitude 1.5 \
  --history-weight 0.6 \
  --reset-steps 8 18 \
  --out-dir analysis_results/reset_bypass_custom
```

For a paper, show the trace and retention bars as the mechanism figure, then
repeat the same forced reset intervention inside trained networks at several
layers and report paired changes in logits or accuracy as a separate task-level
validation.

## Relationship to existing experiments

- **Prefix masking:** measures dependence on early input; retain only as
  indirect, task-level evidence.
- **History zero/shuffle/time-shift:** establishes that a trained checkpoint
  uses the LS branch, but still does not quantify state preserved exactly at a
  reset.
- **Forced-reset trace (this experiment):** directly isolates and visualizes
  the proposed mechanism. It should be the first figure used to support the
  reset-bypass description.
