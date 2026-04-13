# LSLIF 在 DVS 数据（尤其 T=16）上的问题分析与改进建议

> 面向当前仓库实现（`modules/neuron.py`, `train.py`）的工程化诊断清单。

## 1) 先看当前 LSLIF 机制可能的“长时程衰减点”

当前 LSLIF 的放电膜电位为：

- `m_t`: 会 reset 的主膜分支
- `n_t`: 不 reset 的历史分支
- `total_mem = m_t + history_weight * n_t / step_t^history_power`

其中历史分支被 `step_t^history_power` 归一化。对于 `T=16`，若 `history_power=1.0`，则在后期步长历史项会显著缩小（量级约 1/16）。这会让 LSLIF 在长时间窗时“本该利用的历史信息”反而被主动压弱。

### 建议

- **优先扫参**：`history_power` 从 `1.0` 下调到 `0.0~0.5`。
- **可学习化**：将 `history_power` 参数化为可学习参数（建议约束在 `[0, 1]`），让模型自己决定时间归一强度。
- **替代归一化**：改为 EMA 或 `sqrt(t)` 归一（比线性 `t` 更温和）。

## 2) history 分支是否在“无意义时刻”被过早注入

当前 `history_mode` 支持 `all` 与 `post_spike`。DVS 序列前段常有噪声/稀疏事件，`all` 模式会在神经元尚未建立有效响应前持续注入历史分支，可能降低时序判别。

### 建议

- 在 DVS、T=16 设置下优先尝试 `history_mode=post_spike`。
- 做分层设置：浅层 `post_spike`，深层 `all`（需要将参数扩展为分层可配）。

## 3) history_weight 的可学习区间可能过窄

当启用 `history_learn_weight` 时，当前实现会把权重限制在 `[-0.8, 0.8]`。这对长序列任务可能不够，尤其当你希望历史分支在后期更“有存在感”时。

### 建议

- 扩大可学习范围（例如 `[-1.5, 1.5]`）。
- 或者改为按层/按通道学习（从“全局一个权重”升级为更细粒度参数）。

## 4) LSLIF 目前缺少动态时间常数（tau）机制

代码里 `newLIF / newLIFTauDep / newCLIF` 已有动态 tau，但 LSLIF 仍使用固定 tau。DVS 的时间统计明显非平稳（动作快慢、事件密度变化），固定 tau 在 T 增大时通常更难兼顾短时与长时模式。

### 建议

- 融合 LSLIF 与动态 tau（可参考 `BPTTNeuronTauDependent` 的 tau 更新方式）。
- 至少做一个 ablation：
  - `LSLIF-fixed-tau`
  - `LSLIF + tau_mode=spike`
  - `LSLIF + tau-dependent`

## 5) 训练目标对长序列不够“时间感知”

当前训练把每个时间步输出拼接后统一做 CE，相当于每个时刻都强制同等监督。对于 T=16 的 DVS，早期帧信息不完整、后期更有判别力，等权监督可能稀释有效梯度。

### 建议

- 改为时间加权 loss（例如后期更高权重）。
- 或采用“累计输出 + 中间深监督”的混合目标：
  - `L = CE(sum_t y_t, label) + λ * mean_t w_t * CE(y_t, label)`
- 在验证中同步报告：
  - early accuracy（前 1/4 时间）
  - late accuracy（后 1/4 时间）

## 6) DVS 数据预处理与增广还有优化空间

当前 DVSCIFAR10 路径下数据读取 `normalization=None`，空间增广有旋转/翻转，但缺少事件流强相关的时间增广。

### 建议

- 引入时间维增广：随机时间裁剪、时间抖动、事件丢弃（event drop）。
- 帧构建策略上尝试：
  - `split_by='time'` 与 `split_by='number'` 对比
  - 更细粒度帧数 + temporal pooling
- 增加按样本事件数归一化，缓解不同样本事件密度差异。

## 7) 结构层面的两条高收益尝试

- 在分类头前加入轻量 temporal attention / temporal conv（只在时间维建模）。
- 从“逐时刻独立前向 + 最后相加”升级到“显式时间聚合头”，让网络学习哪些时间片更关键。

## 8) 建议的最小实验矩阵（先小成本定位瓶颈）

固定 backbone 与训练预算，只改关键项：

1. `history_power`: `1.0 / 0.5 / 0.0`
2. `history_mode`: `all / post_spike`
3. `history_weight`: 固定 vs 可学习（并放宽范围）
4. loss：`等权 per-step` vs `后期加权`
5. tau：`fixed` vs `spike-adaptive`

优先观察：

- T=8 与 T=16 的精度差（是否缩小）
- 放电率曲线（是否后期塌陷）
- 梯度范数随时间步的分布（是否前强后弱）

---

如果你愿意，我下一步可以直接在代码里给出一个“最小侵入”的改动包（例如：`history_power` 可学习 + 时间加权 loss + 日志中新增分时段准确率），便于你快速复现实验。
