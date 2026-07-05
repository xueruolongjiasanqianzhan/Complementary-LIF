# -*- coding: utf-8 -*-
"""
SAMFF_Net (ERTrans) —— 三分支卷积 + 各分支卷积下采样 + SNN + 脉冲 STAtten 版本

本版本修改点：
1. 所有 MultiStepLIFNode 外面包了一层“五受体节律门控”：
   - 5 个固定方波 gate（周期 & 占空比手动设定，模拟 δ/θ/α/β/γ）
   - 5 个可学习权重 w_k
   - 对于每个时间步 t，计算 c[t] = sum_k w_k * m_k[t]，然后把输入乘以 c[t]
   - 再送入原本的 MultiStepLIFNode
2. 参数量几乎不变，只多了 5 个 w_k；计算量也只是多了一点标量运算。

整体流程（以单个脑区为例）：
1. 输入 raw_fea: [B, 1, C_i, T_raw]    （T_raw = samples，例如 750）
2. 三分支卷积（仍然是 ANN）：
   - spa_fea  = SpatialConv(raw_fea)           -> [B, T_raw, C_i]
   - temp_fea = TemporalConv(raw_fea)          -> [B, T_raw, C_i]
   - freq_fea = FFT(raw_fea).abs() + TemporalConv -> [B, T_raw, C_i]
3. 三个分支分别用 TimeDownsampleConv1D 在时间维卷积下采样：
   - spa_fea_ds  = TimeDownsampleConv1D_spa(spa_fea)   -> [B, T_enc, C_i]
   - temp_fea_ds = TimeDownsampleConv1D_temp(temp_fea) -> [B, T_enc, C_i]
   - freq_fea_ds = TimeDownsampleConv1D_freq(freq_fea) -> [B, T_enc, C_i]
4. 对下采样后的三个分支分别加时间位置编码 PE，然后相加融合：
   - cat_fea = PE(spa_ds) + PE(temp_ds) + PE(freq_ds)  -> [B, T_enc, C_i]
5. 脑区内部：BN1d(通道) -> RhythmFiveReceptorLIFNode + MultiStepLIF -> 脉冲 STAtten1D（Q/K/V 经过同样的节律 LIF）
6. 线性层 C_i -> 1，得到 [B, T_enc, 1]，所有脑区拼接成 [B, T_enc, R]
7. 脑区级：[B, T_enc, R] 再 BN1d + RhythmFiveReceptorLIFNode + STAtten1D
8. 时间维全局平均池化 -> [B, R]，最后全连接输出 [B, 2]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode


# -------------------- 五受体节律 LIF 封装 --------------------
class RhythmFiveReceptorLIFNode(nn.Module):
    """
    在 MultiStepLIFNode 外面包一层“五受体节律门控”：
    - 输入 x_tb: [T, B, C]（或 [T, B*, C]）
    - 对时间维 T 上的每个步 t，构造 5 个方波 gate m_k[t]（0/1）
    - 通过 5 个可学习权重 w_k 得到 c[t] = sum_k w_k * m_k[t]
    - 得到 x_mod[t] = c[t] * x[t]
    - 再送进 spikingjelly 的 MultiStepLIFNode，输出脉冲序列 s[t]

    注意：
    - gate 只依赖于 t，不依赖 batch / 通道；
    - 这在数学上等价于：共享权重 + 5 个受体通路 + 汇总到一个胞体电流。
    """

    def __init__(
        self,
        tau: float = 2.0,
        detach_reset: bool = True,
        backend: str = "torch",
        # 下面是 5 个受体的周期(以“时间步数量”为单位)，可根据 T_enc 调整
        gate_periods=(32, 16, 8, 4, 2),
        duty_ratios=(0.5, 0.5, 0.5, 0.5, 0.5),
    ):
        super().__init__()
        assert len(gate_periods) == 5
        assert len(duty_ratios) == 5

        # 里面是真正的 MultiStepLIFNode（胞体膜电位 + 发放）
        self.lif = MultiStepLIFNode(
            tau=tau,
            detach_reset=detach_reset,
            backend=backend,
        )

        # 5 个频段权重 w_k（可学习，初始值均值为 1/5，保证尺度稳定）
        self.w = nn.Parameter(torch.ones(5) / 5.0)

        # 方波的周期（步长）和占空比（固定，不可学习）
        periods = torch.tensor(gate_periods, dtype=torch.long)
        duty = torch.tensor(duty_ratios, dtype=torch.float32)
        self.register_buffer("periods", periods)
        self.register_buffer("duty_ratios", duty)

    def reset(self):
        # 兼容 spikingjelly.functional.reset_net
        self.lif.reset()

    def _compute_gate_coeff(self, T: int, device: torch.device):
        """
        根据当前序列长度 T 生成时间依赖的标量系数 c[t]：
        c[t] = sum_k w_k * m_k[t]，其中 m_k[t] 是 0/1 方波。
        返回: c [T]，已在对应 device 上。
        """
        P = self.periods.to(device)          # [5]
        rho = self.duty_ratios.to(device)    # [5]
        D = (rho * P).long().clamp(min=1)    # 每个受体打开的步数 [5]

        # t: [T,1]
        t = torch.arange(T, device=device).view(T, 1)     # [T,1]
        P_b = P.view(1, 5)                                # [1,5]
        D_b = D.view(1, 5)                                # [1,5]
        t_mod = torch.remainder(t, P_b)                   # [T,5]
        m = (t_mod < D_b).float()                         # [T,5] 0/1 方波

        # w: [5] -> [1,5]
        w = self.w.view(1, 5)
        c = (m * w).sum(dim=1)                            # [T]
        return c  # 每个时间步一个标量系数

    def forward(self, x_tb: torch.Tensor) -> torch.Tensor:
        """
        x_tb: [T, B, C]，时间维在最前
        返回: [T, B, C] 脉冲序列
        """
        assert x_tb.dim() == 3, "RhythmFiveReceptorLIFNode 目前假定输入为 [T,B,C]"
        T, B, C = x_tb.shape
        device = x_tb.device

        # 1) 计算时间依赖的缩放系数 c[t]
        c = self._compute_gate_coeff(T, device)          # [T]
        c = c.view(T, 1, 1)                              # [T,1,1]

        # 2) 对每个时间步的输入电流乘以 c[t]
        x_mod = x_tb * c                                 # [T,B,C]

        # 3) 送入 MultiStepLIFNode
        out_spk = self.lif(x_mod)                        # [T,B,C]
        return out_spk


# -------------------- 空间卷积（原版） --------------------
class SpatialConv(nn.Module):
    def __init__(self, conv_dim: int):
        """
        conv_dim: 该脑区通道数 C_i
        输入:  x [B, 1, C_i, T]
        输出:  [B, T, C_i]
        """
        super(SpatialConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, (conv_dim, 1), groups=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, conv_dim, (1, 1), bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ELU(),
            nn.Dropout(p=0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,C_i,T] -> [B,C_i,1,T]
        x = self.conv(x)                     # [B,C_i,1,T]
        x = x.view(x.size(0), x.size(1), -1) # [B,C_i,T]
        x = x.transpose(1, 2)                # [B,T,C_i]
        return x


# -------------------- 时间卷积（原版） --------------------
class TemporalConv(nn.Module):
    def __init__(self):
        """
        输入:  [B, 1, C_i, T]
        输出:  [B, T, C_i]
        """
        super(TemporalConv, self).__init__()

        self.conv = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, 16, (1, 64), groups=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 1, (1, 1)),
            nn.BatchNorm2d(1),
            nn.ELU(),
            nn.Dropout(p=0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,C_i,T]
        x = self.conv(x)                     # [B,1,C_i,T]
        x = x.view(x.size(0), -1, x.size(2)) # [B,T,C_i]
        return x


# -------------------- 时间位置编码 --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int):
        """
        max_len: 序列最大长度（这里用原始 samples）
        """
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len

    def computer_pe(self, d_model: int) -> nn.Parameter:
        pe = torch.zeros(self.max_len, d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-np.math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # [1,max_len,d_model]
        pe = nn.Parameter(pe, requires_grad=False)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        """
        d_model = x.size(-1)
        pe = self.computer_pe(d_model)  # [1,max_len,d_model]
        x = x + pe[:, : x.size(1)].to(x.device)
        return x


# -------------------- 时间维卷积下采样 --------------------
class TimeDownsampleConv1D(nn.Module):
    """
    对 [B, T, C] 在时间维做卷积下采样：
    - 先转 [B, C, T]
    - Conv1d(C->C, kernel_size=k_t, stride=ds_factor, padding=k_t//2)
    - 再转回 [B, T_enc, C]
    """

    def __init__(self, channels: int, ds_factor: int = 3, k_t: int = 5):
        """
        channels : 通道数（C_i）
        ds_factor: 下采样因子，T_enc = T_raw / ds_factor
        k_t      : 时间卷积核大小（建议奇数：3/5/7）
        """
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=k_t,
            stride=ds_factor,
            padding=k_t // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        返回: [B, T_enc, C]
        """
        B, T, C = x.shape
        x = x.transpose(1, 2)  # [B,C,T]
        x = self.conv(x)       # [B,C,T_enc]
        x = x.transpose(1, 2)  # [B,T_enc,C]
        return x


# -------------------- 1D STAtten（时间 × token）—— 脉冲版 Q/K/V --------------------
class STAtten1D(nn.Module):
    """
    一维 STAtten，用在：
    - 脑区内部：token = 通道 C_i
    - 脑区级：  token = 脑区数 R

    输入 / 输出：
    - 输入 x_tb: [T_enc, B, N]  （N = token 数）
    - 输出 out:  [T_enc, B, N]

    思路：
    - 每个 token j 有三个 head_dim 维向量 (E_q, E_k, E_v)
    - q = x * E_q, k = x * E_k, v = x * E_v
    - q/k/v 先通过 RhythmFiveReceptorLIFNode+MultiStepLIF 脉冲化，再参与 K^T V 和 Q(K^T V)
    - 时间上按 chunk_size 分块：
        K^T V -> [hd,hd]，然后 Q(K^T V) -> [L,hd]，L=chunk_size*N
    """

    def __init__(
        self,
        num_tokens: int,
        emb_dim: int = 64,
        num_heads: int = 1,
        chunk_size: int = 25,
        lif_tau: float = 2.0,
    ):
        super().__init__()
        assert emb_dim % num_heads == 0
        self.num_tokens = num_tokens
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.chunk_size = chunk_size

        # 每个 token 一个 head_dim 维 embedding（Q/K/V 三套）
        self.E_q = nn.Parameter(torch.randn(1, 1, num_tokens, self.head_dim))
        self.E_k = nn.Parameter(torch.randn(1, 1, num_tokens, self.head_dim))
        self.E_v = nn.Parameter(torch.randn(1, 1, num_tokens, self.head_dim))

        # 输出投影：head_dim -> 1
        self.out_proj = nn.Linear(self.head_dim, 1)

        # 缩放因子
        self.scale = 1.0 / (num_tokens * chunk_size)

        # -------- 新：Q/K/V 的“节律版” MultiStep LIF 封装 --------
        # 输入视作 [T, B*N, head_dim]
        self.lif_q = RhythmFiveReceptorLIFNode(
            tau=lif_tau, detach_reset=True, backend="torch"
        )
        self.lif_k = RhythmFiveReceptorLIFNode(
            tau=lif_tau, detach_reset=True, backend="torch"
        )
        self.lif_v = RhythmFiveReceptorLIFNode(
            tau=lif_tau, detach_reset=True, backend="torch"
        )

    def forward(self, x_tb: torch.Tensor) -> torch.Tensor:
        """
        x_tb: [T_enc, B, N]
        返回: [T_enc, B, N]
        """
        T, B, N = x_tb.shape
        assert N == self.num_tokens, f"num_tokens mismatch: got {N}, expect {self.num_tokens}"
        assert (
            T % self.chunk_size == 0
        ), f"T={T} 必须能被 chunk_size={self.chunk_size} 整除"

        num_chunks = T // self.chunk_size
        h = self.num_heads
        hd = self.head_dim

        # x: [T,B,N] -> [T,B,N,1]
        x_coeff = x_tb.unsqueeze(-1)  # [T,B,N,1]

        # ---------- 先得到实值的 q,k,v ----------

        # q,k,v: [T,B,N,hd]
        q = x_coeff * self.E_q
        k = x_coeff * self.E_k
        v = x_coeff * self.E_v

        # ---------- 脉冲化：把 q/k/v 送入节律 LIF ----------

        # 将 (B,N) 合并成 batch 维度，head_dim 作为通道：形状 [T, B*N, hd]
        q_flat = q.view(T, B * N, hd)
        k_flat = k.view(T, B * N, hd)
        v_flat = v.view(T, B * N, hd)

        # 通过 RhythmFiveReceptorLIFNode+MultiStepLIF 得到“脉冲版”的 Q/K/V
        q_spk = self.lif_q(q_flat)  # [T, B*N, hd]
        k_spk = self.lif_k(k_flat)  # [T, B*N, hd]
        v_spk = self.lif_v(v_flat)  # [T, B*N, hd]

        # reshape 回 [T,B,N,hd]
        q = q_spk.view(T, B, N, hd)
        k = k_spk.view(T, B, N, hd)
        v = v_spk.view(T, B, N, hd)

        # ---------- 后续和原版 STAtten 一样 ----------

        # 加 head 维： -> [T,B,h,N,hd]
        q = q.view(T, B, N, h, hd).permute(0, 1, 3, 2, 4).contiguous()
        k = k.view(T, B, N, h, hd).permute(0, 1, 3, 2, 4).contiguous()
        v = v.view(T, B, N, h, hd).permute(0, 1, 3, 2, 4).contiguous()

        # 时间分块：[num_chunks, chunk_size, B, h, N, hd] -> [num_chunks,B,h,chunk_size,N,hd]
        q_chunks = (
            q.view(num_chunks, self.chunk_size, B, h, N, hd)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
        )
        k_chunks = (
            k.view(num_chunks, self.chunk_size, B, h, N, hd)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
        )
        v_chunks = (
            v.view(num_chunks, self.chunk_size, B, h, N, hd)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
        )

        # 合并时间块和 token：L = chunk_size * N
        L = self.chunk_size * N
        q_chunks = q_chunks.view(num_chunks, B, h, L, hd)
        k_chunks = k_chunks.view(num_chunks, B, h, L, hd)
        v_chunks = v_chunks.view(num_chunks, B, h, L, hd)

        # STAtten 核心：K^T V -> [hd,hd]；Q(K^T V) -> [L,hd]
        attn = torch.matmul(
            k_chunks.transpose(-2, -1), v_chunks
        ) * self.scale  # [num_chunks,B,h,hd,hd]
        out = torch.matmul(q_chunks, attn)  # [num_chunks,B,h,L,hd]

        # 还原回 [T_enc,B,h,N,hd]
        out = (
            out.view(num_chunks, B, h, self.chunk_size, N, hd)
            .permute(0, 3, 1, 2, 4, 5)
            .contiguous()
        )
        out = out.view(T, B, h, N, hd)  # [T,B,h,N,hd]

        # 合并 head：-> [T,B,N,emb_dim]
        out = out.permute(0, 1, 3, 2, 4).reshape(T, B, N, self.emb_dim)

        # 输出投影：-> [T,B,N]
        out_proj = self.out_proj(out).squeeze(-1)

        # 残差
        x_out = x_tb + out_proj
        return x_out


# -------------------- ERTrans 主体 --------------------
class ERTrans(nn.Module):
    def __init__(
        self,
        samples: int,          # 原始时间长度 T_raw（例如 750）
        sa_emb_dim: int,       # 保留接口（原 TransBlock 的 emb_dim）
        d_ff: int,             # 保留接口（原 FFN 宽度）
        region_indices,        # 脑区划分列表（每个子列表是 1-based 通道索引）
        device,
        c_embed: int = 64,     # 脑区内部 STAtten embedding 维度
        d_br: int = 64,        # 脑区级 STAtten embedding 维度
        chunk_size: int = 25,  # STAtten 时间块大小
        num_heads: int = 1,    # STAtten 头数
        ds_factor: int = 3,    # 时间下采样因子：T_enc = samples / ds_factor
        ds_kernel: int = 5,    # 时间卷积下采样 kernel_size
    ):
        super(ERTrans, self).__init__()
        self.region_indices = region_indices
        self.samples = samples
        self.ds_factor = ds_factor

        assert self.samples % self.ds_factor == 0, "samples 必须能被 ds_factor 整除"
        self.T_enc = self.samples // self.ds_factor  # 下采样后的 SNN 时间步
        assert (
            self.T_enc % chunk_size == 0
        ), f"T_enc={self.T_enc} 必须能被 chunk_size={chunk_size} 整除"

        self.device = device

        spatial_conv = []
        temporal_conv = []
        frequency_conv = []

        time_ds_spa = []
        time_ds_temp = []
        time_ds_freq = []

        bn_region = []
        lif_region = []
        region_statten = []
        br_linear = []

        for i in range(len(self.region_indices)):
            C_i = len(self.region_indices[i])

            # 三分支卷积
            spatial_conv.append(SpatialConv(C_i))
            temporal_conv.append(TemporalConv())
            frequency_conv.append(TemporalConv())

            # 各分支各自时间卷积下采样
            time_ds_spa.append(TimeDownsampleConv1D(C_i, ds_factor=self.ds_factor, k_t=ds_kernel))
            time_ds_temp.append(TimeDownsampleConv1D(C_i, ds_factor=self.ds_factor, k_t=ds_kernel))
            time_ds_freq.append(TimeDownsampleConv1D(C_i, ds_factor=self.ds_factor, k_t=ds_kernel))

            # 脑区内部 BN（对 [B,T_enc,C_i] 按 C_i 归一化）
            bn_region.append(nn.BatchNorm1d(C_i))

            # 脑区内部 “节律五受体” LIF（替代原 MultiStepLIFNode）
            lif_region.append(
                RhythmFiveReceptorLIFNode(
                    tau=2.0, detach_reset=True, backend="torch"
                )
            )

            # 脑区内部 脉冲 STAtten: token = 通道 C_i
            region_statten.append(
                STAtten1D(
                    num_tokens=C_i,
                    emb_dim=c_embed,
                    num_heads=num_heads,
                    chunk_size=chunk_size,
                    lif_tau=2.0,
                )
            )

            # 通道聚合 C_i -> 1
            br_linear.append(nn.Linear(C_i, 1))

        self.spatial_conv = nn.ModuleList(spatial_conv)
        self.temporal_conv = nn.ModuleList(temporal_conv)
        self.frequency_conv = nn.ModuleList(frequency_conv)

        self.time_ds_spa = nn.ModuleList(time_ds_spa)
        self.time_ds_temp = nn.ModuleList(time_ds_temp)
        self.time_ds_freq = nn.ModuleList(time_ds_freq)

        self.bn_region = nn.ModuleList(bn_region)
        self.lif_region = nn.ModuleList(lif_region)
        self.region_statten = nn.ModuleList(region_statten)
        self.br_linear = nn.ModuleList(br_linear)

        # 时间位置编码（max_len 用原始 samples）
        self.pe = PositionalEncoding(self.samples)

        # 脑区级 STAtten：token = 脑区数 R
        self.num_regions = len(self.region_indices)
        self.br_bn = nn.BatchNorm1d(self.num_regions)

        # 脑区级 “节律五受体” LIF（替代原 MultiStepLIFNode）
        self.lif_br = RhythmFiveReceptorLIFNode(
            tau=2.0, detach_reset=True, backend="torch"
        )

        self.br_statten = STAtten1D(
            num_tokens=self.num_regions,
            emb_dim=d_br,
            num_heads=num_heads,
            chunk_size=chunk_size,
            lif_tau=2.0,
        )

        # 全局时间池化 + 分类
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.num_regions, 2)

    # ------- 脑区划分 -------
    def chunk_data(self, data, region_indices, dim: int):
        """
        data: [B,1,C_total,T]
        region_indices: 脑区划分（1-based 通道）
        返回: list，每个元素 [B,1,C_i,T]
        """
        chunks_list = []
        for indices in region_indices:
            idx0 = [i - 1 for i in indices]
            chunk = data.index_select(dim, torch.tensor(idx0).to(self.device))
            chunks_list.append(chunk)
        return chunks_list

    # ------- 前向 -------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, C_total, T_raw]，T_raw = samples
        """
        feas_out = []
        feas_in = self.chunk_data(x, self.region_indices, dim=2)  # list，每个 [B,1,C_i,T_raw]

        for i in range(len(self.region_indices)):
            raw_fea = feas_in[i]  # [B,1,C_i,T_raw]

            # ---------- 频域分支：FFT -> 幅值 -> TemporalConv ----------
            freq_signal = torch.fft.fft(raw_fea, dim=3)   # complex [B,1,C_i,T]
            freq_mag = freq_signal.abs().detach()
            freq_fea = self.frequency_conv[i](freq_mag)   # [B,T_raw,C_i]

            # ---------- 空间 / 时间分支 ----------
            spa_fea = self.spatial_conv[i](raw_fea)       # [B,T_raw,C_i]
            temp_fea = self.temporal_conv[i](raw_fea)     # [B,T_raw,C_i]

            # ---------- 各分支各自用卷积做时间下采样 ----------
            spa_fea = self.time_ds_spa[i](spa_fea)        # [B,T_enc,C_i]
            temp_fea = self.time_ds_temp[i](temp_fea)     # [B,T_enc,C_i]
            freq_fea = self.time_ds_freq[i](freq_fea)     # [B,T_enc,C_i]

            # ---------- 下采样后加位置编码，然后再融合 ----------
            spa_fea = self.pe(spa_fea)
            temp_fea = self.pe(temp_fea)
            freq_fea = self.pe(freq_fea)

            cat_fea = spa_fea + temp_fea + freq_fea       # [B,T_enc,C_i]

            # ---------- 脑区内部 BN + 节律 LIF + 脉冲 STAtten ----------
            cat_fea = self.bn_region[i](cat_fea.transpose(1, 2)).transpose(1, 2)
            region_tb = cat_fea.permute(1, 0, 2)          # [T_enc,B,C_i]
            region_tb = self.lif_region[i](region_tb)     # [T_enc,B,C_i]
            region_tb = self.region_statten[i](region_tb) # [T_enc,B,C_i]
            region_fea = region_tb.permute(1, 0, 2)       # [B,T_enc,C_i]

            # ---------- 通道聚合到 1 ----------
            br_fea = self.br_linear[i](region_fea)        # [B,T_enc,1]
            feas_out.append(br_fea)

        # ---------- 拼接所有脑区：token = R ----------
        local_fea = torch.cat(feas_out, dim=2)            # [B,T_enc,R]

        # ---------- 脑区级 BN + 节律 LIF + 脉冲 STAtten ----------
        local_fea = self.br_bn(local_fea.transpose(1, 2)).transpose(1, 2)
        local_tb = local_fea.permute(1, 0, 2)             # [T_enc,B,R]
        local_tb = self.lif_br(local_tb)                  # [T_enc,B,R]
        local_tb = self.br_statten(local_tb)              # [T_enc,B,R]
        local_fea = local_tb.permute(1, 0, 2)             # [B,T_enc,R]

        # ---------- 时间维全局平均池化 ----------
        output_fea = self.global_avg_pooling(
            local_fea.permute(0, 2, 1)
        ).squeeze(2)                                      # [B,R]

        # ---------- 分类 ----------
        out = self.fc(output_fea)                         # [B,2]
        return out


# -------------------- 自检 --------------------
if __name__ == "__main__":
    region_indices = [
        [1, 60, 2, 50, 36, 37, 51, 11, 44, 3, 30, 17, 31, 4, 45, 12],
        [58, 52, 53, 59, 13, 14, 54, 55],
        [25, 38, 21, 22, 39, 26, 46, 5, 32, 18, 33, 6, 47, 27, 40, 23, 61, 24, 41, 28],
        [15, 48, 7, 34, 19, 35, 8, 49, 16, 56, 42, 29, 43, 57],
        [9, 20, 10],
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B, C, T = 32, 61, 750
    x = torch.randn(B, 1, C, T).to(device)

    model = ERTrans(
        samples=T,
        sa_emb_dim=128,
        d_ff=128,
        region_indices=region_indices,
        device=device,
        c_embed=64,
        d_br=64,
        chunk_size=25,
        num_heads=1,
        ds_factor=3,   # 750 -> 250 个时间步
        ds_kernel=5,
    ).to(device)

    out = model(x)
    print("out shape:", out.shape)
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))
    print("T_enc (实际 SNN 时间步) =", model.T_enc)
