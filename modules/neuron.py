from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import LIFNode as LIFNode_sj
from spikingjelly.clock_driven.neuron import ParametricLIFNode as PLIFNode_sj
from torch import nn

from modules.surrogate import Rectangle


def _success_modulation_kwargs(kwargs):
    return dict(
        success_modulation_enable=kwargs.get('success_modulation_enable', False),
        success_modulation_gamma=kwargs.get('success_modulation_gamma', 0.05),
        success_modulation_mu=kwargs.get('success_modulation_mu', 0.05),
        success_modulation_q_max=kwargs.get('success_modulation_q_max', 0.1),
        success_modulation_delta=kwargs.get('success_modulation_delta', 0.0),
        success_modulation_warmup_epochs=kwargs.get('success_modulation_warmup_epochs', 5),
        success_modulation_min_count=kwargs.get('success_modulation_min_count', 1),
    )


class SuccessModulationMixin:
    """Epoch-level success-correlated neuronal modulation.

    Q and all spike-rate accumulators are buffers/statistics only; they never
    participate in gradient updates or loss computation.
    """

    def _init_success_modulation(
        self,
        success_modulation_enable: bool = False,
        success_modulation_gamma: float = 0.05,
        success_modulation_mu: float = 0.05,
        success_modulation_q_max: float = 0.1,
        success_modulation_delta: float = 0.0,
        success_modulation_warmup_epochs: int = 5,
        success_modulation_min_count: int = 1,
    ):
        self.success_modulation_enable = bool(success_modulation_enable)
        self.success_modulation_gamma = float(success_modulation_gamma)
        self.success_modulation_mu = float(success_modulation_mu)
        self.success_modulation_q_max = float(success_modulation_q_max)
        self.success_modulation_delta = float(success_modulation_delta)
        self.success_modulation_warmup_epochs = int(success_modulation_warmup_epochs)
        self.success_modulation_min_count = int(success_modulation_min_count)
        self.success_modulation_epoch = 0
        self._success_cached_spikes = []
        self.success_count = 0.0
        self.fail_count = 0.0
        self.register_buffer('Q', None)
        self.register_buffer('success_spike_sum', None)
        self.register_buffer('fail_spike_sum', None)

    def _ensure_success_buffers(self, spike_or_mem: torch.Tensor):
        if not self.success_modulation_enable or spike_or_mem.dim() < 2:
            return
        q_shape = (1,) + tuple(spike_or_mem.shape[1:])
        sum_shape = tuple(spike_or_mem.shape[1:])
        need_new = self.Q is None or tuple(self.Q.shape) != q_shape or self.Q.device != spike_or_mem.device
        if need_new:
            self.Q = torch.zeros(q_shape, device=spike_or_mem.device, dtype=torch.float32)
            self.success_spike_sum = torch.zeros(sum_shape, device=spike_or_mem.device, dtype=torch.float32)
            self.fail_spike_sum = torch.zeros(sum_shape, device=spike_or_mem.device, dtype=torch.float32)
            self.success_count = 0.0
            self.fail_count = 0.0

    def _success_modulation(self, mem: torch.Tensor) -> torch.Tensor:
        if (not self.success_modulation_enable) or self.success_modulation_epoch < self.success_modulation_warmup_epochs:
            return torch.zeros_like(mem)
        self._ensure_success_buffers(mem)
        if self.Q is None:
            return torch.zeros_like(mem)
        q = self.Q.to(device=mem.device, dtype=mem.dtype)
        if self.success_modulation_delta > 0.0:
            q = torch.sign(q) * torch.relu(torch.abs(q) - self.success_modulation_delta)
        q = torch.clamp(q, -self.success_modulation_q_max, self.success_modulation_q_max)
        return self.success_modulation_gamma * q

    def _success_fire(self, mem: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        return self._asn_fire(mem + self._success_modulation(mem), threshold)

    def _cache_success_spike(self, spike: torch.Tensor):
        if self.success_modulation_enable and self.training:
            self._ensure_success_buffers(spike)
            self._success_cached_spikes.append(spike.detach())

    def set_epoch(self, epoch: int):
        self.success_modulation_epoch = int(epoch)

    def clear_cached_spikes(self):
        self._success_cached_spikes = []

    def reset_epoch_stats(self):
        if self.success_spike_sum is not None:
            self.success_spike_sum.zero_()
        if self.fail_spike_sum is not None:
            self.fail_spike_sum.zero_()
        self.success_count = 0.0
        self.fail_count = 0.0
        self.clear_cached_spikes()

    @torch.no_grad()
    def update_success_stats(self, correct_mask: torch.Tensor):
        if (not self.success_modulation_enable) or (not self.training) or not self._success_cached_spikes:
            self.clear_cached_spikes()
            return
        spike_seq = torch.stack(self._success_cached_spikes, dim=0).detach().float()
        self.clear_cached_spikes()
        correct_mask = correct_mask.to(device=spike_seq.device, dtype=torch.bool)
        if spike_seq.shape[1] != correct_mask.numel():
            return
        self._ensure_success_buffers(spike_seq[0])
        T = spike_seq.shape[0]
        if correct_mask.any():
            self.success_spike_sum += spike_seq[:, correct_mask].sum(dim=(0, 1))
            self.success_count += float(correct_mask.sum().item() * T)
        fail_mask = ~correct_mask
        if fail_mask.any():
            self.fail_spike_sum += spike_seq[:, fail_mask].sum(dim=(0, 1))
            self.fail_count += float(fail_mask.sum().item() * T)

    @torch.no_grad()
    def finalize_epoch_stats(self):
        if not self.success_modulation_enable:
            return
        if (self.Q is not None and self.success_spike_sum is not None and self.fail_spike_sum is not None
                and self.success_count >= self.success_modulation_min_count
                and self.fail_count >= self.success_modulation_min_count):
            p_pos = self.success_spike_sum / max(self.success_count, 1.0)
            p_neg = self.fail_spike_sum / max(self.fail_count, 1.0)
            q = (p_pos - p_neg).reshape_as(self.Q)
            self.Q.mul_(1.0 - self.success_modulation_mu).add_(q, alpha=self.success_modulation_mu)
            self.Q.clamp_(-self.success_modulation_q_max, self.success_modulation_q_max)
        self.reset_epoch_stats()


def _iter_success_modules(model: nn.Module):
    for module in model.modules():
        if hasattr(module, 'update_success_stats') and getattr(module, 'success_modulation_enable', False):
            yield module


def set_epoch(model: nn.Module, epoch: int):
    for module in _iter_success_modules(model):
        module.set_epoch(epoch)


def reset_epoch_stats(model: nn.Module):
    for module in _iter_success_modules(model):
        module.reset_epoch_stats()


@torch.no_grad()
def update_success_stats(model: nn.Module, correct_mask: torch.Tensor):
    for module in _iter_success_modules(model):
        module.update_success_stats(correct_mask)


@torch.no_grad()
def finalize_epoch_stats(model: nn.Module):
    for module in _iter_success_modules(model):
        module.finalize_epoch_stats()


class ASNFireMixin(SuccessModulationMixin):
    """Optional ASN-style local lateral inhibition for 4D spike maps.

    The mixin is intentionally inert unless ``asn_enable`` is true so existing
    non-ASN experiments keep their original spike generation path.
    """

    def _init_asn(
        self,
        asn_enable: bool = False,
        asn_p: float = 0.5,
        asn_rho: float = 0.5,
        asn_seed: int = 2022,
        asn_detach_lateral: bool = False,
        layer_index: Optional[int] = None,
        **kwargs,
    ):
        self.asn_enable = bool(asn_enable)
        self.asn_p = float(asn_p)
        if not 0.0 <= self.asn_p <= 1.0:
            raise ValueError('asn_p must be in [0, 1].')
        self.asn_rho = float(asn_rho)
        self.asn_seed = int(asn_seed)
        self.asn_detach_lateral = bool(asn_detach_lateral)
        self.asn_layer_index = int(layer_index) if layer_index is not None else 0
        self.register_buffer('asn_mask', None, persistent=False)
        base_kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32) / 8.0
        base_kernel[..., 1, 1] = 0.0
        self.register_buffer('asn_kernel_base', base_kernel, persistent=False)
        self._init_success_modulation(**_success_modulation_kwargs(kwargs))

    def _asn_build_mask(self, mem: torch.Tensor) -> torch.Tensor:
        c, h, w = int(mem.shape[1]), int(mem.shape[2]), int(mem.shape[3])
        seed = self.asn_seed + self.asn_layer_index * 1000003
        gen = torch.Generator()
        gen.manual_seed(seed)
        mask = (torch.rand((1, c, h, w), generator=gen, dtype=torch.float32) < self.asn_p).to(torch.float32)
        return mask.to(device=mem.device, dtype=mem.dtype)

    def _asn_get_mask(self, mem: torch.Tensor) -> torch.Tensor:
        need_new_mask = (
            self.asn_mask is None
            or tuple(self.asn_mask.shape) != (1, int(mem.shape[1]), int(mem.shape[2]), int(mem.shape[3]))
            or self.asn_mask.device != mem.device
            or self.asn_mask.dtype != mem.dtype
        )
        if need_new_mask:
            self.asn_mask = self._asn_build_mask(mem)
        return self.asn_mask

    def _asn_fire(self, mem: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        if not self.asn_enable or mem.dim() != 4:
            return self.surrogate_function(mem - threshold)

        mask = self._asn_get_mask(mem)
        asn_spike = mask * self.surrogate_function(mem - threshold)
        lateral_source = asn_spike.detach() if self.asn_detach_lateral else asn_spike
        c = int(mem.shape[1])
        kernel = self.asn_kernel_base.to(device=mem.device, dtype=mem.dtype).repeat(c, 1, 1, 1)
        lateral = F.conv2d(lateral_source, kernel, bias=None, stride=1, padding=1, groups=c)
        non_asn_spike = (1.0 - mask) * self.surrogate_function(mem - self.asn_rho * lateral - threshold)
        return asn_spike + non_asn_spike


# multistep torch version
class CLIFSpike(nn.Module):
    def __init__(self, tau: float):
        super(CLIFSpike, self).__init__()
        # the symbol is corresponding to the paper
        # self.spike_func = surrogate_function
        self.spike_func = Rectangle()

        self.v_th = 1.
        self.gamma = 1 - 1. / tau

    def forward(self, x_seq):
        # x_seq.shape should be [T, N, *]
        _spike = []
        u = 0
        m = 0
        T = x_seq.shape[0]
        for t in range(T):
            u = self.gamma * u + x_seq[t, ...]
            spike = self.spike_func(u - self.v_th)
            _spike.append(spike)
            m = m * torch.sigmoid_((1. - self.gamma) * u) + spike
            u = u - spike * (self.v_th + torch.sigmoid_(m))
        # self.pre_spike_mem = torch.stack(_mem)
        return torch.stack(_spike, dim=0)


# spikingjelly single step version
class ComplementaryLIFNeuron(ASNFireMixin, LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)
        self._init_asn(
            asn_enable=kwargs.get('asn_enable', False),
            asn_p=kwargs.get('asn_p', 0.5),
            asn_rho=kwargs.get('asn_rho', 0.5),
            asn_seed=kwargs.get('asn_seed', 2022),
            asn_detach_lateral=kwargs.get('asn_detach_lateral', False),
            layer_index=kwargs.get('layer_index', None),
            **_success_modulation_kwargs(kwargs),
        )
        self.register_memory('m', 0.)  # Complementary memory

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)  # LIF charging
        self.m = self.m * torch.sigmoid(self.v / self.tau)  # Forming
        th_f = torch.as_tensor(self.v_threshold, device=self.v.device, dtype=self.v.dtype)
        spike = self._success_fire(self.v, th_f)  # LIF fire with optional ASN/success modulation
        self.m += spike  # Strengthen
        self.neuronal_reset(spike)  # LIF reset
        self.v = self.v - spike * torch.sigmoid(self.m)  # Reset
        self._cache_success_spike(spike)
        return spike

    def neuronal_charge(self, x: torch.Tensor):
        self._charging_v(x)

    def neuronal_reset(self, spike: torch.Tensor):
        self._reset(spike)

    def _charging_v(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0:
            if type(self.v) is float:
                self.v = x
            else:
                self.v = self.v * (1 - 1. / self.tau) + x
        else:
            if type(self.v) is float:
                self.v = self.v_reset * (1 - 1. / self.tau) + self.v_reset / self.tau + x
            else:
                self.v = self.v * (1 - 1. / self.tau) + self.v_reset / self.tau + x

    def _reset(self, spike):
        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold
        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset


# spikingjelly multiple step version
class MultiStepCLIFNeuron(ComplementaryLIFNeuron):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]
        spike_seq = []
        self.v_seq = []
        for t in range(x_seq.shape[0]):
            spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
            self.v_seq.append(self.v.unsqueeze(0))
        spike_seq = torch.cat(spike_seq, 0)
        self.v_seq = torch.cat(self.v_seq, 0)
        return spike_seq


class ReLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)




def ternary_spike_activation(x: torch.Tensor, binary: bool = False, temp: float = 1.0) -> torch.Tensor:
    """STE spike activation that emits {-1, 0, +1} in ternary mode."""
    if binary:
        out_s = torch.gt(x, 0.5)
        out_bp = torch.clamp(x, 0, 1)
        return (out_s.float() - out_bp).detach() + out_bp
    out_s = torch.sign(x)
    out_s = torch.where(torch.abs(x) < 0.5, torch.zeros((), device=x.device, dtype=x.dtype), out_s)
    out_bp = torch.clamp(x, -1, 1)
    return (out_s.float() - out_bp).detach() + out_bp


class TernarySpikeNeuron(SuccessModulationMixin, nn.Module):
    """LIF neuron that emits ternary spikes {-1, 0, +1}.

    This is aligned with ``三值神经元/models/spike_layer.py``: membrane
    follows the fixed-decay update ``mem = mem * 0.25 + x`` by default, then
    fires on normalized membrane ``mem / v_threshold`` with a dead zone of
    (-0.5, 0.5). Firing sites are reset to zero for both positive and
    negative ternary spikes.
    """

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = None,
        detach_reset: bool = False,
        tau_eps: float = 1e-6,
        fire_ratio: float = 1.0,
        temp: float = 3.0,
        ternary_decay: float = 0.25,
        **kwargs,
    ):
        super().__init__()
        self.tau = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.detach_reset = bool(detach_reset)
        self.tau_eps = float(tau_eps)
        self.fire_ratio = float(fire_ratio)
        self.temp = float(temp)
        self.ternary_decay = float(ternary_decay)
        self._init_success_modulation(**_success_modulation_kwargs(kwargs))
        self.v = None

    def reset(self):
        self.v = None

    def _ensure_state(self, x: torch.Tensor):
        if self.v is None or self.v.shape != x.shape or self.v.device != x.device:
            self.v = torch.zeros_like(x, dtype=torch.float32, device=x.device)

    def _ternary_fire(self, mem: torch.Tensor) -> torch.Tensor:
        threshold = torch.as_tensor(self.v_threshold, device=mem.device, dtype=mem.dtype)
        modulation = self._success_modulation(mem)
        return ternary_spike_activation((mem + modulation) / threshold, temp=self.temp) * self.fire_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)
        decay = torch.as_tensor(self.ternary_decay, device=self.v.device, dtype=self.v.dtype)
        mem = self.v * decay + x_f
        spike = self._ternary_fire(mem)
        rs = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = mem * (1.0 - torch.abs(rs))
        else:
            v_reset_t = torch.as_tensor(self.v_reset, device=mem.device, dtype=mem.dtype)
            self.v = torch.where(torch.abs(rs) > 0, v_reset_t, mem)
        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class LSLIFNeuron(ASNFireMixin, nn.Module):
    """
    LIF variant with an auxiliary history branch.

    The primary membrane ``v`` follows the usual leaky integration and is the state
    that gets reset after spiking. In parallel, an auxiliary state ``n`` integrates
    the same inputs with the same leakage but never spikes and never resets. Before
    thresholding, they are fused into

      M_t = m_t + beta * n_t / step_t^power

    so that the auxiliary branch acts like a residual path carrying longer-range
    membrane history into the firing decision while keeping a smoother gradient path.
    """

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = None,
        surrogate_function: Optional[Callable] = None,
        detach_reset: bool = False,
        tau_eps: float = 1e-6,
        history_weight: float = 1.0,
        history_power: float = 1.0,
        history_eps: float = 1e-6,
        history_learn_weight: bool = False,
        history_weight_lo: Optional[float] = None,
        history_weight_hi: Optional[float] = None,
        history_weight_per_step: bool = False,
        history_max_steps: int = 16,
        history_learn_power: bool = False,
        history_mode: str = 'all',
        layer_index: Optional[int] = None,
        total_layers: Optional[int] = None,
        asn_enable: bool = False,
        asn_p: float = 0.5,
        asn_rho: float = 0.5,
        asn_seed: int = 2022,
        asn_detach_lateral: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.tau = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.detach_reset = bool(detach_reset)
        self.tau_eps = float(tau_eps)
        self.history_weight = float(history_weight)
        self.history_power = float(history_power)
        self.history_eps = float(history_eps)
        self.history_learn_weight = bool(history_learn_weight)
        self.history_weight_per_step = bool(history_weight_per_step)
        self.history_max_steps = int(max(1, history_max_steps))
        self.history_learn_power = bool(history_learn_power)
        self.history_mode = str(history_mode).lower()
        if self.history_mode not in {'all', 'post_spike', 'half'}:
            raise ValueError(f"Unsupported history_mode: {history_mode}. Expected 'all', 'post_spike', or 'half'.")
        self.layer_index = int(layer_index) if layer_index is not None else None
        self.total_layers = int(total_layers) if total_layers is not None else None
        if self.history_mode == 'half':
            if self.layer_index is None or self.total_layers is None:
                self.history_mode = 'all'
            else:
                self.history_mode = 'post_spike' if self.layer_index < (self.total_layers // 2) else 'all'
        self.history_weight_bounded = history_weight_lo is not None or history_weight_hi is not None
        if self.history_weight_bounded and (history_weight_lo is None or history_weight_hi is None):
            raise ValueError('history_weight_lo and history_weight_hi must be provided together.')
        self.history_weight_lo = float(history_weight_lo) if history_weight_lo is not None else None
        self.history_weight_hi = float(history_weight_hi) if history_weight_hi is not None else None
        if self.history_weight_bounded and self.history_weight_hi <= self.history_weight_lo:
            raise ValueError('history_weight_hi must be larger than history_weight_lo.')
        self.history_power_lo = 0.0
        self.history_power_hi = 2.0
        self.surrogate_function = surrogate_function if surrogate_function is not None else Rectangle()
        self._init_asn(
            asn_enable=asn_enable,
            asn_p=asn_p,
            asn_rho=asn_rho,
            asn_seed=asn_seed,
            asn_detach_lateral=asn_detach_lateral,
            layer_index=self.layer_index,
            **_success_modulation_kwargs(kwargs),
        )

        def _inv_sigmoid(x: float) -> float:
            x_t = torch.tensor(float(x), dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
            return float(torch.log(x_t / (1.0 - x_t)).item())

        def _inv_softplus(x: float) -> float:
            x_t = torch.tensor(float(x), dtype=torch.float32).clamp_min(1e-6)
            return float(torch.log(torch.expm1(x_t)).item())

        if self.history_learn_weight:
            if self.history_weight_bounded:
                init_weight = float(np.clip(self.history_weight, self.history_weight_lo, self.history_weight_hi))
                scale = self.history_weight_hi - self.history_weight_lo
                init_unit = (init_weight - self.history_weight_lo) / max(scale, 1e-6)
                init_raw = _inv_sigmoid(init_unit)
            else:
                init_raw = _inv_softplus(self.history_weight)
            if self.history_weight_per_step:
                init_tensor = torch.full((self.history_max_steps,), init_raw, dtype=torch.float32)
                self.history_weight_raw = nn.Parameter(init_tensor)
            else:
                self.history_weight_raw = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))
        if self.history_learn_power:
            init_power = float(np.clip(self.history_power, self.history_power_lo, self.history_power_hi))
            scale = self.history_power_hi - self.history_power_lo
            init_unit = (init_power - self.history_power_lo) / max(scale, 1e-6)
            init_raw = _inv_sigmoid(init_unit)
            self.history_power_raw = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

        self.v = None
        self.n = None
        self.has_fired = None
        self.step_count = 0
        # Evaluation-only controls used by the history-branch intervention
        # experiment. Plain attributes keep checkpoints backward compatible.
        self.history_intervention = 'normal'
        self.history_intervention_shift = 1
        self._history_intervention_buffer = []

    def reset(self):
        self.v = None
        self.n = None
        self.has_fired = None
        self.step_count = 0
        self._history_intervention_buffer = []

    def set_history_intervention(self, mode: str = 'normal', shift: int = 1):
        """Configure an evaluation-time intervention on the fused history term.

        ``shuffle`` deterministically rolls the batch by one sample, preserving
        values while breaking sample correspondence. ``time_shift`` delays the
        term by ``shift`` steps and emits zeros until enough history exists.
        """
        mode = str(mode).lower()
        if mode not in {'normal', 'zero', 'shuffle', 'time_shift'}:
            raise ValueError(f'Unsupported history intervention: {mode}')
        if int(shift) < 1:
            raise ValueError(f'history intervention shift must be >= 1, got {shift}')
        self.history_intervention = mode
        self.history_intervention_shift = int(shift)
        self._history_intervention_buffer = []

    def _intervene_history_term(self, history_term: torch.Tensor) -> torch.Tensor:
        mode = self.history_intervention
        if mode == 'normal':
            return history_term
        if mode == 'zero':
            return torch.zeros_like(history_term)
        if mode == 'shuffle':
            if history_term.shape[0] < 2:
                raise ValueError('shuffle history intervention requires batch size >= 2')
            return torch.roll(history_term, shifts=1, dims=0)
        if mode == 'time_shift':
            self._history_intervention_buffer.append(history_term)
            if len(self._history_intervention_buffer) <= self.history_intervention_shift:
                return torch.zeros_like(history_term)
            delayed = self._history_intervention_buffer.pop(0)
            return delayed
        raise RuntimeError(f'Unexpected history intervention: {mode}')

    def _ensure_state(self, x: torch.Tensor):
        need_init = (
            self.v is None
            or self.v.shape != x.shape
            or self.v.device != x.device
        )
        if need_init:
            self.v = torch.zeros_like(x, dtype=torch.float32, device=x.device)
            self.n = torch.zeros_like(x, dtype=torch.float32, device=x.device)
            self.has_fired = torch.zeros_like(x, dtype=torch.bool, device=x.device)
            self.step_count = 0


    def _get_history_weight(self, dtype: torch.dtype, device: torch.device, step_count: Optional[int] = None):
        if self.history_learn_weight:
            if self.history_weight_per_step:
                idx = 0 if step_count is None else max(0, min(int(step_count) - 1, self.history_max_steps - 1))
                weight_raw = self.history_weight_raw[idx]
            else:
                weight_raw = self.history_weight_raw
            if self.history_weight_bounded:
                weight_unit = torch.sigmoid(weight_raw)
                weight = self.history_weight_lo + (self.history_weight_hi - self.history_weight_lo) * weight_unit
            else:
                weight = F.softplus(weight_raw)
            return weight.to(dtype=dtype, device=device)
        return torch.as_tensor(self.history_weight, dtype=dtype, device=device)

    def _get_history_power(self, dtype: torch.dtype, device: torch.device):
        if self.history_learn_power:
            power_unit = torch.sigmoid(self.history_power_raw)
            power = self.history_power_lo + (self.history_power_hi - self.history_power_lo) * power_unit
            return power.to(dtype=dtype, device=device)
        return torch.as_tensor(self.history_power, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)

        tau_eff = torch.as_tensor(self.tau, device=self.v.device, dtype=self.v.dtype)
        if self.decay_input:
            m_t = self.v + (x_f - self.v) / (tau_eff + self.tau_eps)
            n_t = self.n + (x_f - self.n) / (tau_eff + self.tau_eps)
        else:
            decay = 1.0 - 1.0 / (tau_eff + self.tau_eps)
            decay = torch.clamp(decay, 0.0, 1.0)
            m_t = self.v * decay + x_f
            n_t = self.n * decay + x_f

        self.step_count += 1
        step_t = torch.as_tensor(float(self.step_count), device=m_t.device, dtype=m_t.dtype)
        history_power = self._get_history_power(dtype=m_t.dtype, device=m_t.device)
        norm = torch.pow(step_t + self.history_eps, history_power)
        history_weight = self._get_history_weight(dtype=m_t.dtype, device=m_t.device, step_count=self.step_count)
        history_term = history_weight * (n_t / norm)
        if self.history_mode == 'post_spike':
            history_term = history_term * self.has_fired.to(dtype=history_term.dtype)
        history_term = self._intervene_history_term(history_term)
        total_mem = m_t + history_term
        # Keep diagnostics opt-in: retaining this intermediate during ordinary
        # training would unnecessarily extend the autograd graph's lifetime.
        if getattr(self, 'gradient_probe_enabled', False):
            self.last_v_pre = total_mem

        th_f = torch.as_tensor(self.v_threshold, device=self.v.device, dtype=self.v.dtype)
        spike = self._success_fire(total_mem, th_f)

        rs = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = m_t - rs * th_f
        else:
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            self.v = torch.where(rs.bool(), v_reset_t, m_t)

        self.n = n_t
        self.has_fired = torch.logical_or(self.has_fired, rs.bool())
        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class RPLIFNeuron(ASNFireMixin, nn.Module):
    """Refractory-Period LIF with spike-triggered threshold dynamics.

    After each spike decision and hard reset, every neuron/position updates its
    own threshold for the next step as

        V_th_next = V_init_th * (1 - S_t) + alpha * V_th_current * S_t

    This realizes a relative refractory period without freezing membrane
    dynamics or discarding the next input current.
    """

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = None,
        surrogate_function: Optional[Callable] = None,
        detach_reset: bool = True,
        tau_eps: float = 1e-6,
        rplif_alpha: float = 1.5,
        rplif_v_init_th: Optional[float] = None,
        refractory_step: int = 1,
        layer_index: Optional[int] = None,
        asn_enable: bool = False,
        asn_p: float = 0.5,
        asn_rho: float = 0.5,
        asn_seed: int = 2022,
        asn_detach_lateral: bool = False,
        **kwargs,
    ):
        super().__init__()
        if int(refractory_step) != 1:
            raise ValueError('RPLIF currently implements the paper default refractory_step=1 threshold dynamics.')
        self.tau = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_init_th = float(v_threshold if rplif_v_init_th is None else rplif_v_init_th)
        self.v_reset = 0.0 if v_reset is None else float(v_reset)
        self.detach_reset = bool(detach_reset)
        self.tau_eps = float(tau_eps)
        self.rplif_alpha = float(rplif_alpha)
        self.refractory_step = int(refractory_step)
        self.surrogate_function = surrogate_function if surrogate_function is not None else Rectangle()
        self._init_asn(
            asn_enable=asn_enable,
            asn_p=asn_p,
            asn_rho=asn_rho,
            asn_seed=asn_seed,
            asn_detach_lateral=asn_detach_lateral,
            layer_index=layer_index,
            **_success_modulation_kwargs(kwargs),
        )
        self.v = None
        self.dynamic_threshold = None

    def reset(self):
        self.v = None
        self.dynamic_threshold = None

    def reset_state(self):
        self.reset()

    def _ensure_state(self, x: torch.Tensor):
        need_init = self.v is None or self.v.shape != x.shape or self.v.device != x.device
        if need_init:
            self.v = torch.zeros_like(x, dtype=torch.float32, device=x.device)
            self.dynamic_threshold = torch.full_like(self.v, self.v_init_th)

    def _charge(self, x_f: torch.Tensor) -> torch.Tensor:
        tau_eff = torch.as_tensor(self.tau, device=self.v.device, dtype=self.v.dtype)
        if self.decay_input:
            return self.v + (x_f - self.v) / (tau_eff + self.tau_eps)
        decay = 1.0 - 1.0 / (tau_eff + self.tau_eps)
        decay = torch.clamp(decay, 0.0, 1.0)
        return self.v * decay + x_f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)
        u_t = self._charge(x_f)
        threshold = self.dynamic_threshold.to(device=u_t.device, dtype=u_t.dtype)
        spike = self._success_fire(u_t, threshold)
        rs = spike.detach() if self.detach_reset else spike
        v_reset_t = torch.as_tensor(self.v_reset, device=u_t.device, dtype=u_t.dtype)
        self.v = u_t * (1.0 - rs) + v_reset_t * rs
        s_for_threshold = spike.detach()
        init_th = torch.as_tensor(self.v_init_th, device=u_t.device, dtype=u_t.dtype)
        alpha = torch.as_tensor(self.rplif_alpha, device=u_t.device, dtype=u_t.dtype)
        self.dynamic_threshold = init_th * (1.0 - s_for_threshold) + alpha * threshold * s_for_threshold
        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class LSRPLIFNeuron(LSLIFNeuron):
    """LSLIF auxiliary-history branch plus RPLIF dynamic thresholds."""

    def __init__(self, *args, rplif_alpha: float = 1.5, rplif_v_init_th: Optional[float] = None,
                 refractory_step: int = 1, detach_reset: bool = True, **kwargs):
        if int(refractory_step) != 1:
            raise ValueError('LSRPLIF currently implements the paper default refractory_step=1 threshold dynamics.')
        super().__init__(*args, detach_reset=detach_reset, **kwargs)
        self.v_init_th = float(self.v_threshold if rplif_v_init_th is None else rplif_v_init_th)
        self.rplif_alpha = float(rplif_alpha)
        self.refractory_step = int(refractory_step)
        self.dynamic_threshold = None

    def reset(self):
        super().reset()
        self.dynamic_threshold = None

    def _ensure_state(self, x: torch.Tensor):
        super()._ensure_state(x)
        if self.dynamic_threshold is None or self.dynamic_threshold.shape != x.shape or self.dynamic_threshold.device != x.device:
            self.dynamic_threshold = torch.full_like(self.v, self.v_init_th)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)
        tau_eff = torch.as_tensor(self.tau, device=self.v.device, dtype=self.v.dtype)
        if self.decay_input:
            m_t = self.v + (x_f - self.v) / (tau_eff + self.tau_eps)
            n_t = self.n + (x_f - self.n) / (tau_eff + self.tau_eps)
        else:
            decay = torch.clamp(1.0 - 1.0 / (tau_eff + self.tau_eps), 0.0, 1.0)
            m_t = self.v * decay + x_f
            n_t = self.n * decay + x_f

        self.step_count += 1
        step_t = torch.as_tensor(float(self.step_count), device=m_t.device, dtype=m_t.dtype)
        norm = torch.pow(step_t + self.history_eps, self._get_history_power(dtype=m_t.dtype, device=m_t.device))
        history_term = self._get_history_weight(dtype=m_t.dtype, device=m_t.device, step_count=self.step_count) * (n_t / norm)
        if self.history_mode == 'post_spike':
            history_term = history_term * self.has_fired.to(dtype=history_term.dtype)
        total_mem = m_t + history_term

        threshold = self.dynamic_threshold.to(device=total_mem.device, dtype=total_mem.dtype)
        spike = self._success_fire(total_mem, threshold)
        rs = spike.detach() if self.detach_reset else spike
        self.v = m_t * (1.0 - rs)
        self.n = n_t
        self.has_fired = torch.logical_or(self.has_fired, rs.bool())

        s_for_threshold = spike.detach()
        init_th = torch.as_tensor(self.v_init_th, device=total_mem.device, dtype=total_mem.dtype)
        alpha = torch.as_tensor(self.rplif_alpha, device=total_mem.device, dtype=total_mem.dtype)
        self.dynamic_threshold = init_th * (1.0 - s_for_threshold) + alpha * threshold * s_for_threshold
        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class LSTernarySpikeNeuron(LSLIFNeuron):
    """Ternary spike neuron with the LSLIF auxiliary history (LS) branch.

    The ternary membrane dynamics are aligned with
    ``三值神经元/models/spike_layer.py`` by using the same fixed decay as
    ``TernarySpikeNeuron`` before applying the LS auxiliary history branch.
    """

    def __init__(
        self, *args, fire_ratio: float = 1.0, temp: float = 3.0,
        ternary_decay: float = 0.25, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.fire_ratio = float(fire_ratio)
        self.temp = float(temp)
        self.ternary_decay = float(ternary_decay)

    def _ternary_fire(self, mem: torch.Tensor) -> torch.Tensor:
        threshold = torch.as_tensor(self.v_threshold, device=mem.device, dtype=mem.dtype)
        modulation = self._success_modulation(mem)
        return ternary_spike_activation((mem + modulation) / threshold, temp=self.temp) * self.fire_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)

        decay = torch.as_tensor(self.ternary_decay, device=self.v.device, dtype=self.v.dtype)
        m_t = self.v * decay + x_f
        n_t = self.n * decay + x_f

        self.step_count += 1
        step_t = torch.as_tensor(float(self.step_count), device=m_t.device, dtype=m_t.dtype)
        history_power = self._get_history_power(dtype=m_t.dtype, device=m_t.device)
        norm = torch.pow(step_t + self.history_eps, history_power)
        history_weight = self._get_history_weight(dtype=m_t.dtype, device=m_t.device, step_count=self.step_count)
        history_term = history_weight * (n_t / norm)
        if self.history_mode == 'post_spike':
            history_term = history_term * self.has_fired.to(dtype=history_term.dtype)
        total_mem = m_t + history_term

        spike = self._ternary_fire(total_mem)
        rs = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = m_t * (1.0 - torch.abs(rs))
        else:
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            self.v = torch.where(torch.abs(rs) > 0, v_reset_t, m_t)

        self.n = n_t
        self.has_fired = torch.logical_or(self.has_fired, torch.abs(rs) > 0)
        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class HALIFNeuron(ASNFireMixin, nn.Module):
    """
    Heterogeneous Autonomous LIF neuron.

    A fixed ``auto_ratio`` of neuron positions use non-leaky autonomous dynamics
    with heterogeneous intrinsic drives, while the remaining positions use the
    standard leaky LIF update. Normal neurons use soft reset; autonomous neurons
    use hard reset to ``v_reset``.
    """

    _instance_count = 0

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        surrogate_function: Optional[Callable] = None,
        detach_reset: bool = False,
        tau_eps: float = 1e-6,
        auto_ratio: float = 0.1,
        num_auto_groups: int = 3,
        drive_periods: Optional[List[int]] = None,
        auto_drive_values: Optional[List[float]] = None,
        auto_T: int = 16,
        auto_seed: int = 2022,
        layer_index: Optional[int] = None,
        asn_enable: bool = False,
        asn_p: float = 0.5,
        asn_rho: float = 0.5,
        asn_seed: int = 2022,
        asn_detach_lateral: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.tau = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_reset = float(v_reset)
        self.detach_reset = bool(detach_reset)
        self.tau_eps = float(tau_eps)
        self.auto_ratio = float(auto_ratio)
        if not 0.0 <= self.auto_ratio <= 1.0:
            raise ValueError('auto_ratio must be in [0, 1].')
        self.num_auto_groups = int(max(1, num_auto_groups))
        self.drive_periods = None if drive_periods is None else [int(max(1, p)) for p in drive_periods]
        self.auto_drive_values = None if auto_drive_values is None else [float(v) for v in auto_drive_values]
        self.auto_T = int(max(1, auto_T))
        self.auto_seed = int(auto_seed)
        self.layer_index = int(layer_index) if layer_index is not None else HALIFNeuron._instance_count
        HALIFNeuron._instance_count += 1
        self.surrogate_function = surrogate_function if surrogate_function is not None else Rectangle()
        self._init_asn(
            asn_enable=asn_enable,
            asn_p=asn_p,
            asn_rho=asn_rho,
            asn_seed=asn_seed,
            asn_detach_lateral=asn_detach_lateral,
            layer_index=self.layer_index,
            **_success_modulation_kwargs(kwargs),
        )

        self.v = None
        self.register_buffer('auto_mask', None)
        self.register_buffer('auto_drive', None)
        self._auto_shape = None
        self._auto_init_T = None

    def reset(self):
        self.v = None

    def reset_state(self):
        self.reset()

    def _step_shape(self, x: torch.Tensor):
        if x.dim() < 2:
            raise ValueError('HALIF expects a batched input with shape [B, ...] or [T, B, ...].')
        return (1,) + tuple(x.shape[1:])

    def _get_drive_values(self, T: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.auto_drive_values is not None:
            values = self.auto_drive_values
        else:
            if self.drive_periods is not None:
                periods = self.drive_periods
            else:
                T = int(max(1, T))
                periods = [2 * T, T, max(1, T // 2)]
            values = [self.v_threshold / float(max(1, p)) for p in periods]

        if len(values) == 0:
            raise ValueError('HALIF requires at least one auto drive value.')
        if len(values) < self.num_auto_groups:
            values = values + [values[-1]] * (self.num_auto_groups - len(values))
        return torch.as_tensor(values[:self.num_auto_groups], dtype=dtype, device=device)

    def _ensure_auto_buffers(self, x: torch.Tensor, T: int):
        mask_shape = self._step_shape(x)
        need_init = (
            self.auto_mask is None
            or tuple(self.auto_mask.shape) != mask_shape
            or self.auto_mask.device != x.device
            or self._auto_init_T != int(T)
        )
        if not need_init:
            return

        dtype = torch.float32
        device = x.device
        auto_mask = torch.zeros(mask_shape, dtype=dtype, device=device)
        auto_drive = torch.zeros(mask_shape, dtype=dtype, device=device)
        num_positions = int(np.prod(mask_shape[1:]))
        num_auto = int(round(num_positions * self.auto_ratio))
        num_auto = max(0, min(num_positions, num_auto))

        if num_auto > 0:
            seed = self.auto_seed + self.layer_index * 1000003
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            perm = torch.randperm(num_positions, generator=generator, device=device)
            auto_indices = perm[:num_auto]
            flat_mask = auto_mask.view(-1)
            flat_drive = auto_drive.view(-1)
            flat_mask[auto_indices] = 1.0

            drive_values = self._get_drive_values(T, dtype=dtype, device=device)
            group_ids = torch.arange(num_auto, device=device) % self.num_auto_groups
            shuffled_group_ids = group_ids[torch.randperm(num_auto, generator=generator, device=device)]
            flat_drive[auto_indices] = drive_values[shuffled_group_ids]

        self.auto_mask = auto_mask
        self.auto_drive = auto_drive
        self._auto_shape = mask_shape
        self._auto_init_T = int(T)

    def _ensure_state(self, x: torch.Tensor):
        if self.v is None or self.v.shape != x.shape or self.v.device != x.device:
            self.v = torch.zeros_like(x, dtype=torch.float32, device=x.device)

    def _lambda_decay(self, device: torch.device) -> torch.Tensor:
        decay = 1.0 - 1.0 / (torch.as_tensor(self.tau, device=device, dtype=torch.float32) + self.tau_eps)
        return torch.clamp(decay, 0.0, 1.0)

    def _single_step_forward(self, x: torch.Tensor, T: int) -> torch.Tensor:
        self._ensure_state(x)
        self._ensure_auto_buffers(x, T)
        x_f = x.to(torch.float32)
        th_f = torch.as_tensor(self.v_threshold, device=x.device, dtype=torch.float32)
        normal_mask = 1.0 - self.auto_mask

        lambda_decay = self._lambda_decay(x.device)
        if self.decay_input:
            v_normal = self.v * lambda_decay + x_f / (torch.as_tensor(self.tau, device=x.device, dtype=torch.float32) + self.tau_eps)
        else:
            v_normal = self.v * lambda_decay + x_f
        v_auto = self.v + x_f + self.auto_drive
        v_pre = normal_mask * v_normal + self.auto_mask * v_auto

        spike = self._success_fire(v_pre, th_f)
        rs = spike.detach() if self.detach_reset else spike
        v_normal_reset = v_pre - rs * th_f
        v_reset_t = torch.as_tensor(self.v_reset, device=x.device, dtype=torch.float32)
        v_auto_reset = (1.0 - rs) * v_pre + rs * v_reset_t
        self.v = normal_mask * v_normal_reset + self.auto_mask * v_auto_reset
        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() >= 5 or (x.dim() == 3 and int(x.shape[0]) == self.auto_T):
            T = int(x.shape[0])
            return torch.stack([self._single_step_forward(x[t], T) for t in range(T)], dim=0)
        return self._single_step_forward(x, self.auto_T)


class QKVLIFNeuron(ASNFireMixin, nn.Module):
    """
    LIF neuron with a causal QKV attention branch inside membrane charging.

    At each step, the residual membrane before injecting the current input is used
    as the current query. Historical residual membranes are used as keys, and
    historical inputs are used as values. The attention context is injected into
    the current membrane before firing/reset, so the neuron can dynamically recall
    useful historical inputs while preserving a standard causal SNN interface.
    """

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = None,
        surrogate_function: Optional[Callable] = None,
        detach_reset: bool = False,
        tau_eps: float = 1e-6,
        qkv_alpha: float = 0.1,
        qkv_learn_alpha: bool = True,
        qkv_w_q: float = 1.0,
        qkv_w_k: float = 1.0,
        qkv_w_v: float = 1.0,
        qkv_learn_w: bool = True,
        qkv_max_history: int = 0,
        qkv_detach_history: bool = False,
        layer_index: Optional[int] = None,
        asn_enable: bool = False,
        asn_p: float = 0.5,
        asn_rho: float = 0.5,
        asn_seed: int = 2022,
        asn_detach_lateral: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.tau = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.detach_reset = bool(detach_reset)
        self.tau_eps = float(tau_eps)
        self.qkv_max_history = int(max(0, qkv_max_history))
        self.qkv_detach_history = bool(qkv_detach_history)
        self.layer_index = int(layer_index) if layer_index is not None else None
        self.surrogate_function = surrogate_function if surrogate_function is not None else Rectangle()
        self._init_asn(
            asn_enable=asn_enable,
            asn_p=asn_p,
            asn_rho=asn_rho,
            asn_seed=asn_seed,
            asn_detach_lateral=asn_detach_lateral,
            layer_index=self.layer_index,
            **_success_modulation_kwargs(kwargs),
        )

        qkv_weights = {
            'w_q': float(qkv_w_q),
            'w_k': float(qkv_w_k),
            'w_v': float(qkv_w_v),
        }
        for name, value in qkv_weights.items():
            tensor = torch.tensor(value, dtype=torch.float32)
            if qkv_learn_w:
                setattr(self, name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)
        if qkv_learn_alpha:
            self.qkv_alpha = nn.Parameter(torch.tensor(float(qkv_alpha), dtype=torch.float32))
        else:
            self.register_buffer('qkv_alpha', torch.tensor(float(qkv_alpha), dtype=torch.float32))

        self.v = None
        self.mem_history = []
        self.input_history = []

    def reset(self):
        self.v = None
        self.mem_history = []
        self.input_history = []

    def _ensure_state(self, x: torch.Tensor):
        need_init = (
            self.v is None
            or self.v.shape != x.shape
            or self.v.device != x.device
        )
        if need_init:
            self.v = torch.zeros_like(x, dtype=torch.float32, device=x.device)
            self.mem_history = []
            self.input_history = []

    def _lif_residual(self) -> torch.Tensor:
        tau_eff = torch.as_tensor(self.tau, device=self.v.device, dtype=self.v.dtype)
        if self.decay_input:
            return self.v - self.v / (tau_eff + self.tau_eps)
        decay = 1.0 - 1.0 / (tau_eff + self.tau_eps)
        decay = torch.clamp(decay, 0.0, 1.0)
        return self.v * decay

    def _lif_charge(self, x: torch.Tensor, residual_mem: Optional[torch.Tensor] = None) -> torch.Tensor:
        if residual_mem is None:
            residual_mem = self._lif_residual()
        if self.decay_input:
            tau_eff = torch.as_tensor(self.tau, device=self.v.device, dtype=self.v.dtype)
            return residual_mem + x / (tau_eff + self.tau_eps)
        return residual_mem + x

    def _qkv_context(self, mem_t: torch.Tensor) -> torch.Tensor:
        if not self.mem_history:
            return torch.zeros_like(mem_t)

        query = mem_t * self.w_q.to(device=mem_t.device, dtype=mem_t.dtype)
        keys = torch.stack([
            h.to(device=mem_t.device, dtype=mem_t.dtype) * self.w_k.to(device=mem_t.device, dtype=mem_t.dtype)
            for h in self.mem_history
        ], dim=0)
        values = torch.stack([
            h.to(device=mem_t.device, dtype=mem_t.dtype) * self.w_v.to(device=mem_t.device, dtype=mem_t.dtype)
            for h in self.input_history
        ], dim=0)

        scores = query.unsqueeze(0) * keys
        if scores.dim() > 2:
            scores = scores.flatten(start_dim=2).sum(dim=-1)
        scale = float(max(1, query[0].numel() if query.dim() > 1 else query.numel())) ** 0.5
        scores = scores / scale
        attn = torch.softmax(scores, dim=0)
        while attn.dim() < values.dim():
            attn = attn.unsqueeze(-1)
        return (attn * values).sum(dim=0)

    def _append_history(self, mem_t: torch.Tensor, x_t: torch.Tensor):
        if self.qkv_detach_history:
            mem_t = mem_t.detach()
            x_t = x_t.detach()
        self.mem_history.append(mem_t)
        self.input_history.append(x_t)
        if self.qkv_max_history > 0 and len(self.mem_history) > self.qkv_max_history:
            self.mem_history = self.mem_history[-self.qkv_max_history:]
            self.input_history = self.input_history[-self.qkv_max_history:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)

        residual_mem = self._lif_residual()
        lif_mem = self._lif_charge(x_f, residual_mem)
        context = self._qkv_context(residual_mem)
        alpha = self.qkv_alpha.to(device=lif_mem.device, dtype=lif_mem.dtype)
        total_mem = lif_mem + alpha * context

        th_f = torch.as_tensor(self.v_threshold, device=total_mem.device, dtype=total_mem.dtype)
        spike = self._success_fire(total_mem, th_f)

        rs = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = total_mem - rs * th_f
        else:
            v_reset_t = torch.as_tensor(self.v_reset, device=total_mem.device, dtype=total_mem.dtype)
            self.v = torch.where(rs.bool(), v_reset_t, total_mem)

        self._append_history(residual_mem, x_f)
        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class LSLIF3Neuron(LSLIFNeuron):
    """
    LSLIF variant that keeps only the auxiliary history branch.

    Unlike ``LSLIFNeuron``, this neuron does not maintain or fuse the resettable
    primary membrane into the firing decision. The non-reset history state ``n``
    integrates the input with the same LIF leakage as LSLIF and the spike is
    generated only from the weighted, time-normalized history term:

      n_t = decay * n_{t-1} + x_t
      M_t = beta * n_t / step_t^power

    The history state is not reset after spikes, and all existing history
    controls (fixed/learnable weight, per-step weight, learnable power, and
    history_mode) are inherited from ``LSLIFNeuron``.
    """

    def reset(self):
        self.v = None
        self.n = None
        self.has_fired = None
        self.step_count = 0

    def _ensure_state(self, x: torch.Tensor):
        need_init = (
            self.n is None
            or self.n.shape != x.shape
            or self.n.device != x.device
        )
        if need_init:
            self.v = None
            self.n = torch.zeros_like(x, dtype=torch.float32, device=x.device)
            self.has_fired = torch.zeros_like(x, dtype=torch.bool, device=x.device)
            self.step_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)

        tau_eff = torch.as_tensor(self.tau, device=self.n.device, dtype=self.n.dtype)
        if self.decay_input:
            n_t = self.n + (x_f - self.n) / (tau_eff + self.tau_eps)
        else:
            decay = 1.0 - 1.0 / (tau_eff + self.tau_eps)
            decay = torch.clamp(decay, 0.0, 1.0)
            n_t = self.n * decay + x_f

        self.step_count += 1
        step_t = torch.as_tensor(float(self.step_count), device=n_t.device, dtype=n_t.dtype)
        history_power = self._get_history_power(dtype=n_t.dtype, device=n_t.device)
        norm = torch.pow(step_t + self.history_eps, history_power)
        history_weight = self._get_history_weight(dtype=n_t.dtype, device=n_t.device, step_count=self.step_count)
        total_mem = history_weight * (n_t / norm)
        if self.history_mode == 'post_spike':
            total_mem = total_mem * self.has_fired.to(dtype=total_mem.dtype)

        th_f = torch.as_tensor(self.v_threshold, device=self.n.device, dtype=self.n.dtype)
        spike = self._success_fire(total_mem, th_f)

        rs = spike.detach() if self.detach_reset else spike
        self.n = n_t
        self.has_fired = torch.logical_or(self.has_fired, rs.bool())
        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class LSLIF4Neuron(LSLIFNeuron):
    """
    LSLIF variant whose auxiliary branch stores the membrane value lost by the
    resettable primary membrane.

    Compared with ``LSLIFNeuron``, the auxiliary state ``n`` no longer receives
    the input directly and does not leak. Instead, it accumulates every value
    removed from the primary membrane:

      - leakage loss: when ``v`` decays, add the leaked amount to ``n``;
      - soft reset: after a spike, subtract one threshold from the fused
        firing membrane and add that threshold value to ``n``;
      - hard reset: after a spike, reset the primary membrane from the fused
        firing membrane and add that whole fused value to ``n``.

    The firing membrane keeps the LSLIF-style fusion

      M_t = v_t + beta * n_t / step_t^power

    so the loss-history branch is time-normalized, weighted, and added back to
    the primary membrane before thresholding.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)

        tau_eff = torch.as_tensor(self.tau, device=self.v.device, dtype=self.v.dtype)
        inv_tau = 1.0 / (tau_eff + self.tau_eps)
        if self.decay_input:
            leak_loss = self.v * inv_tau
            v_t = self.v + (x_f - self.v) * inv_tau
        else:
            decay = 1.0 - inv_tau
            decay = torch.clamp(decay, 0.0, 1.0)
            leak_loss = self.v * (1.0 - decay)
            v_t = self.v * decay + x_f

        n_t = self.n + leak_loss

        self.step_count += 1
        step_t = torch.as_tensor(float(self.step_count), device=v_t.device, dtype=v_t.dtype)
        history_power = self._get_history_power(dtype=v_t.dtype, device=v_t.device)
        norm = torch.pow(step_t + self.history_eps, history_power)
        history_weight = self._get_history_weight(dtype=v_t.dtype, device=v_t.device, step_count=self.step_count)
        history_term = history_weight * (n_t / norm)
        if self.history_mode == 'post_spike':
            history_term = history_term * self.has_fired.to(dtype=history_term.dtype)
        total_mem = v_t + history_term

        th_f = torch.as_tensor(self.v_threshold, device=self.v.device, dtype=self.v.dtype)
        spike = self._success_fire(total_mem, th_f)

        rs = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            reset_loss = rs * th_f
            self.v = torch.where(rs.bool(), total_mem - reset_loss, v_t)
        else:
            reset_loss = rs * total_mem
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            self.v = torch.where(rs.bool(), v_reset_t, v_t)

        self.n = n_t + reset_loss
        self.has_fired = torch.logical_or(self.has_fired, rs.bool())
        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class LSLIF2Neuron(LSLIFNeuron):
    """
    LSLIF variant with a residual auxiliary membrane.

    The primary membrane integrates the current input and is hard-reset to zero
    after a spike. A separate auxiliary membrane keeps only the residual left by
    the total firing membrane after soft reset. By default, this residual
    auxiliary membrane is added directly to the primary membrane at every time
    step:

      u'_t = decay * u_{t-1}
      M_t = m_t + u'_t
      u_t = u'_t + reset_spike * (M_t - threshold)

    For compatibility, ``lslif2_aux_mode='scaled_avg'`` restores the older
    weighted time-normalized fusion ``M_t = m_t + beta * u'_t / t``. The
    auxiliary membrane leaks with the same ``tau`` as the primary membrane,
    never receives ``x_t`` directly, and never resets. ``history_growth`` is kept
    as a backward-compatible constructor argument but is not used by this
    residual-memory formulation.
    """

    def __init__(self, *args, **kwargs):
        self.history_growth = float(kwargs.pop('history_growth', 1.1))
        self.lslif2_aux_mode = str(kwargs.pop('lslif2_aux_mode', 'direct')).lower()
        if self.lslif2_aux_mode not in {'direct', 'scaled_avg'}:
            raise ValueError(
                f"Unsupported lslif2_aux_mode: {self.lslif2_aux_mode}. "
                "Expected 'direct' or 'scaled_avg'."
            )
        kwargs['history_power'] = 1.0
        kwargs['history_learn_power'] = False
        super().__init__(*args, **kwargs)
        self.history_state = None

    def reset(self):
        super().reset()
        self.history_state = None

    def _ensure_state(self, x: torch.Tensor):
        super()._ensure_state(x)
        need_init = (
            self.history_state is None
            or self.history_state.shape != x.shape
            or self.history_state.device != x.device
        )
        if need_init:
            self.history_state = torch.zeros_like(x, dtype=torch.float32, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)

        tau_eff = torch.as_tensor(self.tau, device=self.v.device, dtype=self.v.dtype)
        decay = 1.0 - 1.0 / (tau_eff + self.tau_eps)
        decay = torch.clamp(decay, 0.0, 1.0)
        if self.decay_input:
            m_t = self.v + (x_f - self.v) / (tau_eff + self.tau_eps)
        else:
            m_t = self.v * decay + x_f

        self.step_count += 1
        residual_mem = self.history_state * decay
        if self.lslif2_aux_mode == 'direct':
            history_term = residual_mem
        else:
            step_t = torch.as_tensor(float(self.step_count), device=m_t.device, dtype=m_t.dtype)
            history_avg = residual_mem / (step_t + self.history_eps)
            history_weight = self._get_history_weight(dtype=m_t.dtype, device=m_t.device, step_count=self.step_count)
            history_term = history_weight * history_avg
        if self.history_mode == 'post_spike':
            history_term = history_term * self.has_fired.to(dtype=history_term.dtype)
        total_mem = m_t + history_term

        th_f = torch.as_tensor(self.v_threshold, device=self.v.device, dtype=self.v.dtype)
        spike = self._success_fire(total_mem, th_f)

        rs = spike.detach() if self.detach_reset else spike
        self.history_state = residual_mem + rs * (total_mem - th_f)
        self.v = m_t * (1.0 - rs)

        self.has_fired = torch.logical_or(self.has_fired, rs.bool())
        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class LSCLIFNeuron(LSLIFNeuron):
    """
    CLIF enhanced with LSLIF-style auxiliary history branch.

    This neuron keeps all history-related interfaces from ``LSLIFNeuron`` while
    adding CLIF's complementary memory state ``m``:
      - history branch: n_t (no spike, no reset) for long-range residual memory
      - complementary memory: m <- m * sigmoid(v / tau) + spike
      - reset: standard LIF reset followed by complementary CLIF reset term
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m = None

    def reset(self):
        super().reset()
        self.m = None

    def _ensure_state(self, x: torch.Tensor):
        super()._ensure_state(x)
        need_init_m = (
            self.m is None
            or self.m.shape != x.shape
            or self.m.device != x.device
        )
        if need_init_m:
            self.m = torch.zeros_like(x, dtype=torch.float32, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)

        tau_eff = torch.as_tensor(self.tau, device=self.v.device, dtype=self.v.dtype)
        if self.decay_input:
            v_t = self.v + (x_f - self.v) / (tau_eff + self.tau_eps)
            n_t = self.n + (x_f - self.n) / (tau_eff + self.tau_eps)
        else:
            decay = 1.0 - 1.0 / (tau_eff + self.tau_eps)
            decay = torch.clamp(decay, 0.0, 1.0)
            v_t = self.v * decay + x_f
            n_t = self.n * decay + x_f

        self.step_count += 1
        step_t = torch.as_tensor(float(self.step_count), device=v_t.device, dtype=v_t.dtype)
        history_power = self._get_history_power(dtype=v_t.dtype, device=v_t.device)
        norm = torch.pow(step_t + self.history_eps, history_power)
        history_weight = self._get_history_weight(dtype=v_t.dtype, device=v_t.device, step_count=self.step_count)
        history_term = history_weight * (n_t / norm)
        if self.history_mode == 'post_spike':
            history_term = history_term * self.has_fired.to(dtype=history_term.dtype)
        total_mem = v_t + history_term

        # CLIF complementary memory forming/strengthening
        self.m = self.m * torch.sigmoid(v_t / (tau_eff + self.tau_eps))
        th_f = torch.as_tensor(self.v_threshold, device=self.v.device, dtype=self.v.dtype)
        spike = self._success_fire(total_mem, th_f)
        self.m = self.m + spike

        rs = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = v_t - rs * th_f
        else:
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            self.v = torch.where(rs.bool(), v_reset_t, v_t)

        # CLIF complementary reset
        self.v = self.v - rs * torch.sigmoid(self.m)

        self.n = n_t
        self.has_fired = torch.logical_or(self.has_fired, rs.bool())
        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class ThresholdLadderLIFNeuron(LSLIFNeuron):
    """
    Threshold-modulated LIF with a single membrane state.

    TLIF keeps one non-reset, non-leaky membrane ``v``. The membrane only
    integrates the input, while the dynamic activity that used to belong to the
    membrane is moved to the threshold ladder:

      V_t = V_{t-1} + I_t
      Theta_t^time = Theta_t + lambda * (Theta_t - B_t)
      s_t = H(V_t - Theta_t^time)

    The threshold rises every step by a ratio of the current interval
    ``Theta_t - B_t``. When the neuron spikes, the threshold ladder then
    advances with the existing LSLIF-style interval adjustment from ``V_t``:

      A_t = beta * V_t / (step_t + eps)^power
      D_t = clamp(theta - A_t, min_interval)
      B_{t+1} = (1 - s_t) * B_t + s_t * Theta_t^time
      Theta_{t+1} = Theta_t^time + s_t * D_t

    Thus the membrane itself is a pure accumulator, and both the per-step
    progression and spike-triggered adaptation are represented by the threshold.
    """

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = None,
        surrogate_function: Optional[Callable] = None,
        detach_reset: bool = False,
        tau_eps: float = 1e-6,
        history_weight: float = 1.0,
        history_power: float = 1.0,
        history_eps: float = 1e-6,
        history_learn_weight: bool = False,
        history_weight_lo: Optional[float] = None,
        history_weight_hi: Optional[float] = None,
        history_weight_per_step: bool = False,
        history_max_steps: int = 16,
        history_learn_power: bool = False,
        history_mode: str = 'all',
        tlif_lambda: float = 0.5,
        tlif_theta: Optional[float] = None,
        tlif_alpha: float = 0.5,
        tlif_w: float = 1.0,
        tlif_b: float = 0.0,
        tlif_min_interval: float = 1e-3,
        layer_index: Optional[int] = None,
        total_layers: Optional[int] = None,
        asn_enable: bool = False,
        asn_p: float = 0.5,
        asn_rho: float = 0.5,
        asn_seed: int = 2022,
        asn_detach_lateral: bool = False,
        **kwargs,
    ):
        super().__init__(
            tau=tau,
            decay_input=decay_input,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            tau_eps=tau_eps,
            history_weight=history_weight,
            history_power=history_power,
            history_eps=history_eps,
            history_learn_weight=history_learn_weight,
            history_weight_lo=history_weight_lo,
            history_weight_hi=history_weight_hi,
            history_weight_per_step=history_weight_per_step,
            history_max_steps=history_max_steps,
            history_learn_power=history_learn_power,
            history_mode=history_mode,
            layer_index=layer_index,
            total_layers=total_layers,
            asn_enable=asn_enable,
            asn_p=asn_p,
            asn_rho=asn_rho,
            asn_seed=asn_seed,
            asn_detach_lateral=asn_detach_lateral,
            **_success_modulation_kwargs(kwargs),
        )
        self.tlif_lambda = float(tlif_lambda)
        if not 0.0 <= self.tlif_lambda <= 1.0:
            raise ValueError('tlif_lambda must be in [0, 1].')
        self.tlif_theta = float(self.v_threshold if tlif_theta is None else tlif_theta)
        if self.tlif_theta <= 0.0:
            raise ValueError('tlif_theta must be positive.')
        self.tlif_min_interval = float(tlif_min_interval)
        if self.tlif_min_interval <= 0.0:
            raise ValueError('tlif_min_interval must be positive.')
        # Kept only so old CLI/checkpoints can pass these names without changing
        # the current formula. The tanh(alpha, w, b) modulation is not used.
        self.tlif_alpha = float(tlif_alpha)
        self.tlif_w = float(tlif_w)
        self.tlif_b = float(tlif_b)
        self.b_base = None
        self.theta = None

    def reset(self):
        self.v = None
        self.b_base = None
        self.theta = None
        self.has_fired = None
        self.step_count = 0

    def _ensure_state(self, x: torch.Tensor):
        need_init = (
            self.v is None
            or self.b_base is None
            or self.theta is None
            or self.has_fired is None
            or self.v.shape != x.shape
            or self.b_base.shape != x.shape
            or self.theta.shape != x.shape
            or self.has_fired.shape != x.shape
            or self.v.device != x.device
            or self.b_base.device != x.device
            or self.theta.device != x.device
            or self.has_fired.device != x.device
        )
        if need_init:
            self.v = torch.zeros_like(x, dtype=torch.float32, device=x.device)
            self.b_base = torch.zeros_like(x, dtype=torch.float32, device=x.device)
            self.theta = torch.full_like(x, self.v_threshold, dtype=torch.float32, device=x.device)
            self.has_fired = torch.zeros_like(x, dtype=torch.bool, device=x.device)
            self.step_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)

        lambda_t = torch.as_tensor(self.tlif_lambda, device=self.v.device, dtype=self.v.dtype)
        v_t = self.v + x_f
        theta_time = self.theta + lambda_t * (self.theta - self.b_base)

        spike = self._success_fire(v_t, theta_time)
        rs = spike.detach() if self.detach_reset else spike
        rs_f = rs.to(dtype=v_t.dtype)

        self.step_count += 1
        step_t = torch.as_tensor(float(self.step_count), device=v_t.device, dtype=v_t.dtype)
        history_power = self._get_history_power(dtype=v_t.dtype, device=v_t.device)
        norm = torch.pow(step_t + self.history_eps, history_power)
        history_weight = self._get_history_weight(dtype=v_t.dtype, device=v_t.device, step_count=self.step_count)
        threshold_drop = history_weight * (v_t / norm)
        if self.history_mode == 'post_spike':
            threshold_drop = threshold_drop * self.has_fired.to(dtype=threshold_drop.dtype)

        theta_step = torch.as_tensor(self.tlif_theta, dtype=v_t.dtype, device=v_t.device)
        min_interval = torch.as_tensor(self.tlif_min_interval, dtype=v_t.dtype, device=v_t.device)
        next_interval = torch.clamp(theta_step - threshold_drop, min=min_interval)

        b_next = (1.0 - rs_f) * self.b_base + rs_f * theta_time
        theta_next = theta_time + rs_f * next_interval

        self.v = v_t
        self.b_base = b_next
        self.theta = theta_next
        self.has_fired = torch.logical_or(self.has_fired, rs.bool())
        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class RCMLIFNeuron(ASNFireMixin, nn.Module):
    """
    Reset-Compensated Memory LIF neuron.

    The main membrane ``v`` performs standard LIF charging, firing, and reset.
    The auxiliary state ``r`` does not integrate the current input. Instead, it
    stores reset-induced membrane loss:

      r_t = lambda_r * r_{t-1} + eta * Delta_t

    The previous reset-loss memory is added before firing:

      M_t = v_t^pre + beta * phi(r_{t-1})

    so RCMLIF preserves a trajectory of reset losses rather than the input
    history stored by LSLIF's ``n`` branch.
    """

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Optional[Callable] = None,
        detach_reset: bool = False,
        tau_eps: float = 1e-6,
        rcm_lambda: float = 0.5,
        rcm_eta: float = 1.0,
        rcm_beta: float = 1.0,
        rcm_learn_eta: bool = False,
        rcm_learn_beta: bool = False,
        rcm_phi: str = 'tanh',
        rcm_power: float = 1.0,
        layer_index: Optional[int] = None,
        asn_enable: bool = False,
        asn_p: float = 0.5,
        asn_rho: float = 0.5,
        asn_seed: int = 2022,
        asn_detach_lateral: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.tau = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.detach_reset = bool(detach_reset)
        self.tau_eps = float(tau_eps)
        self.rcm_lambda = float(rcm_lambda)
        if not 0.0 <= self.rcm_lambda <= 1.0:
            raise ValueError('rcm_lambda must be in [0, 1].')
        self.rcm_eta = float(rcm_eta)
        self.rcm_beta = float(rcm_beta)
        self.rcm_learn_eta = bool(rcm_learn_eta)
        self.rcm_learn_beta = bool(rcm_learn_beta)
        self.rcm_phi = str(rcm_phi).lower()
        if self.rcm_phi not in {'tanh', 'identity', 'time_norm'}:
            raise ValueError("Unsupported rcm_phi. Expected 'tanh', 'identity', or 'time_norm'.")
        self.rcm_power = float(rcm_power)
        self.layer_index = int(layer_index) if layer_index is not None else None
        self.surrogate_function = surrogate_function if surrogate_function is not None else Rectangle()
        self._init_asn(
            asn_enable=asn_enable,
            asn_p=asn_p,
            asn_rho=asn_rho,
            asn_seed=asn_seed,
            asn_detach_lateral=asn_detach_lateral,
            layer_index=self.layer_index,
            **_success_modulation_kwargs(kwargs),
        )

        if self.rcm_learn_eta:
            self.rcm_eta_param = nn.Parameter(torch.tensor(self.rcm_eta, dtype=torch.float32))
        if self.rcm_learn_beta:
            self.rcm_beta_param = nn.Parameter(torch.tensor(self.rcm_beta, dtype=torch.float32))

        self.v = None
        self.r = None
        self.step_count = 0

    def reset(self):
        self.v = None
        self.r = None
        self.step_count = 0

    def _ensure_state(self, x: torch.Tensor):
        need_init = (
            self.v is None
            or self.r is None
            or self.v.shape != x.shape
            or self.r.shape != x.shape
            or self.v.device != x.device
            or self.r.device != x.device
        )
        if need_init:
            self.v = torch.zeros_like(x, dtype=torch.float32, device=x.device)
            self.r = torch.zeros_like(x, dtype=torch.float32, device=x.device)
            self.step_count = 0

    def _get_rcm_eta(self, dtype: torch.dtype, device: torch.device):
        if self.rcm_learn_eta:
            return self.rcm_eta_param.to(dtype=dtype, device=device)
        return torch.as_tensor(self.rcm_eta, dtype=dtype, device=device)

    def _get_rcm_beta(self, dtype: torch.dtype, device: torch.device):
        if self.rcm_learn_beta:
            return self.rcm_beta_param.to(dtype=dtype, device=device)
        return torch.as_tensor(self.rcm_beta, dtype=dtype, device=device)

    def _rcm_transform(self, r: torch.Tensor, step_count: int) -> torch.Tensor:
        if self.rcm_phi == 'tanh':
            return torch.tanh(r)
        if self.rcm_phi == 'identity':
            return r
        step_t = torch.as_tensor(float(step_count), device=r.device, dtype=r.dtype)
        norm = torch.pow(step_t, self.rcm_power)
        return r / norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)

        tau_eff = torch.as_tensor(self.tau, device=self.v.device, dtype=self.v.dtype)
        if self.decay_input:
            m_t = self.v + (x_f - self.v) / (tau_eff + self.tau_eps)
        else:
            decay = 1.0 - 1.0 / (tau_eff + self.tau_eps)
            decay = torch.clamp(decay, 0.0, 1.0)
            m_t = self.v * decay + x_f

        self.step_count += 1
        beta = self._get_rcm_beta(dtype=m_t.dtype, device=m_t.device)
        total_mem = m_t + beta * self._rcm_transform(self.r, self.step_count)

        th_f = torch.as_tensor(self.v_threshold, device=self.v.device, dtype=self.v.dtype)
        spike = self._success_fire(total_mem, th_f)

        rs = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            v_next = m_t - rs * th_f
        else:
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            v_next = torch.where(rs.bool(), v_reset_t, m_t)

        reset_loss = m_t - v_next
        eta = self._get_rcm_eta(dtype=m_t.dtype, device=m_t.device)
        lambda_r = torch.as_tensor(self.rcm_lambda, dtype=m_t.dtype, device=m_t.device)
        self.r = lambda_r * self.r + eta * reset_loss
        self.v = v_next
        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class SCRLIFNeuron(ASNFireMixin, nn.Module):
    """Spike-Cause Reset LIF neuron.

    SCR-LIF keeps the vanilla LIF charge/fire rule but modulates the soft-reset
    amount for each emitted spike according to current-input contribution,
    threshold overshoot, and the pre-spike inter-spike interval state.
    """

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = None,
        surrogate_function: Optional[Callable] = None,
        detach_reset: bool = False,
        tau_eps: float = 1e-6,
        scr_r0: float = 0.8,
        scr_alpha_in: float = 0.2,
        scr_alpha_exc: float = 0.1,
        scr_alpha_isi: float = 0.1,
        scr_r_min: float = 0.7,
        scr_r_max: float = 1.2,
        scr_tau_isi: float = 16.0,
        scr_init_isi: Optional[float] = None,
        asn_enable: bool = False,
        asn_p: float = 0.5,
        asn_rho: float = 0.5,
        asn_seed: int = 2022,
        asn_detach_lateral: bool = False,
        layer_index: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.tau = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        if self.v_reset is not None:
            raise ValueError('SCRLIF uses soft reset and requires v_reset=None.')
        self.detach_reset = bool(detach_reset)
        self.tau_eps = float(tau_eps)
        self.scr_r0 = float(scr_r0)
        self.scr_alpha_in = float(scr_alpha_in)
        self.scr_alpha_exc = float(scr_alpha_exc)
        self.scr_alpha_isi = float(scr_alpha_isi)
        self.scr_r_min = float(scr_r_min)
        self.scr_r_max = float(scr_r_max)
        if self.scr_r_max < self.scr_r_min:
            raise ValueError('scr_r_max must be >= scr_r_min.')
        self.scr_tau_isi = float(scr_tau_isi)
        self.scr_init_isi = float(scr_init_isi) if scr_init_isi is not None else float(scr_tau_isi)
        self.surrogate_function = surrogate_function if surrogate_function is not None else Rectangle()
        self._init_asn(
            asn_enable=asn_enable,
            asn_p=asn_p,
            asn_rho=asn_rho,
            asn_seed=asn_seed,
            asn_detach_lateral=asn_detach_lateral,
            layer_index=layer_index,
            **_success_modulation_kwargs(kwargs),
        )
        self.v = None
        self.d = None

    def reset(self):
        self.v = None
        self.d = None

    def _ensure_state(self, x: torch.Tensor):
        if self.v is None or self.v.shape != x.shape or self.v.device != x.device:
            self.v = torch.zeros_like(x, dtype=torch.float32, device=x.device)
            self.d = torch.full_like(self.v, self.scr_init_isi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)
        tau_eff = torch.as_tensor(self.tau, device=self.v.device, dtype=self.v.dtype)
        if self.decay_input:
            hist = self.v * (1.0 - 1.0 / (tau_eff + self.tau_eps))
            input_term = x_f / (tau_eff + self.tau_eps)
            h_t = hist + input_term
        else:
            decay = 1.0 - 1.0 / (tau_eff + self.tau_eps)
            decay = torch.clamp(decay, 0.0, 1.0)
            hist = self.v * decay
            input_term = x_f
            h_t = hist + input_term

        th_f = torch.as_tensor(self.v_threshold, device=h_t.device, dtype=h_t.dtype)
        spike = self._success_fire(h_t, th_f)

        # The cause terms are only meaningful for neurons that actually spike.
        # Avoid materializing full-size q_in/q_exc/q_isi/r_t tensors on silent
        # steps; when spikes exist, compute them only on the spiking positions.
        rs = spike.detach() if self.detach_reset else spike
        spike_mask = spike.detach().to(dtype=torch.bool)
        if spike_mask.any():
            eps = torch.as_tensor(self.tau_eps, device=h_t.device, dtype=h_t.dtype)
            hist_spike = hist[spike_mask]
            input_spike = input_term[spike_mask]
            h_spike = h_t[spike_mask]
            d_spike = self.d[spike_mask].to(dtype=h_t.dtype)
            q_in_spike = torch.abs(input_spike) / (torch.abs(input_spike) + torch.abs(hist_spike) + eps)
            q_exc_spike = torch.tanh(torch.relu(h_spike - th_f) / (th_f + eps))
            tau_isi = torch.as_tensor(max(self.scr_tau_isi, self.tau_eps), device=h_t.device, dtype=h_t.dtype)
            q_isi_spike = torch.exp(-d_spike / tau_isi)
            r_spike = (self.scr_r0 + self.scr_alpha_in * q_in_spike
                       + self.scr_alpha_exc * q_exc_spike + self.scr_alpha_isi * q_isi_spike)
            r_spike = torch.clamp(r_spike, self.scr_r_min, self.scr_r_max)

            v_next = h_t.clone()
            v_next[spike_mask] = h_spike - rs[spike_mask] * th_f * r_spike
            self.v = v_next
            self.d.add_(1.0)
            self.d[spike_mask] = 0.0
        else:
            self.v = h_t
            self.d.add_(1.0)
        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class SCRLIFV2Neuron(ASNFireMixin, nn.Module):
    """Spike-Cause Reset LIF V2 neuron.

    SCRLIFV2 keeps the vanilla LIF charge/fire rule and applies a
    proportional complementary reset only to neurons that emit one spike. The
    reset compares the decayed history contribution with the current input
    contribution, using only their positive parts for the ratio. For example,
    history=0.9 and input=0.3 gives mix=0.45 and reset membrane=1.2-0.45;
    history=0.3 and input=0.9 gives the same mix and reset membrane=0.45.
    """

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = None,
        surrogate_function: Optional[Callable] = None,
        detach_reset: bool = False,
        tau_eps: float = 1e-6,
        asn_enable: bool = False,
        asn_p: float = 0.5,
        asn_rho: float = 0.5,
        asn_seed: int = 2022,
        asn_detach_lateral: bool = False,
        layer_index: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.tau = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        if self.v_reset is not None:
            raise ValueError('SCRLIFV2 uses soft reset and requires v_reset=None.')
        self.detach_reset = bool(detach_reset)
        self.tau_eps = float(tau_eps)
        self.surrogate_function = surrogate_function if surrogate_function is not None else Rectangle()
        self._init_asn(
            asn_enable=asn_enable,
            asn_p=asn_p,
            asn_rho=asn_rho,
            asn_seed=asn_seed,
            asn_detach_lateral=asn_detach_lateral,
            layer_index=layer_index,
            **_success_modulation_kwargs(kwargs),
        )
        self.v = None

    def reset(self):
        self.v = None

    def _ensure_state(self, x: torch.Tensor):
        if self.v is None or self.v.shape != x.shape or self.v.device != x.device:
            self.v = torch.zeros_like(x, dtype=torch.float32, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)
        tau_eff = torch.as_tensor(self.tau, device=self.v.device, dtype=self.v.dtype)
        if self.decay_input:
            hist = self.v * (1.0 - 1.0 / (tau_eff + self.tau_eps))
            input_term = x_f / (tau_eff + self.tau_eps)
            h_t = hist + input_term
        else:
            decay = 1.0 - 1.0 / (tau_eff + self.tau_eps)
            decay = torch.clamp(decay, 0.0, 1.0)
            hist = self.v * decay
            input_term = x_f
            h_t = hist + input_term

        th_f = torch.as_tensor(self.v_threshold, device=h_t.device, dtype=h_t.dtype)
        spike = self._success_fire(h_t, th_f)

        eps = torch.as_tensor(self.tau_eps, device=h_t.device, dtype=h_t.dtype)
        hist_pos = torch.relu(hist)
        input_pos = torch.relu(input_term)
        positive_total = hist_pos + input_pos
        mix = 2.0 * hist_pos * input_pos / (positive_total + eps)
        v_after_spike = torch.where(hist_pos >= input_pos, h_t - mix, mix)

        rs = spike.detach() if self.detach_reset else spike
        self.v = h_t + rs * (v_after_spike - h_t)
        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)




class VanillaLIFNeuron(ASNFireMixin, LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)
        self._init_asn(
            asn_enable=kwargs.get('asn_enable', False),
            asn_p=kwargs.get('asn_p', 0.5),
            asn_rho=kwargs.get('asn_rho', 0.5),
            asn_seed=kwargs.get('asn_seed', 2022),
            asn_detach_lateral=kwargs.get('asn_detach_lateral', False),
            layer_index=kwargs.get('layer_index', None),
            **_success_modulation_kwargs(kwargs),
        )

    def forward(self, x: torch.Tensor):
        LIFNode_sj.neuronal_charge(self, x)
        th_f = torch.as_tensor(self.v_threshold, device=self.v.device, dtype=self.v.dtype)
        self.last_v_pre = self.v
        spike = self._success_fire(self.v, th_f)
        LIFNode_sj.neuronal_reset(self, spike)
        self._cache_success_spike(spike)
        return spike




class SRLIFNeuron(LIFNode_sj):
    """Synaptic Release LIF with deterministic learnable release threshold.

    The soma LIF dynamics remain identical to vanilla LIF: the pre-reset
    membrane decides the ordinary spike and that ordinary spike triggers reset.
    A deterministic subset of output paths is gated by a second-stage synaptic
    release event, while the remaining paths transmit the ordinary spike just
    like vanilla LIF.
    """

    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False,
                 release_threshold_init: float = 1.0, srlif_release_ratio: float = 0.5, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)
        if release_threshold_init < v_threshold:
            raise ValueError('release_threshold_init must be at least v_threshold.')
        if not 0.0 <= srlif_release_ratio <= 1.0:
            raise ValueError('srlif_release_ratio must be in [0, 1].')
        self.release_threshold = nn.Parameter(torch.tensor(float(release_threshold_init), dtype=torch.float32))
        self.srlif_release_ratio = float(srlif_release_ratio)
        self.last_spike = None
        self.last_release_spike = None
        self.last_release_drive = None
        self.last_release_path_mask = None

    def _get_release_threshold(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.clamp(self.release_threshold, min=float(self.v_threshold)).to(dtype=dtype, device=device)

    def _get_release_path_mask(self, spike: torch.Tensor) -> torch.Tensor:
        """Return a deterministic mask for paths that use the release threshold.

        A mask value of 1 means the path is SRLIF-gated.  A mask value of 0 means
        the path behaves like vanilla LIF and transmits the ordinary spike.
        """
        if self.srlif_release_ratio <= 0.0:
            return torch.zeros_like(spike)
        if self.srlif_release_ratio >= 1.0:
            return torch.ones_like(spike)

        mask_shape = [1] + list(spike.shape[1:]) if spike.dim() > 1 else list(spike.shape)
        num_paths = 1
        for dim in mask_shape:
            num_paths *= int(dim)
        num_gated = int(round(num_paths * self.srlif_release_ratio))
        if num_gated <= 0:
            return torch.zeros(mask_shape, dtype=spike.dtype, device=spike.device).expand_as(spike)
        if num_gated >= num_paths:
            return torch.ones(mask_shape, dtype=spike.dtype, device=spike.device).expand_as(spike)

        flat_mask = torch.zeros(num_paths, dtype=spike.dtype, device=spike.device)
        gated_idx = torch.div(
            torch.arange(num_gated, device=spike.device) * num_paths,
            num_gated,
            rounding_mode='floor',
        )
        flat_mask[gated_idx.long()] = 1.0
        return flat_mask.view(mask_shape).expand_as(spike)

    def forward(self, x: torch.Tensor):
        LIFNode_sj.neuronal_charge(self, x)
        th_f = torch.as_tensor(self.v_threshold, device=self.v.device, dtype=self.v.dtype)
        self.last_v_pre = self.v
        spike = self.surrogate_function(self.v - th_f)
        release_drive = self.v - th_f
        release_threshold = self._get_release_threshold(dtype=self.v.dtype, device=self.v.device)
        release_gate = self.surrogate_function(release_drive - release_threshold)
        release_path_mask = self._get_release_path_mask(spike)
        effective_release_gate = release_path_mask * release_gate + (1.0 - release_path_mask)
        release_spike = spike * effective_release_gate
        LIFNode_sj.neuronal_reset(self, spike)
        self.last_spike = spike
        self.last_release_spike = release_spike
        self.last_release_drive = release_drive
        self.last_release_path_mask = release_path_mask
        return release_spike




class IDISISpikeFunction(torch.autograd.Function):
    """Binary spike with ID-ISI-BP pseudo-gradient over one unrolled sequence.

    The forward value is the ordinary threshold spike.  During backward, each
    postsynaptic spike at time t redistributes its upstream gradient over the
    inter-spike interval [t_last + 1, t] with inverse leak-decay compensation.
    """

    @staticmethod
    def forward(ctx, u_pre: torch.Tensor, spike: torch.Tensor, module: nn.Module, step_idx: int):
        ctx.module = module
        ctx.step_idx = int(step_idx)
        return spike

    @staticmethod
    def backward(ctx, grad_spike: torch.Tensor):
        module = ctx.module
        step_idx = ctx.step_idx
        module._idisi_grad_spike_seq[step_idx] = grad_spike.detach()
        grad_input = module._compute_idisi_grad_for_step(step_idx)
        return grad_input.to(dtype=grad_spike.dtype), None, None, None


class IDISILIFNeuron(VanillaLIFNeuron):
    """Vanilla LIF forward with ID-ISI-BP pseudo-gradient backward.

    Forward dynamics are intentionally kept the same as ``VanillaLIFNeuron``:
    charge, threshold fire, and reset.  Only the backward path through the spike
    output is replaced by the inverse-decay inter-spike-interval credit rule.
    """

    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference, **kwargs)
        self.idisi_max_inverse_decay = float(kwargs.get('idisi_max_inverse_decay', 8.0))
        self.idisi_total_steps = int(kwargs.get('idisi_total_steps', 0) or 0)
        self.idisi_eps = float(kwargs.get('idisi_eps', 1e-6))
        self.idisi_fan_in = int(kwargs.get('idisi_fan_in', 0) or 0)
        self._idisi_u_pre_seq = []
        self._idisi_spike_seq = []
        self._idisi_grad_spike_seq = []

    def reset(self):
        super().reset()
        self._idisi_u_pre_seq = []
        self._idisi_spike_seq = []
        self._idisi_grad_spike_seq = []

    def forward(self, x: torch.Tensor):
        LIFNode_sj.neuronal_charge(self, x)
        u_pre = self.v
        th_f = torch.as_tensor(self.v_threshold, device=u_pre.device, dtype=u_pre.dtype)
        spike = (u_pre >= th_f).to(dtype=x.dtype)
        step_idx = len(self._idisi_u_pre_seq)
        self._idisi_u_pre_seq.append(u_pre.detach())
        self._idisi_spike_seq.append(spike.detach())
        self._idisi_grad_spike_seq.append(None)
        spike_with_idisi_grad = IDISISpikeFunction.apply(u_pre, spike, self, step_idx)
        LIFNode_sj.neuronal_reset(self, spike_with_idisi_grad.detach() if self.detach_reset else spike_with_idisi_grad)
        self._cache_success_spike(spike_with_idisi_grad)
        return spike_with_idisi_grad

    def _lambda_decay(self, device, dtype):
        tau = torch.as_tensor(self.tau, device=device, dtype=dtype)
        return torch.clamp(1.0 - 1.0 / torch.clamp(tau, min=1.0 + self.idisi_eps), min=self.idisi_eps)

    def _compute_idisi_grad_for_step(self, target_step: int):
        if not self._idisi_u_pre_seq:
            return None
        T = len(self._idisi_u_pre_seq)
        device = self._idisi_u_pre_seq[0].device
        dtype = self._idisi_u_pre_seq[0].dtype
        grad_input_seq = [torch.zeros_like(u) for u in self._idisi_u_pre_seq]
        threshold = torch.as_tensor(self.v_threshold, device=device, dtype=dtype).clamp_min(self.idisi_eps)
        lambda_decay = self._lambda_decay(device, dtype)
        max_inv = torch.as_tensor(self.idisi_max_inverse_decay, device=device, dtype=dtype)

        last_spike_time = torch.full_like(self._idisi_spike_seq[0], -1, dtype=torch.long)
        n_in = max(1, self.idisi_fan_in if self.idisi_fan_in > 0 else int(self._idisi_spike_seq[0][0].numel()))

        for t in range(T):
            spike_t = self._idisi_spike_seq[t].to(dtype=torch.bool)
            if not spike_t.any():
                continue
            L = (t - last_spike_time).to(dtype=dtype).clamp_min(1.0)
            base_credit = self._idisi_u_pre_seq[t] / (L * threshold)
            grad_t = self._idisi_grad_spike_seq[t]
            if grad_t is None:
                grad_t = torch.zeros_like(self._idisi_u_pre_seq[t])
            else:
                grad_t = grad_t.to(device=device, dtype=dtype)

            for k in range(T):
                in_window = spike_t & (last_spike_time < k) & (k <= t)
                if not in_window.any():
                    continue
                delta_t = t - k
                inverse_decay = torch.clamp(lambda_decay.pow(-delta_t), max=max_inv)
                credit = grad_t * base_credit * inverse_decay / float(n_in)
                grad_input_seq[k] = grad_input_seq[k] + torch.where(in_window, credit, torch.zeros_like(credit))
            last_spike_time = torch.where(spike_t, torch.full_like(last_spike_time, t), last_spike_time)
        return grad_input_seq[target_step]



class ZELIFNeuron(VanillaLIFNeuron):
    """
    ZELIF = LIF + pattern branch with shared code->parameter lookup.
    The pattern branch is enabled only for 3x3-conv associated activations.
    """

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: float = None,
        surrogate_function: Callable = Rectangle(),
        detach_reset: bool = False,
        cupy_fp32_inference: bool = False,
        zelif_alpha: float = 0.1,
        zelif_kernel_size: int = 3,
        **kwargs,
    ):
        super().__init__(
            tau=tau,
            decay_input=decay_input,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            cupy_fp32_inference=cupy_fp32_inference,
            **kwargs,
        )
        self.zelif_alpha = float(zelif_alpha)
        self.zelif_enabled = 1.0 if int(zelif_kernel_size) == 3 else 0.0
        self.register_buffer('pattern_basis', torch.pow(2, torch.arange(9, dtype=torch.float32)).view(1, 1, 3, 3))
        self.register_buffer('count_basis', torch.ones((1, 1, 3, 3), dtype=torch.float32))
        valid_codes = []
        for code in range(1 << 9):
            spikes = int(bin(code).count('1'))
            if 2 <= spikes <= 3:
                valid_codes.append(code)
        valid_codes_t = torch.tensor(valid_codes, dtype=torch.long)
        self.register_buffer('valid_codes', valid_codes_t)
        code_to_idx = torch.full((1 << 9,), -1, dtype=torch.long)
        code_to_idx[valid_codes_t] = torch.arange(valid_codes_t.numel(), dtype=torch.long)
        self.register_buffer('code_to_idx', code_to_idx)
        self.pattern_params = nn.Parameter(torch.zeros(valid_codes_t.numel(), dtype=torch.float32))
        self._kernel_cache = {}

    def _get_depthwise_kernel(self, base_kernel: torch.Tensor, channels: int, dtype: torch.dtype, device: torch.device):
        key = (id(base_kernel), channels, dtype, device)
        cached = self._kernel_cache.get(key)
        if cached is None:
            cached = base_kernel.to(dtype=dtype, device=device).repeat(channels, 1, 1, 1)
            self._kernel_cache[key] = cached
        return cached

    def _pattern_branch(self, x: torch.Tensor) -> torch.Tensor:
        if self.zelif_enabled == 0.0 or x.dim() != 4:
            return torch.zeros_like(x)
        spikes = (x > 0).to(dtype=x.dtype)
        c = spikes.shape[1]
        count_kernel = self._get_depthwise_kernel(self.count_basis, c, spikes.dtype, spikes.device)
        counts = F.conv2d(spikes, count_kernel, bias=None, stride=1, padding=1, groups=c)
        candidate_mask = (counts == 2) | (counts == 3)
        if not bool(candidate_mask.any()):
            return torch.zeros_like(x)
        code_kernel = self._get_depthwise_kernel(self.pattern_basis, c, spikes.dtype, spikes.device)
        codes = F.conv2d(spikes, code_kernel, bias=None, stride=1, padding=1, groups=c).to(torch.long)
        idx = self.code_to_idx[codes]
        legal = (idx >= 0) & candidate_mask
        idx_safe = idx.clamp_min(0)
        pattern_values = self.pattern_params[idx_safe] * legal.to(dtype=self.pattern_params.dtype)
        return self.zelif_alpha * pattern_values.to(dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        return super().forward(x + self._pattern_branch(x))

class BPTTNeuron(SuccessModulationMixin, nn.Module):
    """
    Baseline LIF with surrogate gradient and membrane state v (fp32).

    Spike-driven dynamic tau with multiplicative (log-domain) step:
      log_tau <- log_tau - eta * (alpha_up)     if spike==0  (more leaky, tau decreases)
      log_tau <- log_tau + eta * (+alpha_down)  if spike==1  (more retentive, tau increases)
      tau = exp(log_tau), and clamp tau in [tau_lo, tau_hi] by clamping log_tau.
    """

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = None,
        surrogate_function: Optional[Callable] = None,
        detach_reset: bool = False,
        tau_mode: str = 'spike',
        tau_lo: Optional[float] = None,
        tau_hi: Optional[float] = None,
        tau_eta: float = 1.0,
        tau_alpha_up: float = 0.02,
        tau_alpha_down: float = 0.02,
        tau_detach_spike: bool = True,
        tau_eps: float = 1e-6,
        tau_learn_alpha: bool = False,
        tau_alpha_share: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._init_success_modulation(**_success_modulation_kwargs(kwargs))
        self.tau0 = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.detach_reset = bool(detach_reset)
        self.surrogate_function = surrogate_function if surrogate_function is not None else Rectangle()

        tm = str(tau_mode).lower().strip()
        assert tm in ('fixed', 'spike')
        self.tau_mode = tm

        if tau_lo is None:
            tau_lo = max(1.0, 0.5 * self.tau0)
        if tau_hi is None:
            tau_hi = 2.0 * self.tau0
        self.tau_lo = float(tau_lo)
        self.tau_hi = float(tau_hi)
        assert self.tau_hi > self.tau_lo >= 1.0

        self.tau_eta = float(tau_eta)
        self.tau_detach_spike = bool(tau_detach_spike)
        self.tau_eps = float(tau_eps)

        self.tau_learn_alpha = bool(tau_learn_alpha)
        self.tau_alpha_share = bool(tau_alpha_share)

        def _inv_softplus(x: float) -> float:
            x_t = torch.tensor(float(x), dtype=torch.float32)
            return float(torch.log(torch.expm1(x_t)).item())

        if self.tau_learn_alpha:
            if self.tau_alpha_share:
                init_raw = _inv_softplus(float(tau_alpha_up))
                self.alpha_raw = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))
            else:
                init_up = _inv_softplus(float(tau_alpha_up))
                init_dn = _inv_softplus(float(tau_alpha_down))
                self.alpha_up_raw = nn.Parameter(torch.tensor(init_up, dtype=torch.float32))
                self.alpha_down_raw = nn.Parameter(torch.tensor(init_dn, dtype=torch.float32))
        else:
            self.tau_alpha_up = float(tau_alpha_up)
            self.tau_alpha_down = float(tau_alpha_down)

        self.v = None
        self.log_tau_state = None

        self._log_tau_lo = float(np.log(self.tau_lo))
        self._log_tau_hi = float(np.log(self.tau_hi))

    def reset(self):
        self.v = None
        self.log_tau_state = None

    def _ensure_state(self, x: torch.Tensor):
        need_init = (
            self.v is None
            or self.v.shape != x.shape
            or self.v.device != x.device
        )
        if need_init:
            self.v = torch.zeros_like(x, dtype=torch.float32, device=x.device)
            log_tau0 = float(np.log(max(self.tau0, self.tau_lo)))
            self.log_tau_state = torch.full_like(self.v, log_tau0)

    def _get_alpha(self, dtype: torch.dtype, device: torch.device):
        if self.tau_learn_alpha:
            if self.tau_alpha_share:
                a = F.softplus(self.alpha_raw).to(dtype=dtype, device=device)
                return a, a
            a_up = F.softplus(self.alpha_up_raw).to(dtype=dtype, device=device)
            a_dn = F.softplus(self.alpha_down_raw).to(dtype=dtype, device=device)
            return a_up, a_dn
        a_up = torch.as_tensor(self.tau_alpha_up, dtype=dtype, device=device)
        a_dn = torch.as_tensor(self.tau_alpha_down, dtype=dtype, device=device)
        return a_up, a_dn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)

        if self.tau_mode == 'fixed':
            tau_eff = torch.as_tensor(self.tau0, device=self.v.device, dtype=self.v.dtype)
        else:
            tau_eff = torch.exp(self.log_tau_state).clamp(min=self.tau_lo, max=self.tau_hi)

        if self.decay_input:
            self.v = self.v + (x_f - self.v) / tau_eff
        else:
            decay = 1.0 - 1.0 / (tau_eff + self.tau_eps)
            decay = torch.clamp(decay, 0.0, 1.0)
            self.v = self.v * decay + x_f

        th_f = torch.as_tensor(self.v_threshold, device=self.v.device, dtype=self.v.dtype)
        spike = self.surrogate_function(self.v + self._success_modulation(self.v) - th_f)

        rs = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - rs * th_f
        else:
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            self.v = torch.where(rs.bool(), v_reset_t, self.v)

        if self.tau_mode == 'spike':
            s = spike.detach() if self.tau_detach_spike else spike
            alpha_up, alpha_down = self._get_alpha(dtype=self.v.dtype, device=self.v.device)
            step = s * (self.tau_eta * alpha_down) - (1.0 - s) * (self.tau_eta * alpha_up)
            self.log_tau_state = (self.log_tau_state + step).clamp(self._log_tau_lo, self._log_tau_hi)

        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class BPTTNeuronTauDependent(BPTTNeuron):
    """
    LIF with spike-driven dynamic tau where the log-tau step depends on current tau.

    Update rule (tau_mode='spike'):
      delta_tau <- spike * alpha_down * tau
                - (1-spike) * alpha_up / tau
      tau <- (1-eta) * tau + eta * (tau + delta_tau)
      log_tau <- log(tau)

    Compared with BPTTNeuron's fixed +/- step in log-domain, this introduces
    tau-dependent step sizes while preserving binary (spike/non-spike) control,
    now with spikes increasing retention and non-spikes increasing leakage.
    """

    def __init__(
        self,
        tau_learn_eta: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tau_learn_eta = bool(tau_learn_eta)

        def _inv_sigmoid(x: float) -> float:
            x_clamped = min(max(float(x), 1e-6), 1.0 - 1e-6)
            x_t = torch.tensor(x_clamped, dtype=torch.float32)
            return float(torch.log(x_t / (1.0 - x_t)).item())

        if self.tau_learn_eta:
            init_eta = _inv_sigmoid(self.tau_eta)
            self.eta_raw = nn.Parameter(torch.tensor(init_eta, dtype=torch.float32))

    def _get_eta(self, dtype: torch.dtype, device: torch.device):
        if self.tau_learn_eta:
            return torch.sigmoid(self.eta_raw).to(dtype=dtype, device=device)
        return torch.as_tensor(self.tau_eta, dtype=dtype, device=device).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)

        if self.tau_mode == 'fixed':
            tau_eff = torch.as_tensor(self.tau0, device=self.v.device, dtype=self.v.dtype)
        else:
            tau_eff = torch.exp(self.log_tau_state).clamp(min=self.tau_lo, max=self.tau_hi)

        if self.decay_input:
            self.v = self.v + (x_f - self.v) / tau_eff
        else:
            decay = 1.0 - 1.0 / (tau_eff + self.tau_eps)
            decay = torch.clamp(decay, 0.0, 1.0)
            self.v = self.v * decay + x_f

        th_f = torch.as_tensor(self.v_threshold, device=self.v.device, dtype=self.v.dtype)
        spike = self.surrogate_function(self.v + self._success_modulation(self.v) - th_f)

        rs = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - rs * th_f
        else:
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            self.v = torch.where(rs.bool(), v_reset_t, self.v)

        if self.tau_mode == 'spike':
            s = spike.detach() if self.tau_detach_spike else spike
            alpha_up, alpha_down = self._get_alpha(dtype=self.v.dtype, device=self.v.device)
            eta = self._get_eta(dtype=self.v.dtype, device=self.v.device)
            tau_safe = tau_eff.clamp(min=self.tau_lo, max=self.tau_hi)
            delta_up = s * (alpha_down * tau_safe)
            delta_down = (1.0 - s) * (alpha_up / (tau_safe + self.tau_eps))
            delta_tau = delta_up - delta_down
            tau_next = (1.0 - eta) * tau_safe + eta * (tau_safe + delta_tau)
            tau_next = tau_next.clamp(min=self.tau_lo, max=self.tau_hi)
            self.log_tau_state = torch.log(tau_next)

        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class DTLIFNeuron(SuccessModulationMixin, nn.Module):
    """
    Dynamic-Tau-Like LIF with direct rho update (refractory-style).

    Design target:
      - low membrane potential -> more retention (rho increases)
      - high membrane potential -> more leakage  (rho decreases)

    We maintain a leakage-rate state ``lambda_state`` and map it to ``rho`` by:
      rho_t = sigmoid(1 - dt * lambda_t)

    Membrane-modulated update:
      gate_t = sigmoid(V_{t-1} / v_threshold)
      lambda_t = lambda_{t-1} - a * (1 - gate_t) + b * gate_t
    """

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = None,
        surrogate_function: Optional[Callable] = None,
        detach_reset: bool = False,
        tau_eps: float = 1e-6,
        dtlif_dt: float = 1.0,
        dtlif_a: float = 0.1,
        dtlif_b: float = 0.1,
        dtlif_learn_a: bool = False,
        dtlif_learn_b: bool = False,
        dtlif_lambda_lo: float = 0.01,
        dtlif_lambda_hi: float = 5.0,
        **kwargs,
    ):
        super().__init__()
        self._init_success_modulation(**_success_modulation_kwargs(kwargs))
        self.tau0 = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.detach_reset = bool(detach_reset)
        self.tau_eps = float(tau_eps)
        self.surrogate_function = surrogate_function if surrogate_function is not None else Rectangle()

        self.dtlif_dt = float(dtlif_dt)
        self.dtlif_a = float(dtlif_a)
        self.dtlif_b = float(dtlif_b)
        self.dtlif_learn_a = bool(dtlif_learn_a)
        self.dtlif_learn_b = bool(dtlif_learn_b)
        self.dtlif_lambda_lo = float(dtlif_lambda_lo)
        self.dtlif_lambda_hi = float(dtlif_lambda_hi)
        if self.dtlif_lambda_hi <= self.dtlif_lambda_lo:
            raise ValueError('dtlif_lambda_hi must be larger than dtlif_lambda_lo.')

        def _inv_softplus(x: float) -> float:
            x_t = torch.tensor(max(float(x), 1e-6), dtype=torch.float32)
            return float(torch.log(torch.expm1(x_t)).item())

        if self.dtlif_learn_a:
            self.a_raw = nn.Parameter(torch.tensor(_inv_softplus(self.dtlif_a), dtype=torch.float32))
        if self.dtlif_learn_b:
            self.b_raw = nn.Parameter(torch.tensor(_inv_softplus(self.dtlif_b), dtype=torch.float32))

        self.v = None
        self.lambda_state = None

    def reset(self):
        self.v = None
        self.lambda_state = None

    def _ensure_state(self, x: torch.Tensor):
        need_init = (
            self.v is None
            or self.v.shape != x.shape
            or self.v.device != x.device
        )
        if need_init:
            self.v = torch.zeros_like(x, dtype=torch.float32, device=x.device)
            lambda0 = 1.0 / max(self.tau0 + self.tau_eps, self.tau_eps)
            self.lambda_state = torch.full_like(self.v, float(lambda0))

    def _get_a(self, dtype: torch.dtype, device: torch.device):
        if self.dtlif_learn_a:
            return F.softplus(self.a_raw).to(dtype=dtype, device=device)
        return torch.as_tensor(self.dtlif_a, dtype=dtype, device=device)

    def _get_b(self, dtype: torch.dtype, device: torch.device):
        if self.dtlif_learn_b:
            return F.softplus(self.b_raw).to(dtype=dtype, device=device)
        return torch.as_tensor(self.dtlif_b, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)

        a = self._get_a(dtype=self.v.dtype, device=self.v.device)
        b = self._get_b(dtype=self.v.dtype, device=self.v.device)
        dt_t = torch.as_tensor(self.dtlif_dt, dtype=self.v.dtype, device=self.v.device)
        one = torch.ones_like(self.v, dtype=self.v.dtype, device=self.v.device)
        gate = torch.sigmoid(self.v / (self.v_threshold + self.tau_eps))
        self.lambda_state = self.lambda_state - a * (one - gate) + b * gate
        self.lambda_state = self.lambda_state.clamp(min=self.dtlif_lambda_lo, max=self.dtlif_lambda_hi)

        rho = torch.sigmoid(one - dt_t * self.lambda_state)

        if self.decay_input:
            self.v = self.v + (x_f - self.v) * (one - rho)
        else:
            self.v = self.v * rho + x_f

        th_f = torch.as_tensor(self.v_threshold, device=self.v.device, dtype=self.v.dtype)
        spike = self.surrogate_function(self.v + self._success_modulation(self.v) - th_f)

        rs = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - rs * th_f
        else:
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            self.v = torch.where(rs.bool(), v_reset_t, self.v)
        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class DGNNeuron(SuccessModulationMixin, nn.Module):
    """
    DGN-style neuron following Eq. (5)-(8) with one-step delayed soft reset.

    Note:
      In this codebase, neuron inputs are already aggregated post-synaptic currents
      from previous layers. Therefore we keep a trace/state with the same shape as
      the incoming tensor and use separate learnable scalars (C, W) for the
      conductance path and current-injection path, respectively.
    """

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = None,
        surrogate_function: Optional[Callable] = None,
        detach_reset: bool = False,
        tau_eps: float = 1e-6,
        dgn_dt: float = 1.0,
        dgn_gl: float = 0.0,
        dgn_c_init: float = 0.01,
        dgn_w_init: float = 0.01,
        dgn_learn_c: bool = True,
        dgn_learn_w: bool = True,
        dgn_phi: str = 'sigmoid',
        **kwargs,
    ):
        super().__init__()
        self._init_success_modulation(**_success_modulation_kwargs(kwargs))
        self.tau_s = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.detach_reset = bool(detach_reset)
        self.tau_eps = float(tau_eps)
        self.surrogate_function = surrogate_function if surrogate_function is not None else Rectangle()

        self.dgn_dt = float(dgn_dt)
        self.dgn_gl = float(dgn_gl)
        self.dgn_phi = str(dgn_phi).lower().strip()
        if self.dgn_phi not in {'sigmoid', 'hard_sigmoid', 'identity'}:
            raise ValueError(f"Unsupported dgn_phi: {dgn_phi}. Expected 'sigmoid', 'hard_sigmoid', or 'identity'.")

        c_init = torch.tensor(float(dgn_c_init), dtype=torch.float32)
        w_init = torch.tensor(float(dgn_w_init), dtype=torch.float32)
        if dgn_learn_c:
            self.C = nn.Parameter(c_init)
        else:
            self.register_buffer('C', c_init)
        if dgn_learn_w:
            self.W = nn.Parameter(w_init)
        else:
            self.register_buffer('W', w_init)

        self.v = None
        self.syn_trace = None
        self.prev_spike = None

    def reset(self):
        self.v = None
        self.syn_trace = None
        self.prev_spike = None

    def _ensure_state(self, x: torch.Tensor):
        need_init = (
            self.v is None
            or self.v.shape != x.shape
            or self.v.device != x.device
        )
        if need_init:
            self.v = torch.zeros_like(x, dtype=torch.float32, device=x.device)
            self.syn_trace = torch.zeros_like(x, dtype=torch.float32, device=x.device)
            self.prev_spike = torch.zeros_like(x, dtype=torch.float32, device=x.device)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.dgn_phi == 'sigmoid':
            return torch.sigmoid(x)
        if self.dgn_phi == 'hard_sigmoid':
            return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)

        dt_t = torch.as_tensor(self.dgn_dt, dtype=self.v.dtype, device=self.v.device)
        alpha = torch.exp(-dt_t / (self.tau_s + self.tau_eps))

        # Eq. (5): synaptic trace update
        self.syn_trace = alpha * self.syn_trace + x_f

        c_t = self.C.to(dtype=self.v.dtype, device=self.v.device)
        w_t = self.W.to(dtype=self.v.dtype, device=self.v.device)
        one = torch.ones_like(self.v, dtype=self.v.dtype, device=self.v.device)

        # Eq. (6): dynamic conductance gate rho_t
        rho_in = one - self.dgn_gl * dt_t - dt_t * (c_t * self.syn_trace)
        rho_t = self._phi(rho_in)

        # Eq. (7): membrane update with one-step delayed soft reset
        th_f = torch.as_tensor(self.v_threshold, device=self.v.device, dtype=self.v.dtype)
        self.v = rho_t * self.v + dt_t * (w_t * self.syn_trace) - th_f * self.prev_spike

        # Eq. (8): spike generation
        spike = self.surrogate_function(self.v + self._success_modulation(self.v) - th_f)

        rs = spike.detach() if self.detach_reset else spike
        self.prev_spike = rs

        if self.v_reset is not None:
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            self.v = torch.where(rs.bool(), v_reset_t, self.v)

        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class LIFDGNNeuron(SuccessModulationMixin, nn.Module):
    """
    LIF with dynamic leak modulation inspired by DGN.

    Dynamics:
      S_t = exp(-dt / tau_trace) * S_{t-1} + Pool(z_{t-1})
      g_t = g0 + c * S_t
      lambda_t = exp(-dt * g_t)
      V_t = lambda_t * V_{t-1} + I_t - theta * z_{t-1}

    Optional nonlinear input branch (DLIF-style):
      I'_t = I_t + gamma_s * (s_t^T K_s s_t) + gamma_t * (s_t^T K_t s_{t-1})
    where K_s/K_t are symmetric with zero diagonal, and
    pairwise terms reduce to element-wise AND for binary spikes.
    """

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = None,
        surrogate_function: Optional[Callable] = None,
        detach_reset: bool = False,
        tau_eps: float = 1e-6,
        lifdgn_dt: float = 1.0,
        lifdgn_tau_trace: float = 2.0,
        lifdgn_g0: float = 0.5,
        lifdgn_c: float = 0.01,
        lifdgn_learn_g0: bool = True,
        lifdgn_learn_c: bool = True,
        lifdgn_g_max: float = 10.0,
        lifdgn_nonlinear_input: bool = False,
        lifdgn_temporal_gamma: float = 0.0,
        lifdgn_detach_prev: bool = False,
        lifdgn_temporal_mode: str = 'linear',
        lifdgn_disable_temporal: bool = False,
        lifdgn_bilinear_chunk_size: int = 2048,
        **kwargs,
    ):
        super().__init__()
        self._init_success_modulation(**_success_modulation_kwargs(kwargs))
        self.tau0 = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.detach_reset = bool(detach_reset)
        self.tau_eps = float(tau_eps)
        self.surrogate_function = surrogate_function if surrogate_function is not None else Rectangle()

        self.lifdgn_dt = float(lifdgn_dt)
        self.lifdgn_tau_trace = float(lifdgn_tau_trace)
        self.lifdgn_g_max = float(max(lifdgn_g_max, 1e-6))
        self.lifdgn_nonlinear_input = bool(lifdgn_nonlinear_input)
        self.lifdgn_detach_prev = bool(lifdgn_detach_prev)
        self.lifdgn_disable_temporal = bool(lifdgn_disable_temporal)
        self.lifdgn_bilinear_chunk_size = int(max(0, lifdgn_bilinear_chunk_size))
        self.lifdgn_temporal_mode = str(lifdgn_temporal_mode).lower()
        if self.lifdgn_temporal_mode not in {'linear', 'event'}:
            raise ValueError(
                f"Unsupported lifdgn_temporal_mode: {lifdgn_temporal_mode}. Expected 'linear' or 'event'."
            )
        self.temporal_gamma = nn.Parameter(torch.tensor(float(lifdgn_temporal_gamma), dtype=torch.float32))

        g0_init = torch.tensor(float(lifdgn_g0), dtype=torch.float32)
        c_init = torch.tensor(float(lifdgn_c), dtype=torch.float32)
        if lifdgn_learn_g0:
            self.g0 = nn.Parameter(g0_init)
        else:
            self.register_buffer('g0', g0_init)
        if lifdgn_learn_c:
            self.c = nn.Parameter(c_init)
        else:
            self.register_buffer('c', c_init)

        self.v = None
        self.syn_trace = None
        self.prev_spike = None
        self.prev_input = None
        self.weight = None
        self.weight_temporal = None
        self.register_buffer('mask_spatial', torch.empty(0), persistent=False)
        self.register_buffer('mask_temporal', torch.empty(0), persistent=False)

    def reset(self):
        self.v = None
        self.syn_trace = None
        self.prev_spike = None
        self.prev_input = None

    def _ensure_state(self, x: torch.Tensor):
        need_init = (
            self.v is None
            or self.v.shape != x.shape
            or self.v.device != x.device
        )
        if need_init:
            self.v = torch.zeros_like(x, dtype=torch.float32, device=x.device)
            self.syn_trace = torch.zeros_like(x, dtype=torch.float32, device=x.device)
            self.prev_spike = torch.zeros_like(x, dtype=torch.float32, device=x.device)

    def _pool_activity(self, z: torch.Tensor):
        if z.dim() <= 1:
            return z
        reduce_dims = tuple(range(1, z.dim()))
        pooled = z.mean(dim=reduce_dims, keepdim=True)
        return pooled.expand_as(z)

    def _ensure_nonlinear_params(self, x: torch.Tensor):
        if not self.lifdgn_nonlinear_input or x.dim() != 4:
            return
        channels = int(x.shape[1])
        need_init = self.weight is None or self.weight.shape[0] != channels
        if need_init:
            w = torch.zeros((channels, channels, channels), dtype=torch.float32, device=x.device)
            wt = torch.zeros((channels, channels, channels), dtype=torch.float32, device=x.device)
            self.weight = nn.Parameter(w)
            self.weight_temporal = nn.Parameter(wt)

            mask = torch.ones((channels, channels), dtype=torch.float32, device=x.device)
            mask.fill_diagonal_(0.0)
            mask = ((mask + mask.t()) > 0).to(dtype=torch.float32)
            self.mask_spatial = mask.unsqueeze(0).expand(channels, -1, -1).clone()
            self.mask_temporal = mask.unsqueeze(0).expand(channels, -1, -1).clone()

    def _outer_linear(self, x_a: torch.Tensor, x_b: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor):
        if x_a.dim() != 4:
            return torch.zeros_like(x_a)
        bsz, channels, height, width = x_a.shape
        x1 = x_a.permute(0, 2, 3, 1).reshape(-1, channels)
        x2 = x_b.permute(0, 2, 3, 1).reshape(-1, channels)
        masked_weight = (weight * mask).reshape(channels, -1)
        positions = x1.shape[0]
        chunk_size = self.lifdgn_bilinear_chunk_size
        if chunk_size <= 0 or positions <= chunk_size:
            qinput = torch.bmm(x1.unsqueeze(-1), x2.unsqueeze(-2)).reshape(-1, channels * channels)
            y_flat = F.linear(qinput, masked_weight)
        else:
            y_flat = x1.new_empty((positions, channels))
            for start in range(0, positions, chunk_size):
                end = min(start + chunk_size, positions)
                q_chunk = torch.bmm(x1[start:end].unsqueeze(-1), x2[start:end].unsqueeze(-2)).reshape(
                    -1, channels * channels
                )
                y_flat[start:end] = F.linear(q_chunk, masked_weight)
        y = y_flat.reshape(bsz, height, width, channels).permute(0, 3, 1, 2)
        return y

    def _nonlinear_input(self, x: torch.Tensor):
        if not self.lifdgn_nonlinear_input or x.dim() != 4:
            return x
        self._ensure_nonlinear_params(x)
        prev = torch.zeros_like(x) if self.prev_input is None else self.prev_input
        if self.lifdgn_detach_prev:
            prev = prev.detach()

        y_spatial = self._outer_linear(x, x, self.weight, self.mask_spatial)
        y = x + y_spatial
        if not self.lifdgn_disable_temporal:
            y_temporal = self._outer_linear(x, prev, self.weight_temporal, self.mask_temporal)
            if self.lifdgn_temporal_mode == 'event':
                y_temporal = torch.tanh(y_temporal)
            y = y + self.temporal_gamma.to(dtype=x.dtype, device=x.device) * y_temporal
        self.prev_input = x
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = self._nonlinear_input(x.to(torch.float32))

        dt_t = torch.as_tensor(self.lifdgn_dt, dtype=self.v.dtype, device=self.v.device)
        alpha = torch.exp(-dt_t / (self.lifdgn_tau_trace + self.tau_eps))
        pooled_activity = self._pool_activity(self.prev_spike)
        self.syn_trace = alpha * self.syn_trace + pooled_activity

        g0_t = self.g0.to(dtype=self.v.dtype, device=self.v.device)
        c_t = self.c.to(dtype=self.v.dtype, device=self.v.device)
        g_t = (g0_t + c_t * self.syn_trace).clamp(min=0.0, max=self.lifdgn_g_max)
        lambda_t = torch.exp(-dt_t * g_t).clamp(min=0.0, max=1.0)

        th_f = torch.as_tensor(self.v_threshold, device=self.v.device, dtype=self.v.dtype)
        self.v = lambda_t * self.v + x_f - th_f * self.prev_spike

        spike = self.surrogate_function(self.v + self._success_modulation(self.v) - th_f)
        rs = spike.detach() if self.detach_reset else spike
        self.prev_spike = rs

        if self.v_reset is not None:
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            self.v = torch.where(rs.bool(), v_reset_t, self.v)

        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class LIFDGN2Neuron(LIFDGNNeuron):
    """
    Variant of LIFDGN with:
      1) input-driven trace (uses I_t instead of Pool(z_{t-1}))
      2) update order in one step:
         receive input -> leak -> spike -> same-step soft reset

    Dynamics:
      S_t = exp(-dt / tau_trace) * S_{t-1} + I_t
      g_t = g0 + c * S_t
      lambda_t = sigmoid(1 - g_t)
      U_t = lambda_t * (V_{t-1} + I_t)
      z_t = Theta(U_t - theta)
      V_t = U_t - theta * z_t
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = self._nonlinear_input(x.to(torch.float32))

        dt_t = torch.as_tensor(self.lifdgn_dt, dtype=self.v.dtype, device=self.v.device)
        alpha = torch.exp(-dt_t / (self.lifdgn_tau_trace + self.tau_eps))

        # input-driven trace at neuron level:
        # x_f is already post-synaptic aggregated current (weighted sum) for each neuron
        self.syn_trace = alpha * self.syn_trace + x_f

        g0_t = self.g0.to(dtype=self.v.dtype, device=self.v.device)
        c_t = self.c.to(dtype=self.v.dtype, device=self.v.device)
        g_t = (g0_t + c_t * self.syn_trace).clamp(min=0.0, max=self.lifdgn_g_max)
        one = torch.ones_like(g_t, dtype=self.v.dtype, device=self.v.device)
        lambda_t = torch.sigmoid(one - g_t)

        # receive input then leak
        self.v = lambda_t * (self.v + x_f)
        th_f = torch.as_tensor(self.v_threshold, device=self.v.device, dtype=self.v.dtype)

        spike = self.surrogate_function(self.v + self._success_modulation(self.v) - th_f)
        rs = spike.detach() if self.detach_reset else spike

        # same-step reset
        self.v = self.v - th_f * rs
        self.prev_spike = rs

        if self.v_reset is not None:
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            self.v = torch.where(rs.bool(), v_reset_t, self.v)

        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class LIFDGN3Neuron(LIFDGNNeuron):
    """
    Variant of LIFDGN2 with direct-input rho computation and no trace dynamics.

    Differences from LIFDGN2:
      1) remove D/synaptic-trace recursion entirely
      2) directly use current input I_t to compute rho
      3) membrane update uses low-pass mixing:
           U_t = rho_t * V_{t-1} + (1 - rho_t) * I_t
         followed by spike + same-step soft reset.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = self._nonlinear_input(x.to(torch.float32))

        g0_t = self.g0.to(dtype=self.v.dtype, device=self.v.device)
        c_t = self.c.to(dtype=self.v.dtype, device=self.v.device)
        g_t = (g0_t + c_t * x_f).clamp(min=0.0, max=self.lifdgn_g_max)
        one = torch.ones_like(g_t, dtype=self.v.dtype, device=self.v.device)
        rho_t = torch.sigmoid(one - g_t)

        self.v = rho_t * self.v + (one - rho_t) * x_f
        th_f = torch.as_tensor(self.v_threshold, device=self.v.device, dtype=self.v.dtype)

        spike = self.surrogate_function(self.v + self._success_modulation(self.v) - th_f)
        rs = spike.detach() if self.detach_reset else spike

        self.v = self.v - th_f * rs
        self.prev_spike = rs

        if self.v_reset is not None:
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            self.v = torch.where(rs.bool(), v_reset_t, self.v)

        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class NewCLIFNeuron(BPTTNeuronTauDependent):
    """
    CLIF + tau-dependent dynamic tau (newLIFTauDep-style).

    - CLIF complementary memory update is kept: m <- m * sigmoid(v / tau) + spike
    - tau update uses tau-dependent delta + eta interpolation in tau-domain:
        delta_tau <- spike * alpha_down * tau - (1-spike) * alpha_up / tau
        tau <- (1-eta) * tau + eta * (tau + delta_tau)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m = None

    def reset(self):
        super().reset()
        self.m = None

    def _ensure_state(self, x: torch.Tensor):
        super()._ensure_state(x)
        need_init_m = (
            self.m is None
            or self.m.shape != x.shape
            or self.m.device != x.device
        )
        if need_init_m:
            self.m = torch.zeros_like(x, dtype=torch.float32, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)

        if self.tau_mode == 'fixed':
            tau_eff = torch.as_tensor(self.tau0, device=self.v.device, dtype=self.v.dtype)
        else:
            tau_eff = torch.exp(self.log_tau_state).clamp(min=self.tau_lo, max=self.tau_hi)

        if self.decay_input:
            x_f = x_f / (tau_eff + self.tau_eps)

        decay = 1.0 - 1.0 / (tau_eff + self.tau_eps)
        decay = torch.clamp(decay, 0.0, 1.0)

        if self.v_reset is None or self.v_reset == 0:
            self.v = self.v * decay + x_f
        else:
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            self.v = self.v * decay + v_reset_t / (tau_eff + self.tau_eps) + x_f

        self.m = self.m * torch.sigmoid(self.v / (tau_eff + self.tau_eps))

        th_f = torch.as_tensor(self.v_threshold, device=self.v.device, dtype=self.v.dtype)
        spike = self.surrogate_function(self.v + self._success_modulation(self.v) - th_f)

        self.m = self.m + spike

        rs = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - rs * th_f
        else:
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            self.v = torch.where(rs.bool(), v_reset_t, self.v)

        self.v = self.v - rs * torch.sigmoid(self.m)

        if self.tau_mode == 'spike':
            s = spike.detach() if self.tau_detach_spike else spike
            alpha_up, alpha_down = self._get_alpha(dtype=self.v.dtype, device=self.v.device)
            eta = self._get_eta(dtype=self.v.dtype, device=self.v.device)
            tau_safe = tau_eff.clamp(min=self.tau_lo, max=self.tau_hi)
            delta_up = s * (alpha_down * tau_safe)
            delta_down = (1.0 - s) * (alpha_up / (tau_safe + self.tau_eps))
            delta_tau = delta_up - delta_down
            tau_next = (1.0 - eta) * tau_safe + eta * (tau_safe + delta_tau)
            tau_next = tau_next.clamp(min=self.tau_lo, max=self.tau_hi)
            self.log_tau_state = torch.log(tau_next)

        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class LSPLIFNeuron(LSLIFNeuron):
    """
    PLIF + LSLIF history branch.

    - Main membrane uses learnable PLIF-style time constant tau=softplus(w).
    - History branch n follows the same tau (shared with main branch).
    """

    def __init__(self, init_tau: Optional[float] = None, **kwargs):
        tau = float(kwargs.get('tau', 2.0) if init_tau is None else init_tau)
        super().__init__(**kwargs)
        inv_sp = float(np.log(np.exp(max(tau, 1e-4)) - 1.0))
        self.w = nn.Parameter(torch.tensor(inv_sp, dtype=torch.float32))

    def _tau_eff(self, dtype: torch.dtype, device: torch.device):
        # shared tau between PLIF main branch and LS history branch
        tau_eff = F.softplus(self.w)
        return tau_eff.to(dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)

        tau_eff = self._tau_eff(dtype=self.v.dtype, device=self.v.device)
        if self.decay_input:
            m_t = self.v + (x_f - self.v) / (tau_eff + self.tau_eps)
            n_t = self.n + (x_f - self.n) / (tau_eff + self.tau_eps)
        else:
            decay = 1.0 - 1.0 / (tau_eff + self.tau_eps)
            decay = torch.clamp(decay, 0.0, 1.0)
            m_t = self.v * decay + x_f
            n_t = self.n * decay + x_f

        self.step_count += 1
        step_t = torch.as_tensor(float(self.step_count), device=m_t.device, dtype=m_t.dtype)
        history_power = self._get_history_power(dtype=m_t.dtype, device=m_t.device)
        norm = torch.pow(step_t + self.history_eps, history_power)
        history_weight = self._get_history_weight(dtype=m_t.dtype, device=m_t.device, step_count=self.step_count)
        history_term = history_weight * (n_t / norm)
        if self.history_mode == 'post_spike':
            history_term = history_term * self.has_fired.to(dtype=history_term.dtype)
        total_mem = m_t + history_term

        th_f = torch.as_tensor(self.v_threshold, device=self.v.device, dtype=self.v.dtype)
        spike = self._success_fire(total_mem, th_f)

        rs = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = m_t - rs * th_f
        else:
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            self.v = torch.where(rs.bool(), v_reset_t, m_t)

        self.n = n_t
        self.has_fired = torch.logical_or(self.has_fired, rs.bool())
        self._cache_success_spike(spike)
        return spike.to(dtype=x.dtype)


class PLIFNeuron(PLIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = None,
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)


if __name__ == '__main__':
    T = 8
    x_input = torch.rand((T, 3, 32, 32)) * 1.2
    clif = ComplementaryLIFNeuron()
    clif_m = MultiStepCLIFNeuron()

    s_list = []
    for t in range(T):
        s = clif(x_input[t])
        s_list.append(s)

    s_list = torch.stack(s_list, dim=0)
    s_output = clif_m(x_input)

    print(s_list.mean())
    print(s_output.mean())
    assert torch.sum(s_output - torch.Tensor(s_list)) == 0
