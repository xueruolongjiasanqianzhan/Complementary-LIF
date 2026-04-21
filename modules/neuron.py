from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import LIFNode as LIFNode_sj
from spikingjelly.clock_driven.neuron import ParametricLIFNode as PLIFNode_sj
from torch import nn

from modules.surrogate import Rectangle


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
class ComplementaryLIFNeuron(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)
        self.register_memory('m', 0.)  # Complementary memory

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)  # LIF charging
        self.m = self.m * torch.sigmoid(self.v / self.tau)  # Forming
        spike = self.neuronal_fire()  # LIF fire
        self.m += spike  # Strengthen
        self.neuronal_reset(spike)  # LIF reset
        self.v = self.v - spike * torch.sigmoid(self.m)  # Reset
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


class LSLIFNeuron(nn.Module):
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
        history_weight_lo: float = -0.8,
        history_weight_hi: float = 0.8,
        history_weight_per_step: bool = False,
        history_max_steps: int = 16,
        history_learn_power: bool = False,
        history_mode: str = 'all',
        layer_index: Optional[int] = None,
        total_layers: Optional[int] = None,
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
        self.history_weight_lo = float(history_weight_lo)
        self.history_weight_hi = float(history_weight_hi)
        if self.history_weight_hi <= self.history_weight_lo:
            raise ValueError('history_weight_hi must be larger than history_weight_lo.')
        self.history_power_lo = 0.0
        self.history_power_hi = 1.0
        self.surrogate_function = surrogate_function if surrogate_function is not None else Rectangle()

        def _inv_sigmoid(x: float) -> float:
            x_t = torch.tensor(float(x), dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
            return float(torch.log(x_t / (1.0 - x_t)).item())

        if self.history_learn_weight:
            if self.history_weight_per_step:
                init_weight = float(np.clip(self.history_weight, self.history_weight_lo, self.history_weight_hi))
                scale = self.history_weight_hi - self.history_weight_lo
                init_unit = (init_weight - self.history_weight_lo) / max(scale, 1e-6)
                init_raw = _inv_sigmoid(init_unit)
                init_tensor = torch.full((self.history_max_steps,), init_raw, dtype=torch.float32)
                self.history_weight_raw = nn.Parameter(init_tensor)
            else:
                init_weight = float(np.clip(self.history_weight, self.history_weight_lo, self.history_weight_hi))
                scale = self.history_weight_hi - self.history_weight_lo
                init_unit = (init_weight - self.history_weight_lo) / max(scale, 1e-6)
                init_raw = _inv_sigmoid(init_unit)
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

    def reset(self):
        self.v = None
        self.n = None
        self.has_fired = None
        self.step_count = 0

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
            weight_unit = torch.sigmoid(weight_raw)
            weight = self.history_weight_lo + (self.history_weight_hi - self.history_weight_lo) * weight_unit
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
        total_mem = m_t + history_term

        th_f = torch.as_tensor(self.v_threshold, device=self.v.device, dtype=self.v.dtype)
        spike = self.surrogate_function(total_mem - th_f)

        rs = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = m_t - rs * th_f
        else:
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            self.v = torch.where(rs.bool(), v_reset_t, m_t)

        self.n = n_t
        self.has_fired = torch.logical_or(self.has_fired, rs.bool())
        return spike.to(dtype=x.dtype)


class VanillaLIFNeuron(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)

class BPTTNeuron(nn.Module):
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
        spike = self.surrogate_function(self.v - th_f)

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
        spike = self.surrogate_function(self.v - th_f)

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

        return spike.to(dtype=x.dtype)


class DTLIFNeuron(nn.Module):
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
        spike = self.surrogate_function(self.v - th_f)

        rs = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - rs * th_f
        else:
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            self.v = torch.where(rs.bool(), v_reset_t, self.v)
        return spike.to(dtype=x.dtype)


class DGNNeuron(nn.Module):
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
        spike = self.surrogate_function(self.v - th_f)

        rs = spike.detach() if self.detach_reset else spike
        self.prev_spike = rs

        if self.v_reset is not None:
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            self.v = torch.where(rs.bool(), v_reset_t, self.v)

        return spike.to(dtype=x.dtype)


class LIFDGNNeuron(nn.Module):
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

        spike = self.surrogate_function(self.v - th_f)
        rs = spike.detach() if self.detach_reset else spike
        self.prev_spike = rs

        if self.v_reset is not None:
            v_reset_t = torch.as_tensor(self.v_reset, device=self.v.device, dtype=self.v.dtype)
            self.v = torch.where(rs.bool(), v_reset_t, self.v)

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
        spike = self.surrogate_function(self.v - th_f)

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
