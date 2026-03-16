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




class VanillaLIFNeuron(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)

class BPTTNeuron(nn.Module):
    """
    Baseline LIF with surrogate gradient and membrane state v (fp32).

    Spike-driven dynamic tau with multiplicative (log-domain) step:
      log_tau <- log_tau + eta * (+alpha_up)     if spike==0  (more sensitive, tau increases)
      log_tau <- log_tau - eta * (alpha_down)   if spike==1  (more suppressive, tau decreases)
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
            step = (1.0 - s) * (self.tau_eta * alpha_up) - s * (self.tau_eta * alpha_down)
            self.log_tau_state = (self.log_tau_state + step).clamp(self._log_tau_lo, self._log_tau_hi)

        return spike.to(dtype=x.dtype)


class BPTTNeuronTauDependent(BPTTNeuron):
    """
    LIF with spike-driven dynamic tau where the log-tau step depends on current tau.

    Update rule (tau_mode='spike'):
      delta_tau <- (1-spike) * alpha_up * tau
                - spike * alpha_down / tau
      tau <- (1-eta) * tau + eta * (tau + delta_tau)
      log_tau <- log(tau)

    Compared with BPTTNeuron's fixed +/- step in log-domain, this introduces
    tau-dependent step sizes while preserving binary (spike/non-spike) control.
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
            delta_up = (1.0 - s) * (alpha_up * tau_safe)
            delta_down = s * (alpha_down / (tau_safe + self.tau_eps))
            delta_tau = delta_up - delta_down
            tau_next = (1.0 - eta) * tau_safe + eta * (tau_safe + delta_tau)
            tau_next = tau_next.clamp(min=self.tau_lo, max=self.tau_hi)
            self.log_tau_state = torch.log(tau_next)

        return spike.to(dtype=x.dtype)


class NewCLIFNeuron(BPTTNeuronTauDependent):
    """
    CLIF + tau-dependent dynamic tau (newLIFTauDep-style).

    - CLIF complementary memory update is kept: m <- m * sigmoid(v / tau) + spike
    - tau update uses tau-dependent delta + eta interpolation in tau-domain:
        delta_tau <- (1-spike) * alpha_up * tau - spike * alpha_down / tau
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
            step_up = (1.0 - s) * (eta * alpha_up * tau_safe)
            step_down = s * (eta * alpha_down / (tau_safe + self.tau_eps))
            step = step_up - step_down
            self.log_tau_state = (self.log_tau_state + step).clamp(self._log_tau_lo, self._log_tau_hi)

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
