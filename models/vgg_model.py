import torch
import torch.nn.functional as F
from spikingjelly.clock_driven import layer

__all__ = [
    'vggsnn', 'snn5', 'snn5_noAP', 'dgn_dvscifar10_tiny', 'dvscifar10_fc2'
]

from torch import nn


def _build_neuron(neuron: callable, kwargs: dict, zelif_kernel_size: int = 3):
    neuron_kwargs = dict(kwargs)
    counter = neuron_kwargs.get('_layer_counter')
    needs_layer_index = neuron_kwargs.get('history_mode', 'all') == 'half' or neuron_kwargs.get('asn_enable', False)
    if needs_layer_index and isinstance(counter, dict):
        idx = int(counter.get('i', 0))
        total = int(max(1, counter.get('total', 1)))
        neuron_kwargs['layer_index'] = idx
        neuron_kwargs['total_layers'] = total
        counter['i'] = idx + 1
    neuron_kwargs.pop('_layer_counter', None)
    if getattr(neuron, '__name__', '') == 'ZELIFNeuron':
        neuron_kwargs['zelif_kernel_size'] = int(zelif_kernel_size)
    return neuron(**neuron_kwargs)


class SynapticReleaseLinear(nn.Linear):
    """Linear layer with learnable release thresholds for simple SR experiments.

    If ``synaptic_release_groups`` is positive, input-output synapses are mapped
    to deterministic random groups and each group shares one learnable release
    threshold.  The release event depends on presynaptic pre-reset membrane and
    the corresponding threshold.  The release threshold is clamped to at least
    the soma firing threshold, so release requires a membrane level that would
    also trigger a soma spike.
    """

    def __init__(
        self, *args, release_threshold_init=1.0, surrogate_function=None,
        synaptic_release_groups=0, release_threshold_min=1.0,
        synaptic_release_fixed_threshold_ratio=0.5,
        synaptic_release_group_seed=2022, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.release_threshold_min = float(release_threshold_min)
        if release_threshold_init < self.release_threshold_min:
            raise ValueError('release_threshold_init must be at least release_threshold_min.')
        if synaptic_release_groups < 0:
            raise ValueError('synaptic_release_groups must be non-negative.')
        if not 0.0 <= synaptic_release_fixed_threshold_ratio <= 1.0:
            raise ValueError('synaptic_release_fixed_threshold_ratio must be in [0, 1].')
        self.synaptic_release_groups = int(synaptic_release_groups)
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(synaptic_release_group_seed))
        num_synapses = self.weight.numel()
        fixed_count = int(round(num_synapses * float(synaptic_release_fixed_threshold_ratio)))
        fixed_mask_flat = torch.zeros(num_synapses, dtype=torch.bool)
        if fixed_count > 0:
            fixed_idx = torch.randperm(num_synapses, generator=generator)[:fixed_count]
            fixed_mask_flat[fixed_idx] = True
        fixed_mask = fixed_mask_flat.view_as(self.weight)
        learnable_mask = ~fixed_mask
        self.register_buffer('release_fixed_mask', fixed_mask)
        if self.synaptic_release_groups > 0:
            self.release_threshold = nn.Parameter(torch.full(
                (self.synaptic_release_groups,), float(release_threshold_init),
                dtype=self.weight.dtype, device=self.weight.device))
            group_index = torch.randint(
                self.synaptic_release_groups, self.weight.shape,
                generator=generator, dtype=torch.long)
            self.register_buffer('release_group_index', group_index)
            self.register_buffer('release_learnable_mask', None)
        elif fixed_count > 0:
            learnable_count = int(learnable_mask.sum().item())
            self.release_threshold = nn.Parameter(torch.full(
                (learnable_count,), float(release_threshold_init),
                dtype=self.weight.dtype, device=self.weight.device))
            self.register_buffer('release_group_index', None)
            self.register_buffer('release_learnable_mask', learnable_mask)
        else:
            self.release_threshold = nn.Parameter(torch.full_like(self.weight, float(release_threshold_init)))
            self.register_buffer('release_group_index', None)
            self.register_buffer('release_learnable_mask', None)
        self.surrogate_function = surrogate_function
        self.last_release_gate_mean = None

    def _get_release_threshold(self, dtype, device):
        fixed_threshold = torch.full_like(self.weight, self.release_threshold_min)
        release_threshold = torch.clamp(self.release_threshold, min=self.release_threshold_min)
        if self.release_group_index is not None:
            release_threshold = release_threshold[self.release_group_index]
        elif self.release_learnable_mask is not None:
            full_threshold = fixed_threshold.clone()
            full_threshold[self.release_learnable_mask] = release_threshold
            release_threshold = full_threshold
        if self.release_fixed_mask.any():
            release_threshold = torch.where(self.release_fixed_mask, fixed_threshold, release_threshold)
        return release_threshold.to(dtype=dtype, device=device)

    def forward(self, x, release_source=None):
        if release_source is None:
            return F.linear(x, self.weight, self.bias)
        if release_source.shape != x.shape:
            raise ValueError('release_source must have the same shape as the presynaptic input tensor.')
        threshold = self._get_release_threshold(dtype=x.dtype, device=x.device)
        release_arg = release_source.unsqueeze(1) - threshold.unsqueeze(0)
        if self.surrogate_function is None:
            release_gate = (release_arg >= 0.0).to(dtype=x.dtype)
        else:
            release_gate = self.surrogate_function(release_arg)
        out = (release_gate * self.weight.unsqueeze(0)).sum(dim=2)
        if self.bias is not None:
            out = out + self.bias
        self.last_release_gate_mean = release_gate.detach().mean()
        return out


class DVSCIFAR10FC2(nn.Module):
    """Two-hidden-layer fully connected SNN for lightweight SR validation."""

    def __init__(self, neuron, num_classes=10, neuron_dropout=0.0, c_in=2, fc_hw=48, hidden_dim=1024, **kwargs):
        super().__init__()
        kwargs = dict(kwargs)
        kwargs['_layer_counter'] = {'i': 0, 'total': 2}
        input_dim = int(c_in) * int(fc_hw or 48) * int(fc_hw or 48)
        self.synaptic_release_enable = bool(kwargs.get('synaptic_release_enable', False))
        linear2_cls = SynapticReleaseLinear if self.synaptic_release_enable else nn.Linear

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.neuron1 = _build_neuron(neuron, kwargs)
        self.drop1 = layer.Dropout(neuron_dropout)
        if self.synaptic_release_enable:
            self.fc2 = linear2_cls(
                hidden_dim, hidden_dim,
                release_threshold_init=kwargs.get('release_threshold_init', 1.0),
                surrogate_function=kwargs.get('surrogate_function', None),
                synaptic_release_groups=kwargs.get('synaptic_release_groups', 0),
                release_threshold_min=kwargs.get('v_threshold', 1.0),
                synaptic_release_fixed_threshold_ratio=kwargs.get('synaptic_release_fixed_threshold_ratio', 0.5),
                synaptic_release_group_seed=kwargs.get('synaptic_release_group_seed', 2022),
            )
        else:
            self.fc2 = linear2_cls(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.neuron2 = _build_neuron(neuron, kwargs)
        self.drop2 = layer.Dropout(neuron_dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.neuron1(x)
        release_source = getattr(self.neuron1, 'last_v_pre', None)
        x = self.drop1(x)
        if self.synaptic_release_enable:
            x = self.fc2(x, release_source)
        else:
            x = self.fc2(x)
        x = self.bn2(x)
        x = self.neuron2(x)
        x = self.drop2(x)
        return self.classifier(x)


class SNN5(nn.Module):
    def __init__(self, neuron, num_classes=10, dropout=0.0, **kwargs):
        super(SNN5, self).__init__()
        kwargs = dict(kwargs)
        kwargs['_layer_counter'] = {'i': 0, 'total': 5}
        pool = nn.Sequential(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            Layer(3, 16, 3, 1, 1, neuron, **kwargs),
            Layer(16, 64, 5, 1, 1, neuron, **kwargs),
            pool,
            Layer(64, 128, 5, 1, 1, neuron, **kwargs),
            pool,
            Layer(128, 256, 5, 1, 1, neuron, **kwargs),
            pool,
            Layer(256, 512, 3, 1, 1, neuron, **kwargs),
            pool,
        )
        W = int(32 / 2 / 2 / 2 / 2 / 2)

        self.classifier = nn.Linear(512 * W * W, num_classes)
        self.drop = layer.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        # print(x.shape)
        x = self.drop(torch.flatten(x, start_dim=-3, end_dim=-1))
        x = self.classifier(x)
        return x


# use for Figure.2
class SNN5_noAP(nn.Module):
    def __init__(self, neuron, num_classes=10, dropout=0.0, **kwargs):
        super(SNN5_noAP, self).__init__()
        kwargs = dict(kwargs)
        kwargs['_layer_counter'] = {'i': 0, 'total': 5}
        pool = nn.Sequential(nn.AvgPool2d(2))
        # pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(3, 16, 3, 1, 1, neuron, **kwargs),
            Layer(16, 64, 5, 2, 1, neuron, **kwargs),
            Layer(64, 128, 5, 2, 1, neuron, **kwargs),
            Layer(128, 256, 5, 4, 1, neuron, **kwargs),
            Layer(256, 256, 3, 2, 1, neuron, **kwargs),
        )
        # W = int(32 / 2 / 2 / 2 / 4 /  2)
        # if "fc_hw" in kwargs:
        #     W = int(kwargs["fc_hw"] / 2 / 2 / 2 / 2 / 2)

        self.classifier = nn.Linear(256, num_classes)
        self.drop = layer.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = self.drop(torch.flatten(x, start_dim=-3, end_dim=-1))
        x = self.classifier(x)
        return x


def snn5(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SNN5(neuron=neuron, num_classes=num_classes, dropout=neuron_dropout, **kwargs)


def snn5_noAP(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SNN5_noAP(neuron=neuron, num_classes=num_classes, dropout=neuron_dropout, **kwargs)


class DGNDVSCIFAR10Tiny(nn.Module):
    """
    Lightweight 2-hidden-layer SNN for DVS-CIFAR10.

    Hidden layers:
      1) Conv-BN-Neuron (32ch) + AvgPool
      2) Conv-BN-Neuron (64ch) + AvgPool
    Head:
      Flatten + Linear(10)
    """

    def __init__(self, neuron, num_classes=10, neuron_dropout=0.0, c_in=2, fc_hw=48, **kwargs):
        super().__init__()
        kwargs = dict(kwargs)
        kwargs['_layer_counter'] = {'i': 0, 'total': 2}

        self.block1 = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.neuron1 = _build_neuron(neuron, kwargs, zelif_kernel_size=3)
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.neuron2 = _build_neuron(neuron, kwargs, zelif_kernel_size=3)
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        hw = int(fc_hw) if fc_hw is not None else 48
        feat_hw = max(1, hw // 4)
        self.drop = layer.Dropout(neuron_dropout)
        self.classifier = nn.Linear(64 * feat_hw * feat_hw, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.block1(x)
        x = self.neuron1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.neuron2(x)
        x = self.pool2(x)

        x = self.drop(torch.flatten(x, start_dim=-3, end_dim=-1))
        return self.classifier(x)


def dgn_dvscifar10_tiny(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return DGNDVSCIFAR10Tiny(neuron=neuron, num_classes=num_classes, neuron_dropout=neuron_dropout, **kwargs)


def dvscifar10_fc2(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return DVSCIFAR10FC2(neuron=neuron, num_classes=num_classes, neuron_dropout=neuron_dropout, **kwargs)


class Layer(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, neuron, **kwargs):
        super(Layer, self).__init__()
        self.fwd = nn.Sequential(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(out_plane)
        )
        self.act = _build_neuron(neuron, kwargs, zelif_kernel_size=kernel_size)

    def forward(self, x):
        x = self.fwd(x)
        x = self.act(x)
        # print(x.shape)
        return x


class VGGSNN(nn.Module):
    def __init__(self, neuron, num_classes=10, neuron_dropout=0.0, **kwargs):
        super(VGGSNN, self).__init__()
        kwargs = dict(kwargs)
        kwargs['_layer_counter'] = {'i': 0, 'total': 8}
        pool = nn.Sequential(nn.AvgPool2d(2))
        # pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1, neuron, **kwargs),
            Layer(64, 128, 3, 1, 1, neuron, **kwargs),
            pool,
            Layer(128, 256, 3, 1, 1, neuron, **kwargs),
            Layer(256, 256, 3, 1, 1, neuron, **kwargs),
            pool,
            Layer(256, 512, 3, 1, 1, neuron, **kwargs),
            Layer(512, 512, 3, 1, 1, neuron, **kwargs),
            pool,
            Layer(512, 512, 3, 1, 1, neuron, **kwargs),
            Layer(512, 512, 3, 1, 1, neuron, **kwargs),
            pool,
        )
        W = int(48 / 2 / 2 / 2 / 2)
        if "fc_hw" in kwargs:
            W = int(kwargs["fc_hw"] / 2 / 2 / 2 / 2)
        # self.T = 4
        # self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, 10))
        self.classifier = nn.Linear(512 * W * W, num_classes)
        self.drop = layer.Dropout(neuron_dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        # x = torch.flatten(x, 2)
        x = self.drop(torch.flatten(x, start_dim=-3, end_dim=-1))
        x = self.classifier(x)
        return x


class VGGSNNwoAP(nn.Module):
    def __init__(self, neuron, num_classes=10, neuron_dropout=0.0, **kwargs):
        super(VGGSNNwoAP, self).__init__()
        kwargs = dict(kwargs)
        kwargs['_layer_counter'] = {'i': 0, 'total': 8}
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1, neuron, **kwargs),
            Layer(64, 128, 3, 2, 1, neuron, **kwargs),
            Layer(128, 256, 3, 1, 1, neuron, **kwargs),
            Layer(256, 256, 3, 2, 1, neuron, **kwargs),
            Layer(256, 512, 3, 1, 1, neuron, **kwargs),
            Layer(512, 512, 3, 2, 1, neuron, **kwargs),
            Layer(512, 512, 3, 1, 1, neuron, **kwargs),
            Layer(512, 512, 3, 2, 1, neuron, **kwargs),
        )
        W = int(48 / 2 / 2 / 2 / 2)
        if "fc_hw" in kwargs:
            W = int(kwargs["fc_hw"] / 2 / 2 / 2 / 2)

        # self.T = 4
        # self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, 10))
        self.classifier = nn.Linear(512 * W * W, num_classes)
        self.drop = layer.Dropout(neuron_dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # print(input.shape)
        x = self.features(input)
        # print(x.shape)
        x = self.drop(torch.flatten(x, start_dim=-3, end_dim=-1))

        x = self.classifier(x)
        return x


def vggsnn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return VGGSNN(neuron=neuron, num_classes=num_classes, dropout=neuron_dropout, **kwargs)


if __name__ == '__main__':
    # model = VGGSNNwoAP()
    from modules.neuron import ComplementaryLIFNeuron
    from thop import profile

    model = snn5_noAP(neuron=ComplementaryLIFNeuron)
    input = torch.randn(1, 3, 32, 32)
    flops, params = profile(model, inputs=(input,))
    print(model)
