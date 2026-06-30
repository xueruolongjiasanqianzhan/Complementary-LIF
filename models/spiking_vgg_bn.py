import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from spikingjelly.clock_driven import layer

__all__ = [
    'SpikingVGGBN', 'spiking_vgg11_bn', 'spiking_vgg13_bn', 'spiking_vgg16_bn', 'spiking_vgg19_bn'
]

cfg = {

    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}


class IDISIConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups):
        ctx.save_for_backward(weight)
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.input_shape = x.shape
        return F.conv2d(x, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        fan_in = max(1, (weight.shape[1] * weight.shape[2] * weight.shape[3]))
        uniform_weight = torch.ones_like(weight, dtype=grad_output.dtype, device=grad_output.device) / float(fan_in)
        grad_input = torch.nn.grad.conv2d_input(
            ctx.input_shape, uniform_weight, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        per_out = grad_output.sum(dim=(0, 2, 3), dtype=grad_output.dtype).view(-1, 1, 1, 1)
        grad_weight = (per_out.expand_as(weight) / float(fan_in)).to(dtype=weight.dtype)
        grad_bias = grad_output.sum(dim=(0, 2, 3)).to(dtype=weight.dtype) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias, None, None, None, None


class IDISIConv2d(nn.Conv2d):
    """Conv2d with ID-ISI-style uniform connection credit in backward."""

    def forward(self, x):
        return IDISIConv2dFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class IDISILinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(weight)
        ctx.has_bias = bias is not None
        ctx.input_shape = x.shape
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        fan_in = max(1, weight.shape[1])
        uniform_weight = torch.ones_like(weight, dtype=grad_output.dtype, device=grad_output.device) / float(fan_in)
        grad_input = grad_output.matmul(uniform_weight)
        flat_grad = grad_output.reshape(-1, grad_output.shape[-1])
        per_out = flat_grad.sum(dim=0, dtype=grad_output.dtype).view(-1, 1)
        grad_weight = (per_out.expand_as(weight) / float(fan_in)).to(dtype=weight.dtype)
        grad_bias = flat_grad.sum(dim=0).to(dtype=weight.dtype) if ctx.has_bias else None
        return grad_input.reshape(ctx.input_shape), grad_weight, grad_bias


class IDISILinear(nn.Linear):
    """Linear with ID-ISI-style uniform connection credit in backward."""

    def forward(self, x):
        return IDISILinearFunction.apply(x, self.weight, self.bias)


class SynapticReleaseConv2d(nn.Conv2d):
    """Conv2d with a learnable release threshold for every conv synapse.

    By default, the threshold shape matches the convolution weight shape
    ``[out_channels, in_channels, kernel_h, kernel_w]``.  If
    ``synaptic_release_groups`` is positive, synapses are deterministically
    assigned to that many random groups and all synapses in a group share one
    learnable threshold.  When a presynaptic pre-reset membrane tensor is
    provided, each unfolded input connection releases by
    ``surrogate(v_pre[i, k] - release_threshold[o, i, k])`` and the release event
    itself is used for the weighted sum.  The release threshold is clamped to at
    least the soma firing threshold, so release requires a membrane level that
    would also trigger a soma spike.
    """

    def __init__(
        self, *args, release_threshold_init=1.0, surrogate_function=None,
        synaptic_release_chunk_size=16, synaptic_release_groups=0,
        release_threshold_min=1.0,
        synaptic_release_fixed_threshold_ratio=0.5,
        synaptic_release_group_seed=2022, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.release_threshold_min = float(release_threshold_min)
        if release_threshold_init < self.release_threshold_min:
            raise ValueError('release_threshold_init must be at least release_threshold_min.')
        if synaptic_release_chunk_size <= 0:
            raise ValueError('synaptic_release_chunk_size must be positive.')
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
        self.synaptic_release_chunk_size = int(synaptic_release_chunk_size)
        self.last_release_gate = None
        self.last_release_gate_mean = None
        self.last_release_source = None

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
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.groups != 1:
            raise NotImplementedError('SynapticReleaseConv2d currently supports groups=1 only.')
        if release_source.shape != x.shape:
            raise ValueError('release_source must have the same shape as the presynaptic input tensor.')

        v_cols = F.unfold(release_source, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        batch_size, in_kernel, num_locations = v_cols.shape
        weight_flat = self.weight.view(self.out_channels, in_kernel)
        threshold_flat = self._get_release_threshold(dtype=x.dtype, device=x.device).view(self.out_channels, in_kernel)

        def release_chunk(v_cols_chunk, weight_chunk, threshold_chunk):
            release_arg = v_cols_chunk.unsqueeze(1) - threshold_chunk.view(1, weight_chunk.shape[0], in_kernel, 1)
            if self.surrogate_function is None:
                release_gate = (release_arg >= 0.0).to(dtype=x.dtype)
            else:
                release_gate = self.surrogate_function(release_arg)
            return (release_gate * weight_chunk.view(1, weight_chunk.shape[0], in_kernel, 1)).sum(dim=2)

        out_chunks = []
        gate_sum = v_cols.new_zeros(())
        gate_count = 0
        chunk_size = min(self.synaptic_release_chunk_size, self.out_channels)
        for start in range(0, self.out_channels, chunk_size):
            end = min(start + chunk_size, self.out_channels)
            weight_chunk = weight_flat[start:end]
            threshold_chunk = threshold_flat[start:end]
            if self.training and torch.is_grad_enabled():
                out_chunk = checkpoint(release_chunk, v_cols, weight_chunk, threshold_chunk, use_reentrant=False)
            else:
                out_chunk = release_chunk(v_cols, weight_chunk, threshold_chunk)
            out_chunks.append(out_chunk)
            with torch.no_grad():
                release_arg = v_cols.unsqueeze(1) - threshold_chunk.view(1, end - start, in_kernel, 1)
                if self.surrogate_function is None:
                    gate_chunk = (release_arg >= 0.0).to(dtype=x.dtype)
                else:
                    gate_chunk = self.surrogate_function(release_arg)
                gate_sum = gate_sum + gate_chunk.detach().sum()
                gate_count += gate_chunk.numel()
        out = torch.cat(out_chunks, dim=1)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)

        out_h = (x.shape[-2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        out_w = (x.shape[-1] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        self.last_release_gate = None
        self.last_release_gate_mean = gate_sum / max(1, gate_count)
        self.last_release_source = release_source.detach()
        return out.view(batch_size, self.out_channels, out_h, out_w)


class SynapticReleaseSequential(nn.Sequential):
    """Sequential container that forwards presynaptic membrane to SR convs."""

    def forward(self, x, release_source=None):
        for module in self:
            if isinstance(module, SynapticReleaseConv2d):
                x = module(x, release_source)
            else:
                x = module(x)
            if 'Neuron' in module.__class__.__name__:
                release_source = getattr(module, 'last_v_pre', None)
            elif release_source is not None and isinstance(module, nn.AvgPool2d):
                release_source = module(release_source)
        return x, release_source


class SpikingVGGBN(nn.Module):
    def __init__(self, vgg_name, neuron: callable = None, dropout=0.0, num_classes=10, **kwargs):
        super(SpikingVGGBN, self).__init__()
        self.whether_bias = True
        self.init_channels = kwargs.get('c_in', 2)
        self.history_mode = kwargs.get('history_mode', 'all')
        self.synaptic_release_enable = bool(kwargs.get('synaptic_release_enable', False))
        self.total_neuron_layers = sum(1 for stage in cfg[vgg_name] for v in stage if v != 'M')
        self.layer_index = 0

        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout, neuron, **kwargs)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout, neuron, **kwargs)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout, neuron, **kwargs)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout, neuron, **kwargs)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout, neuron, **kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        linear_cls = IDISILinear if getattr(neuron, '__name__', '') == 'IDISILIFNeuron' else nn.Linear
        self.classifier = nn.Sequential(
            nn.Flatten(),
            linear_cls(512 * 7 * 7, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, dropout, neuron, **kwargs):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                neuron_kwargs = dict(kwargs)
                if self.history_mode == 'half' or neuron_kwargs.get('asn_enable', False):
                    neuron_kwargs['layer_index'] = self.layer_index
                    neuron_kwargs['total_layers'] = self.total_neuron_layers
                conv_in_channels = self.init_channels
                conv_cls = IDISIConv2d if getattr(neuron, '__name__', '') == 'IDISILIFNeuron' else nn.Conv2d
                if self.synaptic_release_enable:
                    layers.append(SynapticReleaseConv2d(
                        conv_in_channels, x, kernel_size=3, padding=1, bias=self.whether_bias,
                        release_threshold_init=neuron_kwargs.get('release_threshold_init', 1.0),
                        surrogate_function=neuron_kwargs.get('surrogate_function', None),
                        synaptic_release_chunk_size=neuron_kwargs.get('synaptic_release_chunk_size', 16),
                        synaptic_release_groups=neuron_kwargs.get('synaptic_release_groups', 0),
                        release_threshold_min=neuron_kwargs.get('v_threshold', 1.0),
                        synaptic_release_fixed_threshold_ratio=neuron_kwargs.get('synaptic_release_fixed_threshold_ratio', 0.5),
                        synaptic_release_group_seed=neuron_kwargs.get('synaptic_release_group_seed', 2022),
                    ))
                else:
                    layers.append(conv_cls(conv_in_channels, x, kernel_size=3, padding=1, bias=self.whether_bias))
                layers.append(nn.BatchNorm2d(x))
                if getattr(neuron, '__name__', '') == 'IDISILIFNeuron':
                    neuron_kwargs['idisi_fan_in'] = conv_in_channels * 3 * 3
                layers.append(neuron(**neuron_kwargs))
                layers.append(layer.Dropout(dropout))
                self.init_channels = x
                self.layer_index += 1
        if self.synaptic_release_enable:
            return SynapticReleaseSequential(*layers)
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.synaptic_release_enable:
            release_source = None
            out, release_source = self.layer1(x, release_source)
            out, release_source = self.layer2(out, release_source)
            out, release_source = self.layer3(out, release_source)
            out, release_source = self.layer4(out, release_source)
            out, release_source = self.layer5(out, release_source)
        else:
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
        out = self.avgpool(out)
        out = self.classifier(out)

        return out


def spiking_vgg9_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN('VGG9', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)


def spiking_vgg11_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN('VGG11', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)


def spiking_vgg13_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN('VGG13', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)


def spiking_vgg16_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN('VGG16', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)


def spiking_vgg19_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN('VGG19', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)
