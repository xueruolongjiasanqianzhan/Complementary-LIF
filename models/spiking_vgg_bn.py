import torch
import torch.nn as nn
import torch.nn.functional as F
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


class SpikingVGGBN(nn.Module):
    def __init__(self, vgg_name, neuron: callable = None, dropout=0.0, num_classes=10, **kwargs):
        super(SpikingVGGBN, self).__init__()
        self.whether_bias = True
        self.init_channels = kwargs.get('c_in', 2)
        self.history_mode = kwargs.get('history_mode', 'all')
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
                layers.append(conv_cls(conv_in_channels, x, kernel_size=3, padding=1, bias=self.whether_bias))
                layers.append(nn.BatchNorm2d(x))
                if getattr(neuron, '__name__', '') == 'IDISILIFNeuron':
                    neuron_kwargs['idisi_fan_in'] = conv_in_channels * 3 * 3
                layers.append(neuron(**neuron_kwargs))
                layers.append(layer.Dropout(dropout))
                self.init_channels = x
                self.layer_index += 1
        return nn.Sequential(*layers)

    def forward(self, x):
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
