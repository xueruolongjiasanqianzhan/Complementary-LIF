#!/usr/bin/env python3
"""Prefix-masked layer-wise spike-train similarity analysis.

This standalone script loads two trained checkpoints (LIF and LSLIF by default),
runs DVS-CIFAR10 test sequences with and without zero-masked prefix frames, and
compares selected VGG11 neuron-layer spike trains on the suffix time steps whose
current inputs are identical.
"""

import argparse
import ast
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import numpy as np


def get_plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt

# NumPy>=1.20 compatibility for legacy third-party code used by this repo.
if not hasattr(np, 'object'):
    np.object = object  # type: ignore[attr-defined]
if not hasattr(np, 'bool'):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, 'int'):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict  # type: ignore[attr-defined]

import torch
from spikingjelly.clock_driven import functional, surrogate as surrogate_sj
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import spiking_vgg_bn  # noqa: E402
from modules import neuron  # noqa: E402
from modules import surrogate as surrogate_self  # noqa: E402
from utils.augmentation import Resize, ToPILImage, ToTensor  # noqa: E402
from utils.cifar10_dvs import CIFAR10DVS  # noqa: E402


DEFAULTS = {
    'seed': 2022,
    'T': 16,
    'tau': 2.0,
    'b': 16,
    'j': 0,
    'dataset': 'DVSCIFAR10',
    'model': 'spiking_vgg11_bn',
    'drop_rate': 0.0,
    'surrogate': 'rectangle',
    'neuron_model': 'LIF',
    'tau_mode': 'spike',
    'tau_lo': None,
    'tau_hi': None,
    'tau_eta': 1.0,
    'tau_alpha_up': 0.1,
    'tau_alpha_down': 0.1,
    'tau_detach_spike': True,
    'tau_eps': 1e-6,
    'tau_learn_alpha': False,
    'tau_alpha_share': False,
    'tau_learn_eta': False,
    'history_weight': 1.0,
    'history_power': 1.0,
    'history_eps': 1e-6,
    'history_learn_weight': False,
    'history_weight_lo': -0.8,
    'history_weight_hi': 0.8,
    'history_weight_per_step': False,
    'history_learn_power': False,
    'history_mode': 'all',
    'asn_enable': False,
    'asn_p': 0.5,
    'asn_rho': 0.5,
    'asn_seed': 2022,
    'asn_detach_lateral': False,
}


class SpikeRecorder:
    """Forward-hook recorder for selected neuron modules."""

    def __init__(self, named_modules: Dict[str, torch.nn.Module]):
        self.named_modules = named_modules
        self.buffers = {name: [] for name in named_modules}
        self.handles = [module.register_forward_hook(self._make_hook(name)) for name, module in named_modules.items()]

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if torch.is_tensor(output):
                self.buffers[name].append(output.detach().float().cpu())
        return hook

    def clear(self):
        for values in self.buffers.values():
            values.clear()

    def stacked(self) -> Dict[str, torch.Tensor]:
        return {name: torch.stack(values, dim=1) for name, values in self.buffers.items() if values}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def split_top_level(text: str) -> List[str]:
    parts = []
    start = 0
    depth = 0
    quote = None
    escape = False
    for idx, ch in enumerate(text):
        if quote is not None:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == quote:
                quote = None
            continue
        if ch in {'"', "'"}:
            quote = ch
        elif ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        elif ch == ',' and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    last = text[start:].strip()
    if last:
        parts.append(last)
    return parts


def load_namespace_args(args_path: Path) -> SimpleNamespace:
    if not args_path.is_file():
        raise FileNotFoundError(f'args file does not exist: {args_path}')
    raw = args_path.read_text(encoding='utf-8').strip()
    if raw.startswith('Namespace(') and raw.endswith(')'):
        raw = raw[len('Namespace('):-1]
    values = dict(DEFAULTS)
    for part in split_top_level(raw):
        if '=' not in part:
            continue
        key, value_text = part.split('=', 1)
        key = key.strip()
        value_text = value_text.strip()
        try:
            value = ast.literal_eval(value_text)
        except (SyntaxError, ValueError):
            value = value_text
        values[key] = value
    return SimpleNamespace(**values)


def require_file(path: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f'{label} does not exist: {resolved}')
    return resolved


def choose_surrogate(name: str):
    if name == 'sigmoid':
        return surrogate_sj.Sigmoid()
    if name == 'rectangle':
        return surrogate_self.Rectangle()
    if name == 'triangle':
        return surrogate_sj.PiecewiseQuadratic()
    raise NotImplementedError(f'Unsupported surrogate: {name}')


def choose_neuron(name: str):
    if name == 'LIF':
        return neuron.VanillaLIFNeuron
    if name == 'LSLIF':
        return neuron.LSLIFNeuron
    raise NotImplementedError(f'This analysis currently supports LIF and LSLIF only, got {name}')


def build_vgg11(args_ns: SimpleNamespace, forced_neuron_model: str, device: torch.device) -> torch.nn.Module:
    if args_ns.model != 'spiking_vgg11_bn':
        raise NotImplementedError(f'Expected spiking_vgg11_bn for this analysis, got {args_ns.model}')
    surrogate_function = choose_surrogate(args_ns.surrogate)
    neuron_model = choose_neuron(forced_neuron_model)
    neuron_kwargs = dict(
        tau=args_ns.tau,
        surrogate_function=surrogate_function,
        tau_mode=getattr(args_ns, 'tau_mode', 'spike'),
        tau_lo=getattr(args_ns, 'tau_lo', None),
        tau_hi=getattr(args_ns, 'tau_hi', None),
        tau_eta=getattr(args_ns, 'tau_eta', 1.0),
        tau_alpha_up=getattr(args_ns, 'tau_alpha_up', 0.1),
        tau_alpha_down=getattr(args_ns, 'tau_alpha_down', 0.1),
        tau_detach_spike=getattr(args_ns, 'tau_detach_spike', True),
        tau_eps=getattr(args_ns, 'tau_eps', 1e-6),
        tau_learn_alpha=getattr(args_ns, 'tau_learn_alpha', False),
        tau_alpha_share=getattr(args_ns, 'tau_alpha_share', False),
        tau_learn_eta=getattr(args_ns, 'tau_learn_eta', False),
        history_weight=getattr(args_ns, 'history_weight', 1.0),
        history_power=getattr(args_ns, 'history_power', 1.0),
        history_eps=getattr(args_ns, 'history_eps', 1e-6),
        history_learn_weight=getattr(args_ns, 'history_learn_weight', False),
        history_weight_lo=getattr(args_ns, 'history_weight_lo', -0.8),
        history_weight_hi=getattr(args_ns, 'history_weight_hi', 0.8),
        history_weight_per_step=getattr(args_ns, 'history_weight_per_step', False),
        history_max_steps=getattr(args_ns, 'T', 16),
        history_learn_power=getattr(args_ns, 'history_learn_power', False),
        history_mode=getattr(args_ns, 'history_mode', 'all'),
        asn_enable=getattr(args_ns, 'asn_enable', False),
        asn_p=getattr(args_ns, 'asn_p', 0.5),
        asn_rho=getattr(args_ns, 'asn_rho', 0.5),
        asn_seed=getattr(args_ns, 'asn_seed', 2022),
        asn_detach_lateral=getattr(args_ns, 'asn_detach_lateral', False),
    )
    net = spiking_vgg_bn.spiking_vgg11_bn(
        neuron=neuron_model,
        num_classes=10,
        neuron_dropout=getattr(args_ns, 'drop_rate', 0.0),
        c_in=2,
        **neuron_kwargs,
    )
    return net.to(device)


def safe_torch_load(checkpoint_path: Path, device: torch.device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        return torch.load(checkpoint_path, map_location=device)


def get_checkpoint_state(checkpoint_path: Path, device: torch.device):
    checkpoint = safe_torch_load(checkpoint_path, device)
    state_dict = checkpoint.get('net', checkpoint)
    if any(key.startswith('module.') for key in state_dict):
        state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
    return checkpoint, state_dict


def sync_args_with_checkpoint_state(args_ns: SimpleNamespace, state_dict: Dict[str, torch.Tensor], neuron_model_name: str):
    if neuron_model_name != 'LSLIF':
        return args_ns

    history_weight_keys = [key for key in state_dict if key.endswith('history_weight_raw')]
    history_power_keys = [key for key in state_dict if key.endswith('history_power_raw')]
    args_ns.history_learn_weight = bool(history_weight_keys)
    args_ns.history_learn_power = bool(history_power_keys)
    if history_weight_keys:
        first_weight = state_dict[history_weight_keys[0]]
        args_ns.history_weight_per_step = first_weight.ndim > 0 and first_weight.numel() > 1
        if args_ns.history_weight_per_step:
            args_ns.history_max_steps = int(first_weight.numel())
    else:
        args_ns.history_weight_per_step = False
    return args_ns


def load_checkpoint(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]):
    model.load_state_dict(state_dict)


def build_test_loader(data_dir: str, T: int, batch_size: int, workers: int) -> DataLoader:
    from torchvision import transforms
    transform_test = transforms.Compose([
        ToPILImage(),
        Resize(48),
        ToTensor(),
    ])
    testset = CIFAR10DVS(
        data_dir,
        train=False,
        use_frame=True,
        frames_num=T,
        split_by='number',
        normalization=None,
        transform=transform_test,
    )
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers)


def get_neuron_modules(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    return [(name, module) for name, module in model.named_modules() if 'Neuron' in module.__class__.__name__]


def resolve_target_layers(model: torch.nn.Module, layer_mode: str, layer_indices: List[int]) -> Dict[str, torch.nn.Module]:
    neurons = get_neuron_modules(model)
    if not neurons:
        raise RuntimeError('No neuron modules found in model.')
    if layer_indices:
        positions = []
        for idx in layer_indices:
            if idx < 1 or idx > len(neurons):
                raise ValueError(f'Layer index {idx} is out of range 1..{len(neurons)}')
            positions.append(idx - 1)
    elif layer_mode == 'all':
        positions = list(range(len(neurons)))
    elif layer_mode == 'shallow_middle_deep':
        # For VGG11 this is neuron layer 1 (shallow), 4 (middle), and 8 (deep).
        positions = [0, min(3, len(neurons) - 1), len(neurons) - 1]
    else:
        raise ValueError(f'Unknown layer mode: {layer_mode}')
    selected = [(pos, neurons[pos]) for pos in positions]
    return {f'{pos + 1:02d}_{name}': module for pos, (name, module) in selected}


def normalize_frames(frame, T: int) -> torch.Tensor:
    """Return DVS frames as [T, B, C, H, W].

    CIFAR10DVS uses transforms that return a Python list with one tensor per
    time step. PyTorch's default collate therefore yields ``frame`` as a list of
    length ``T`` where each item is shaped [B, C, H, W]. Other DVS loaders may
    already return a tensor, so both layouts are accepted here.
    """
    if isinstance(frame, (list, tuple)):
        if len(frame) != T:
            raise ValueError(f'Expected a list/tuple with {T} time steps, got {len(frame)}')
        if not all(torch.is_tensor(step) for step in frame):
            bad_types = [type(step).__name__ for step in frame[:3]]
            raise TypeError(f'Expected all list/tuple items to be tensors, got first item types {bad_types}')
        return torch.stack([step.float() for step in frame], dim=0)

    if not torch.is_tensor(frame):
        raise TypeError(f'Expected DVS frames as a tensor or list/tuple of tensors, got {type(frame).__name__}')
    if frame.dim() != 5:
        raise ValueError(f'Expected DVS frame tensor with 5 dims, got shape {tuple(frame.shape)}')
    if frame.shape[0] == T:
        return frame.float()
    if frame.shape[1] == T:
        return frame.transpose(0, 1).float()
    raise ValueError(f'Cannot infer time dimension for shape {tuple(frame.shape)} and T={T}')


def run_sequence(model: torch.nn.Module, recorder: SpikeRecorder, frames: torch.Tensor, mask_prefix: int, device: torch.device):
    functional.reset_net(model)
    recorder.clear()
    logits = []
    with torch.no_grad():
        for t in range(frames.shape[0]):
            x_t = frames[t].to(device, non_blocking=True)
            if t < mask_prefix:
                x_t = torch.zeros_like(x_t)
            out_t = model(x_t)
            logits.append(out_t.detach().float().cpu())
    outputs = recorder.stacked()
    logits = torch.stack(logits, dim=0)
    functional.reset_net(model)
    return outputs, logits


def jaccard_distance(full: torch.Tensor, masked: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return per-sample, per-time Jaccard distance for spike tensors [B, T, ...]."""
    full_b = full.flatten(start_dim=2) > 0
    mask_b = masked.flatten(start_dim=2) > 0
    inter = (full_b & mask_b).float().sum(dim=2)
    union = (full_b | mask_b).float().sum(dim=2)
    sim = (inter + eps) / (union + eps)
    return 1.0 - sim


def update_curve_stats(stats, method: str, layer_key: str, mask_prefix: int, distances: torch.Tensor):
    # distances: [B, T]
    batch_size = distances.shape[0]
    for t in range(distances.shape[1]):
        key = (method, layer_key, mask_prefix, t + 1)
        stats[key]['sum'] += distances[:, t].sum().item()
        stats[key]['count'] += batch_size


def update_suffix_performance_stats(
    stats,
    method: str,
    mask_prefix: int,
    full_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    labels: torch.Tensor,
):
    # logits: [T, B, num_classes]. Only suffix time steps t > k are used.
    full_suffix = full_logits[mask_prefix:].sum(dim=0)
    masked_suffix = masked_logits[mask_prefix:].sum(dim=0)
    labels = labels.detach().cpu().long()

    full_pred = full_suffix.argmax(dim=1)
    masked_pred = masked_suffix.argmax(dim=1)
    full_prob = torch.softmax(full_suffix, dim=1)
    masked_prob = torch.softmax(masked_suffix, dim=1)
    sample_idx = torch.arange(labels.numel())

    key = (method, mask_prefix)
    stats[key]['samples'] += labels.numel()
    stats[key]['full_correct'] += (full_pred == labels).sum().item()
    stats[key]['masked_correct'] += (masked_pred == labels).sum().item()
    stats[key]['full_true_conf_sum'] += full_prob[sample_idx, labels].sum().item()
    stats[key]['masked_true_conf_sum'] += masked_prob[sample_idx, labels].sum().item()
    stats[key]['top1_flip_sum'] += (full_pred != masked_pred).sum().item()


def performance_rows_from_stats(stats) -> List[dict]:
    rows = []
    for (method, mask_prefix), stat in sorted(stats.items(), key=lambda item: (item[0][0], item[0][1])):
        samples = max(int(stat['samples']), 1)
        full_acc = stat['full_correct'] / samples
        masked_acc = stat['masked_correct'] / samples
        full_conf = stat['full_true_conf_sum'] / samples
        masked_conf = stat['masked_true_conf_sum'] / samples
        rows.append({
            'method': method,
            'mask_prefix': mask_prefix,
            'suffix_full_acc': full_acc,
            'suffix_masked_acc': masked_acc,
            'suffix_acc_drop': full_acc - masked_acc,
            'suffix_full_true_conf': full_conf,
            'suffix_masked_true_conf': masked_conf,
            'suffix_true_conf_drop': full_conf - masked_conf,
            'suffix_top1_flip_rate': stat['top1_flip_sum'] / samples,
            'samples': samples,
        })
    return rows


def suffix_summary_from_curves(curve_rows: List[dict], mask_prefixes: Iterable[int]) -> List[dict]:
    grouped = defaultdict(list)
    mask_set = set(mask_prefixes)
    for row in curve_rows:
        k = int(row['mask_prefix'])
        if k not in mask_set:
            continue
        if int(row['time_step']) <= k:
            continue
        grouped[(row['method'], row['layer_key'], row['layer_name'], k)].append(row)
    summary = []
    for (method, layer_key, layer_name, k), rows in grouped.items():
        total = sum(float(row['jaccard_distance']) * int(row['samples']) for row in rows)
        count = sum(int(row['samples']) for row in rows)
        dist = total / max(count, 1)
        summary.append({
            'method': method,
            'layer_key': layer_key,
            'layer_name': layer_name,
            'mask_prefix': k,
            'suffix_jaccard_distance': dist,
            'suffix_jaccard_similarity': 1.0 - dist,
            'samples_x_time': count,
        })
    return sorted(summary, key=lambda r: (r['method'], int(r['mask_prefix']), r['layer_key']))


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(curve_rows: List[dict], methods: List[str], layers: List[str], mask_prefixes: List[int], out_path: Path):
    plt = get_plt()
    layer_names = {row['layer_key']: row['layer_name'] for row in curve_rows}
    lookup = {(row['method'], row['layer_key'], int(row['mask_prefix']), int(row['time_step'])): float(row['jaccard_distance']) for row in curve_rows}
    fig, axes = plt.subplots(len(mask_prefixes), len(layers), figsize=(5 * len(layers), 3.4 * len(mask_prefixes)), squeeze=False)
    for r, k in enumerate(mask_prefixes):
        for c, layer_key in enumerate(layers):
            ax = axes[r][c]
            for method in methods:
                times = sorted(t for (m, l, kk, t) in lookup if m == method and l == layer_key and kk == k)
                values = [lookup[(method, layer_key, k, t)] for t in times]
                ax.plot(times, values, marker='o', label=method)
            ax.axvline(k + 0.5, color='gray', linestyle='--', linewidth=1)
            ax.set_title(f'k={k}, {layer_names.get(layer_key, layer_key)}')
            ax.set_xlabel('time step')
            ax.set_ylabel('Jaccard distance')
            ax.grid(True, alpha=0.3)
            ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_heatmap(summary_rows: List[dict], methods: List[str], layers: List[str], mask_prefixes: List[int], out_path: Path):
    plt = get_plt()
    layer_names = {row['layer_key']: row['layer_name'] for row in summary_rows}
    lookup = {(row['method'], row['layer_key'], int(row['mask_prefix'])): float(row['suffix_jaccard_distance']) for row in summary_rows}
    subplot_count = len(methods) + (1 if len(methods) == 2 else 0)
    fig, axes = plt.subplots(1, subplot_count, figsize=(5 * subplot_count, 4), squeeze=False)
    vmin = 0.0
    vmax = max([float(row['suffix_jaccard_distance']) for row in summary_rows] + [1e-6])
    for idx, method in enumerate(methods):
        matrix = np.array([[lookup.get((method, layer, k), np.nan) for layer in layers] for k in mask_prefixes])
        ax = axes[0][idx]
        im = ax.imshow(matrix, vmin=vmin, vmax=vmax, aspect='auto', cmap='viridis')
        ax.set_title(method)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([layer_names.get(layer, layer) for layer in layers], rotation=35, ha='right')
        ax.set_yticks(range(len(mask_prefixes)))
        ax.set_yticklabels(mask_prefixes)
        ax.set_xlabel('target layer')
        ax.set_ylabel('mask prefix k')
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', color='white')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if len(methods) == 2:
        base, ls = methods[0], methods[1]
        delta = np.array([[lookup.get((ls, layer, k), np.nan) - lookup.get((base, layer, k), np.nan) for layer in layers] for k in mask_prefixes])
        ax = axes[0][-1]
        max_abs = np.nanmax(np.abs(delta)) if np.isfinite(delta).any() else 1e-6
        im = ax.imshow(delta, vmin=-max_abs, vmax=max_abs, aspect='auto', cmap='coolwarm')
        ax.set_title(f'{ls} - {base}')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([layer_names.get(layer, layer) for layer in layers], rotation=35, ha='right')
        ax.set_yticks(range(len(mask_prefixes)))
        ax.set_yticklabels(mask_prefixes)
        ax.set_xlabel('target layer')
        ax.set_ylabel('mask prefix k')
        for i in range(delta.shape[0]):
            for j in range(delta.shape[1]):
                ax.text(j, i, f'{delta[i, j]:+.3f}', ha='center', va='center', color='black')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_raster_panel(spikes: torch.Tensor, max_neurons: int) -> Tuple[np.ndarray, np.ndarray]:
    # spikes: [T, ...] for one sample.
    flat = (spikes.flatten(start_dim=1) > 0).numpy().astype(np.float32)  # [T, M]
    activity = flat.sum(axis=0)
    if flat.shape[1] > max_neurons:
        chosen = np.argsort(activity)[-max_neurons:]
        chosen = np.sort(chosen)
        flat = flat[:, chosen]
    else:
        chosen = np.arange(flat.shape[1])
    return flat.T, chosen


def plot_raster(raster_cache: Dict[str, dict], methods: List[str], deep_layer_key: str, mask_prefix: int, out_path: Path, max_neurons: int):
    plt = get_plt()
    rows = []
    titles = []
    for method in methods:
        if method not in raster_cache:
            continue
        cache = raster_cache[method]
        full = cache['full'][deep_layer_key][0]
        masked = cache['masked'][deep_layer_key][0]
        full_panel, chosen = make_raster_panel(full, max_neurons)
        masked_flat = (masked.flatten(start_dim=1) > 0).numpy().astype(np.float32)[:, chosen].T
        diff_panel = np.abs(full_panel - masked_flat)
        rows.extend([full_panel, masked_flat, diff_panel])
        titles.extend([f'{method} full', f'{method} mask-{mask_prefix}', f'{method} |diff|'])
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 2.4 * len(rows)), squeeze=False)
    for ax, panel, title in zip(axes[:, 0], rows, titles):
        ax.imshow(panel, aspect='auto', interpolation='nearest', cmap='Greys')
        ax.axvline(mask_prefix - 0.5, color='red', linestyle='--', linewidth=1)
        ax.set_title(title)
        ax.set_xlabel('time step')
        ax.set_ylabel('selected neuron')
        ax.set_xticks(range(panel.shape[1]))
        ax.set_xticklabels([str(i + 1) for i in range(panel.shape[1])])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Prefix-masked layer-wise spike-train similarity for DVS-CIFAR10 VGG11.')
    parser.add_argument('--lif-checkpoint', required=True, help='Required path to LIF checkpoint_max.pth.')
    parser.add_argument('--lsl-checkpoint', required=True, help='Required path to LSLIF checkpoint_max.pth.')
    parser.add_argument('--lif-args', required=True, help='Required path to the LIF run args.txt.')
    parser.add_argument('--lsl-args', required=True, help='Required path to the LSLIF run args.txt.')
    parser.add_argument('--data-dir', required=True, help='Required DVS-CIFAR10 data root.')
    parser.add_argument('--out-dir', required=True, help='Required output directory for CSV/JSON/figures.')
    parser.add_argument('--T', type=int, default=16, help='Number of time steps. Must match checkpoints/data frames.')
    parser.add_argument('--mask-prefixes', type=int, nargs='+', default=[2, 4, 8], help='Prefix lengths k to zero-mask.')
    parser.add_argument('--batch-size', type=int, default=16, help='Test batch size.')
    parser.add_argument('--workers', type=int, default=0, help='DataLoader workers.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device, e.g. cuda or cpu.')
    parser.add_argument('--layer-mode', choices=['shallow_middle_deep', 'all'], default='shallow_middle_deep')
    parser.add_argument('--layer-indices', type=int, nargs='*', default=[], help='Optional 1-based neuron-layer indices overriding --layer-mode.')
    parser.add_argument('--max-raster-neurons', type=int, default=512, help='Max active neurons shown in raster plot.')
    parser.add_argument('--seed', type=int, default=2022)
    return parser.parse_args()


def main():
    cli = parse_args()
    lif_checkpoint = require_file(cli.lif_checkpoint, 'LIF checkpoint')
    lsl_checkpoint = require_file(cli.lsl_checkpoint, 'LSLIF checkpoint')
    lif_args_path = require_file(cli.lif_args, 'LIF args.txt')
    lsl_args_path = require_file(cli.lsl_args, 'LSLIF args.txt')
    out_dir = Path(cli.out_dir).expanduser().resolve()
    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)

    device = torch.device(cli.device)
    methods = [
        ('LIF', lif_checkpoint, lif_args_path, 'LIF'),
        ('LSLIF', lsl_checkpoint, lsl_args_path, 'LSLIF'),
    ]
    test_loader = build_test_loader(cli.data_dir, cli.T, cli.batch_size, cli.workers)

    all_curve_stats = defaultdict(lambda: {'sum': 0.0, 'count': 0})
    performance_stats = defaultdict(lambda: {
        'samples': 0,
        'full_correct': 0,
        'masked_correct': 0,
        'full_true_conf_sum': 0.0,
        'masked_true_conf_sum': 0.0,
        'top1_flip_sum': 0,
    })
    layer_names_by_key = {}
    raster_cache = {}
    selected_layer_keys = None
    deep_layer_key = None
    max_mask_prefix = max(cli.mask_prefixes)
    checkpoint_meta = {}

    for method_name, checkpoint_path, args_path, neuron_model_name in methods:
        checkpoint, state_dict = get_checkpoint_state(checkpoint_path, device)
        train_args = load_namespace_args(args_path)
        train_args.T = cli.T
        train_args.b = cli.batch_size
        train_args = sync_args_with_checkpoint_state(train_args, state_dict, neuron_model_name)
        model = build_vgg11(train_args, neuron_model_name, device)
        load_checkpoint(model, state_dict)
        checkpoint_meta[method_name] = {
            'checkpoint': str(checkpoint_path),
            'args': str(args_path),
            'epoch': checkpoint.get('epoch') if isinstance(checkpoint, dict) else None,
            'max_test_acc': checkpoint.get('max_test_acc') if isinstance(checkpoint, dict) else None,
            'history_learn_weight': getattr(train_args, 'history_learn_weight', None),
            'history_weight_per_step': getattr(train_args, 'history_weight_per_step', None),
            'history_learn_power': getattr(train_args, 'history_learn_power', None),
        }
        model.eval()

        target_modules = resolve_target_layers(model, cli.layer_mode, cli.layer_indices)
        if selected_layer_keys is None:
            selected_layer_keys = list(target_modules.keys())
            deep_layer_key = selected_layer_keys[-1]
        layer_names_by_key.update({key: key.split('_', 1)[1] for key in target_modules})
        recorder = SpikeRecorder(target_modules)

        try:
            for batch_idx, (frame, label) in enumerate(test_loader):
                frames = normalize_frames(frame, cli.T)
                full_spikes, full_logits = run_sequence(model, recorder, frames, mask_prefix=0, device=device)
                for k in cli.mask_prefixes:
                    masked_spikes, masked_logits = run_sequence(model, recorder, frames, mask_prefix=k, device=device)
                    update_suffix_performance_stats(performance_stats, method_name, k, full_logits, masked_logits, label)
                    for layer_key in selected_layer_keys:
                        distances = jaccard_distance(full_spikes[layer_key], masked_spikes[layer_key])
                        update_curve_stats(all_curve_stats, method_name, layer_key, k, distances)
                    if batch_idx == 0 and k == max_mask_prefix and method_name not in raster_cache:
                        raster_cache[method_name] = {'full': full_spikes, 'masked': masked_spikes}
                if batch_idx % 20 == 0:
                    print(f'[{method_name}] processed batch {batch_idx + 1}/{len(test_loader)}')
        finally:
            recorder.close()
            functional.reset_net(model)

    curve_rows = []
    for (method, layer_key, mask_prefix, time_step), stat in sorted(all_curve_stats.items(), key=lambda item: (item[0][0], item[0][2], item[0][1], item[0][3])):
        distance = stat['sum'] / max(stat['count'], 1)
        curve_rows.append({
            'method': method,
            'layer_key': layer_key,
            'layer_name': layer_names_by_key.get(layer_key, layer_key),
            'mask_prefix': mask_prefix,
            'time_step': time_step,
            'jaccard_distance': distance,
            'jaccard_similarity': 1.0 - distance,
            'samples': stat['count'],
        })
    summary_rows = suffix_summary_from_curves(curve_rows, cli.mask_prefixes)
    performance_rows = performance_rows_from_stats(performance_stats)

    write_csv(
        out_dir / 'spike_similarity_curves.csv',
        curve_rows,
        ['method', 'layer_key', 'layer_name', 'mask_prefix', 'time_step', 'jaccard_distance', 'jaccard_similarity', 'samples'],
    )
    write_csv(
        out_dir / 'spike_similarity_summary.csv',
        summary_rows,
        ['method', 'layer_key', 'layer_name', 'mask_prefix', 'suffix_jaccard_distance', 'suffix_jaccard_similarity', 'samples_x_time'],
    )
    write_csv(
        out_dir / 'suffix_performance_summary.csv',
        performance_rows,
        [
            'method', 'mask_prefix', 'suffix_full_acc', 'suffix_masked_acc', 'suffix_acc_drop',
            'suffix_full_true_conf', 'suffix_masked_true_conf', 'suffix_true_conf_drop',
            'suffix_top1_flip_rate', 'samples'
        ],
    )

    config = {
        'analysis': 'prefix_mask_spike_similarity',
        'dataset': 'DVSCIFAR10',
        'model': 'spiking_vgg11_bn',
        'methods': [method[0] for method in methods],
        'checkpoints': checkpoint_meta,
        'data_dir': str(Path(cli.data_dir).expanduser().resolve()),
        'T': cli.T,
        'mask_prefixes': cli.mask_prefixes,
        'mask_type': 'zero',
        'metric': 'jaccard_distance',
        'main_result_scope': 'suffix time steps t > k',
        'performance_summary': 'suffix_performance_summary.csv',
        'batch_size': cli.batch_size,
        'workers': cli.workers,
        'device': str(device),
        'layer_mode': cli.layer_mode,
        'layer_indices': cli.layer_indices,
        'selected_layers': [{'layer_key': key, 'layer_name': layer_names_by_key.get(key, key)} for key in (selected_layer_keys or [])],
        'raw_spikes_saved': False,
    }
    (out_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    method_names = [method[0] for method in methods]
    layers = selected_layer_keys or []
    plot_curves(curve_rows, method_names, layers, cli.mask_prefixes, figures_dir / 'similarity_curve.png')
    plot_heatmap(summary_rows, method_names, layers, cli.mask_prefixes, figures_dir / 'layer_heatmap.png')
    if deep_layer_key is not None:
        plot_raster(raster_cache, method_names, deep_layer_key, max_mask_prefix, figures_dir / 'raster_example.png', cli.max_raster_neurons)

    print(f'Analysis complete. Results saved to: {out_dir}')


if __name__ == '__main__':
    main()
