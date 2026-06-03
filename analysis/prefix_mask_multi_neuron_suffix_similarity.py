#!/usr/bin/env python3
"""Suffix-only prefix-mask spike similarity curves for multiple neuron models.

This script is intentionally separate from ``prefix_mask_spike_similarity.py``.
It compares any number of trained neuron-model checkpoints on the same
DVS-CIFAR10/VGG11 prefix-masking intervention, but only plots the suffix
portion (time steps t > k) of the layer-wise spike-train Jaccard distance.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, ANALYSIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models import spiking_vgg_bn  # noqa: E402
from modules import neuron  # noqa: E402
from prefix_mask_spike_similarity import (  # noqa: E402
    SpikeRecorder,
    build_test_loader,
    get_checkpoint_state,
    get_plt,
    jaccard_distance,
    load_checkpoint,
    load_namespace_args,
    normalize_frames,
    require_file,
    resolve_target_layers,
    run_sequence,
    safe_torch_load,  # imported for compatibility/export in config debugging
    sync_args_with_checkpoint_state,
    write_csv,
    choose_surrogate,
)


NEURON_CHOICES = ('LIF', 'LSLIF', 'CLIF', 'LSCLIF', 'PLIF', 'LSPLIF')


def choose_neuron_multi(name: str):
    mapping = {
        'LIF': neuron.VanillaLIFNeuron,
        'LSLIF': neuron.LSLIFNeuron,
        'CLIF': neuron.ComplementaryLIFNeuron,
        'LSCLIF': neuron.LSCLIFNeuron,
        'PLIF': neuron.PLIFNeuron,
        'LSPLIF': neuron.LSPLIFNeuron,
    }
    if name not in mapping:
        raise NotImplementedError(f'Unsupported neuron model {name}. Supported: {", ".join(mapping)}')
    return mapping[name]


def sync_history_flags_from_checkpoint(args_ns: SimpleNamespace, state_dict: Dict[str, torch.Tensor], neuron_model_name: str):
    """Sync learnable LS-history flags for all LS variants from checkpoint keys."""
    if neuron_model_name not in {'LSLIF', 'LSCLIF', 'LSPLIF'}:
        return args_ns
    # Reuse the original helper for LSLIF-shaped history parameters. It only
    # checks for keys ending in history_weight_raw/history_power_raw, which is
    # also how LSCLIF/LSPLIF store their history branch parameters.
    return sync_args_with_checkpoint_state(args_ns, state_dict, 'LSLIF')


def build_vgg11_multi(args_ns: SimpleNamespace, forced_neuron_model: str, device: torch.device) -> torch.nn.Module:
    if args_ns.model != 'spiking_vgg11_bn':
        raise NotImplementedError(f'Expected spiking_vgg11_bn for this analysis, got {args_ns.model}')
    surrogate_function = choose_surrogate(args_ns.surrogate)
    neuron_model = choose_neuron_multi(forced_neuron_model)
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


def update_suffix_curve_stats(stats, method: str, layer_key: str, mask_prefix: int, distances: torch.Tensor):
    """Accumulate only suffix time steps t > k from [B, T] distances."""
    batch_size = distances.shape[0]
    for t in range(mask_prefix, distances.shape[1]):
        key = (method, layer_key, mask_prefix, t + 1)
        stats[key]['sum'] += distances[:, t].sum().item()
        stats[key]['count'] += batch_size


def suffix_summary_from_rows(curve_rows: List[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in curve_rows:
        grouped[(row['method'], row['layer_key'], row['layer_name'], row['mask_prefix'])].append(row)
    summary = []
    for (method, layer_key, layer_name, mask_prefix), rows in grouped.items():
        total = sum(float(row['jaccard_distance']) * int(row['samples']) for row in rows)
        count = sum(int(row['samples']) for row in rows)
        dist = total / max(count, 1)
        summary.append({
            'method': method,
            'layer_key': layer_key,
            'layer_name': layer_name,
            'mask_prefix': mask_prefix,
            'suffix_jaccard_distance': dist,
            'suffix_jaccard_similarity': 1.0 - dist,
            'samples_x_time': count,
        })
    return sorted(summary, key=lambda r: (int(r['mask_prefix']), r['layer_key'], r['method']))


def plot_suffix_curves(curve_rows: List[dict], methods: List[str], layers: List[str], mask_prefixes: List[int], out_path: Path):
    plt = get_plt()
    layer_names = {row['layer_key']: row['layer_name'] for row in curve_rows}
    lookup = {
        (row['method'], row['layer_key'], int(row['mask_prefix']), int(row['time_step'])): float(row['jaccard_distance'])
        for row in curve_rows
    }
    fig, axes = plt.subplots(len(mask_prefixes), len(layers), figsize=(5.6 * len(layers), 3.8 * len(mask_prefixes)), squeeze=False)
    for r, k in enumerate(mask_prefixes):
        for c, layer_key in enumerate(layers):
            ax = axes[r][c]
            for method in methods:
                times = sorted(t for (m, l, kk, t) in lookup if m == method and l == layer_key and kk == k)
                values = [lookup[(method, layer_key, k, t)] for t in times]
                ax.plot(times, values, marker='o', linewidth=1.8, label=method)
            ax.set_title(f'suffix only: k={k}, {layer_names.get(layer_key, layer_key)}')
            ax.set_xlabel('suffix time step t (t > k)')
            ax.set_ylabel('Jaccard distance')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle('Prefix-masked suffix spike-train dissimilarity across neuron models', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Suffix-only prefix-mask spike similarity curves for multiple DVS-CIFAR10 VGG11 neuron models.'
    )
    parser.add_argument(
        '--run',
        action='append',
        nargs=4,
        metavar=('LABEL', 'NEURON_MODEL', 'CHECKPOINT', 'ARGS_TXT'),
        required=True,
        help=(
            'Add one method to compare. Example: --run LIF LIF /path/checkpoint_max.pth /path/args.txt. '
            f'NEURON_MODEL choices: {", ".join(NEURON_CHOICES)}. Repeat this option for every neuron model.'
        ),
    )
    parser.add_argument('--data-dir', required=True, help='Required DVS-CIFAR10 data root.')
    parser.add_argument('--out-dir', required=True, help='Required output directory for CSV/JSON/figures.')
    parser.add_argument('--T', type=int, default=16, help='Number of time steps. Must match checkpoints/data frames.')
    parser.add_argument('--mask-prefixes', type=int, nargs='+', default=[2, 4, 8], help='Prefix lengths k to zero-mask.')
    parser.add_argument('--batch-size', type=int, default=16, help='Test batch size.')
    parser.add_argument('--workers', type=int, default=0, help='DataLoader workers.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device, e.g. cuda or cpu.')
    parser.add_argument('--layer-mode', choices=['shallow_middle_deep', 'all'], default='shallow_middle_deep')
    parser.add_argument('--layer-indices', type=int, nargs='*', default=[], help='Optional 1-based neuron-layer indices overriding --layer-mode.')
    parser.add_argument('--seed', type=int, default=2022)
    return parser.parse_args()


def main():
    cli = parse_args()
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)

    for k in cli.mask_prefixes:
        if k < 0 or k >= cli.T:
            raise ValueError(f'Each mask prefix k must satisfy 0 <= k < T. Got k={k}, T={cli.T}')

    methods: List[Tuple[str, str, Path, Path]] = []
    seen_labels = set()
    for label, neuron_model_name, checkpoint_path, args_path in cli.run:
        if label in seen_labels:
            raise ValueError(f'Duplicate method label: {label}')
        if neuron_model_name not in NEURON_CHOICES:
            raise ValueError(f'Unsupported neuron model {neuron_model_name}. Supported: {", ".join(NEURON_CHOICES)}')
        seen_labels.add(label)
        methods.append((
            label,
            neuron_model_name,
            require_file(checkpoint_path, f'{label} checkpoint'),
            require_file(args_path, f'{label} args.txt'),
        ))

    out_dir = Path(cli.out_dir).expanduser().resolve()
    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cli.device)
    test_loader = build_test_loader(cli.data_dir, cli.T, cli.batch_size, cli.workers)

    curve_stats = defaultdict(lambda: {'sum': 0.0, 'count': 0})
    selected_layer_keys = None
    layer_names_by_key = {}
    checkpoint_meta = {}

    for label, neuron_model_name, checkpoint_path, args_path in methods:
        checkpoint, state_dict = get_checkpoint_state(checkpoint_path, device)
        train_args = load_namespace_args(args_path)
        train_args.T = cli.T
        train_args.b = cli.batch_size
        train_args = sync_history_flags_from_checkpoint(train_args, state_dict, neuron_model_name)
        model = build_vgg11_multi(train_args, neuron_model_name, device)
        load_checkpoint(model, state_dict)
        model.eval()
        checkpoint_meta[label] = {
            'neuron_model': neuron_model_name,
            'checkpoint': str(checkpoint_path),
            'args': str(args_path),
            'epoch': checkpoint.get('epoch') if isinstance(checkpoint, dict) else None,
            'max_test_acc': checkpoint.get('max_test_acc') if isinstance(checkpoint, dict) else None,
            'history_learn_weight': getattr(train_args, 'history_learn_weight', None),
            'history_weight_per_step': getattr(train_args, 'history_weight_per_step', None),
            'history_learn_power': getattr(train_args, 'history_learn_power', None),
        }

        target_modules = resolve_target_layers(model, cli.layer_mode, cli.layer_indices)
        if selected_layer_keys is None:
            selected_layer_keys = list(target_modules.keys())
        layer_names_by_key.update({key: key.split('_', 1)[1] for key in target_modules})
        recorder = SpikeRecorder(target_modules)

        try:
            for batch_idx, (frame, _label) in enumerate(test_loader):
                frames = normalize_frames(frame, cli.T)
                full_spikes, _full_logits = run_sequence(model, recorder, frames, mask_prefix=0, device=device)
                for k in cli.mask_prefixes:
                    masked_spikes, _masked_logits = run_sequence(model, recorder, frames, mask_prefix=k, device=device)
                    for layer_key in selected_layer_keys:
                        distances = jaccard_distance(full_spikes[layer_key], masked_spikes[layer_key])
                        update_suffix_curve_stats(curve_stats, label, layer_key, k, distances)
                if batch_idx % 20 == 0:
                    print(f'[{label}] processed batch {batch_idx + 1}/{len(test_loader)}')
        finally:
            recorder.close()
            from spikingjelly.clock_driven import functional
            functional.reset_net(model)

    curve_rows = []
    for (method, layer_key, mask_prefix, time_step), stat in sorted(curve_stats.items(), key=lambda item: (item[0][2], item[0][1], item[0][0], item[0][3])):
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
    summary_rows = suffix_summary_from_rows(curve_rows)

    write_csv(
        out_dir / 'multi_neuron_suffix_similarity_curves.csv',
        curve_rows,
        ['method', 'layer_key', 'layer_name', 'mask_prefix', 'time_step', 'jaccard_distance', 'jaccard_similarity', 'samples'],
    )
    write_csv(
        out_dir / 'multi_neuron_suffix_similarity_summary.csv',
        summary_rows,
        ['method', 'layer_key', 'layer_name', 'mask_prefix', 'suffix_jaccard_distance', 'suffix_jaccard_similarity', 'samples_x_time'],
    )

    config = {
        'analysis': 'prefix_mask_multi_neuron_suffix_similarity',
        'dataset': 'DVSCIFAR10',
        'model': 'spiking_vgg11_bn',
        'methods': [method[0] for method in methods],
        'checkpoints': checkpoint_meta,
        'data_dir': str(Path(cli.data_dir).expanduser().resolve()),
        'T': cli.T,
        'mask_prefixes': cli.mask_prefixes,
        'mask_type': 'zero',
        'metric': 'jaccard_distance',
        'plot_scope': 'suffix time steps only (t > k)',
        'batch_size': cli.batch_size,
        'workers': cli.workers,
        'device': str(device),
        'layer_mode': cli.layer_mode,
        'layer_indices': cli.layer_indices,
        'selected_layers': [{'layer_key': key, 'layer_name': layer_names_by_key.get(key, key)} for key in (selected_layer_keys or [])],
    }
    (out_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    method_names = [method[0] for method in methods]
    plot_suffix_curves(
        curve_rows,
        method_names,
        selected_layer_keys or [],
        cli.mask_prefixes,
        figures_dir / 'multi_neuron_suffix_similarity_curve.png',
    )
    print(f'Analysis complete. Results saved to: {out_dir}')


if __name__ == '__main__':
    main()
