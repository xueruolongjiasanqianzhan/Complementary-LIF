#!/usr/bin/env python3
"""Evaluate trained DVS-CIFAR10 VGG11 models with early real frames and late noise.

The experiment keeps the first ``cue_steps`` frames from each DVS-CIFAR10 test
sequence and replaces the remaining frames with deterministic random
interference. It loads trained checkpoints only; no training is performed.
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, ANALYSIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prefix_mask_multi_neuron_suffix_similarity import (  # noqa: E402
    NEURON_CHOICES,
    build_vgg11_multi,
    sync_history_flags_from_checkpoint,
)
from prefix_mask_spike_similarity import (  # noqa: E402
    build_test_loader,
    get_checkpoint_state,
    get_plt,
    load_checkpoint,
    load_namespace_args,
    normalize_frames,
    require_file,
    run_sequence,
    write_csv,
)


def make_interference_frames(
    frames: torch.Tensor,
    cue_steps: int,
    noise_type: str,
    generator: torch.Generator,
    event_prob: float,
    gaussian_std: float,
) -> torch.Tensor:
    """Return [T, B, C, H, W] sequence with suffix replaced by random noise."""
    if cue_steps < 0 or cue_steps > frames.shape[0]:
        raise ValueError(f'cue_steps must be in [0, T], got cue_steps={cue_steps}, T={frames.shape[0]}')
    corrupted = frames.clone()
    if cue_steps == frames.shape[0]:
        return corrupted

    suffix = corrupted[cue_steps:]
    if noise_type == 'uniform':
        noise = torch.rand(suffix.shape, generator=generator, dtype=suffix.dtype)
    elif noise_type == 'bernoulli':
        noise = (torch.rand(suffix.shape, generator=generator, dtype=suffix.dtype) < event_prob).to(suffix.dtype)
    elif noise_type == 'gaussian':
        noise = torch.randn(suffix.shape, generator=generator, dtype=suffix.dtype) * gaussian_std
        noise = torch.clamp(noise, 0.0, 1.0)
    elif noise_type == 'shuffle_batch':
        # Real DVS-like interference: use another sample's suffix in the same batch.
        if suffix.shape[1] <= 1:
            raise ValueError('shuffle_batch noise requires batch_size > 1')
        perm = torch.randperm(suffix.shape[1], generator=generator)
        if torch.equal(perm, torch.arange(suffix.shape[1])):
            perm = torch.roll(perm, shifts=1)
        noise = suffix[:, perm]
    else:
        raise ValueError(f'Unsupported noise type: {noise_type}')
    corrupted[cue_steps:] = noise
    return corrupted


def logits_to_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute accuracy/confidence from [T, B, num_classes] logits."""
    summed = logits.sum(dim=0)
    labels = labels.detach().cpu().long()
    pred = summed.argmax(dim=1)
    prob = torch.softmax(summed, dim=1)
    sample_idx = torch.arange(labels.numel())
    return {
        'correct': (pred == labels).sum().item(),
        'true_conf_sum': prob[sample_idx, labels].sum().item(),
        'samples': labels.numel(),
    }


def logits_window_metrics(logits: torch.Tensor, labels: torch.Tensor, start: int, end: int) -> Dict[str, float]:
    if start >= end:
        return {'correct': 0, 'true_conf_sum': 0.0, 'samples': 0}
    return logits_to_metrics(logits[start:end], labels)


def init_stat():
    return {'correct': 0, 'true_conf_sum': 0.0, 'samples': 0}


def update_stat(stat: dict, metric: Dict[str, float]):
    stat['correct'] += metric['correct']
    stat['true_conf_sum'] += metric['true_conf_sum']
    stat['samples'] += metric['samples']


def stat_to_acc_conf(stat: dict) -> Tuple[float, float]:
    samples = max(int(stat['samples']), 1)
    return stat['correct'] / samples, stat['true_conf_sum'] / samples


def plot_accuracy(rows: List[dict], out_path: Path):
    plt = get_plt()
    methods = [row['method'] for row in rows]
    metrics = [
        ('clean_all_acc', 'Clean all-steps acc'),
        ('noise_all_acc', 'Early-cue + noise all-steps acc'),
        ('noise_late_acc', 'Late-noise readout acc'),
        ('noise_cue_acc', 'Cue-only readout acc'),
    ]
    x = np.arange(len(methods))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(1, 1, figsize=(max(10, 1.5 * len(methods)), 5))
    for idx, (key, label) in enumerate(metrics):
        values = [float(row[key]) for row in rows]
        offset = (idx - (len(metrics) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.0, min(1.05, max([float(row[key]) for row in rows for key, _ in metrics] + [0.1]) + 0.12))
    ax.set_title('Early real DVS frames followed by random interference')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_drop(rows: List[dict], out_path: Path):
    plt = get_plt()
    methods = [row['method'] for row in rows]
    metrics = [
        ('noise_all_acc_drop', 'Clean all - noise all'),
        ('noise_late_acc_drop', 'Clean all - late-noise readout'),
    ]
    x = np.arange(len(methods))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(1, 1, figsize=(max(9, 1.3 * len(methods)), 4.6))
    for idx, (key, label) in enumerate(metrics):
        values = [float(row[key]) for row in rows]
        offset = (idx - (len(metrics) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    ax.axhline(0.0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Accuracy drop')
    ax.set_title('Accuracy drop under late random interference')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Early-cue late-noise evaluation for trained DVS-CIFAR10 VGG11 neuron models.')
    parser.add_argument(
        '--run',
        action='append',
        nargs=4,
        metavar=('LABEL', 'NEURON_MODEL', 'CHECKPOINT', 'ARGS_TXT'),
        required=True,
        help=(
            'Add one method. Example: --run LIF LIF /path/checkpoint_max.pth /path/args.txt. '
            f'NEURON_MODEL choices: {", ".join(NEURON_CHOICES)}.'
        ),
    )
    parser.add_argument('--data-dir', required=True, help='Required DVS-CIFAR10 data root.')
    parser.add_argument('--out-dir', required=True, help='Required output directory for CSV/JSON/figures.')
    parser.add_argument('--T', type=int, default=16, help='Total time steps.')
    parser.add_argument('--cue-steps', type=int, default=4, help='Number of initial real DVS frames kept before noise suffix.')
    parser.add_argument('--noise-type', choices=['uniform', 'bernoulli', 'gaussian', 'shuffle_batch'], default='bernoulli')
    parser.add_argument('--event-prob', type=float, default=0.10, help='Event probability for bernoulli noise.')
    parser.add_argument('--gaussian-std', type=float, default=0.20, help='Std for clipped gaussian noise.')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=2022)
    return parser.parse_args()


def main():
    cli = parse_args()
    if cli.cue_steps <= 0 or cli.cue_steps >= cli.T:
        raise ValueError(f'cue_steps must satisfy 0 < cue_steps < T. Got cue_steps={cli.cue_steps}, T={cli.T}')
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)

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
    all_stats = defaultdict(init_stat)
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
        }

        for batch_idx, (frame, labels) in enumerate(test_loader):
            frames = normalize_frames(frame, cli.T)
            # Re-seed by batch so every neuron model sees identical interference
            # for the same test samples, independent of method evaluation order.
            noise_generator = torch.Generator().manual_seed(cli.seed + 1009 + batch_idx)
            noisy_frames = make_interference_frames(
                frames,
                cue_steps=cli.cue_steps,
                noise_type=cli.noise_type,
                generator=noise_generator,
                event_prob=cli.event_prob,
                gaussian_std=cli.gaussian_std,
            )
            _clean_spikes, clean_logits = run_sequence(model, recorder=NullRecorder(), frames=frames, mask_prefix=0, device=device)
            _noise_spikes, noise_logits = run_sequence(model, recorder=NullRecorder(), frames=noisy_frames, mask_prefix=0, device=device)

            update_stat(all_stats[(label, 'clean_all')], logits_to_metrics(clean_logits, labels))
            update_stat(all_stats[(label, 'noise_all')], logits_to_metrics(noise_logits, labels))
            update_stat(all_stats[(label, 'noise_cue')], logits_window_metrics(noise_logits, labels, 0, cli.cue_steps))
            update_stat(all_stats[(label, 'noise_late')], logits_window_metrics(noise_logits, labels, cli.cue_steps, cli.T))

            if batch_idx % 20 == 0:
                print(f'[{label}] processed batch {batch_idx + 1}/{len(test_loader)}')

    rows = []
    for label, _neuron_model_name, _checkpoint_path, _args_path in methods:
        clean_acc, clean_conf = stat_to_acc_conf(all_stats[(label, 'clean_all')])
        noise_acc, noise_conf = stat_to_acc_conf(all_stats[(label, 'noise_all')])
        cue_acc, cue_conf = stat_to_acc_conf(all_stats[(label, 'noise_cue')])
        late_acc, late_conf = stat_to_acc_conf(all_stats[(label, 'noise_late')])
        rows.append({
            'method': label,
            'clean_all_acc': clean_acc,
            'noise_all_acc': noise_acc,
            'noise_cue_acc': cue_acc,
            'noise_late_acc': late_acc,
            'noise_all_acc_drop': clean_acc - noise_acc,
            'noise_late_acc_drop': clean_acc - late_acc,
            'clean_all_true_conf': clean_conf,
            'noise_all_true_conf': noise_conf,
            'noise_cue_true_conf': cue_conf,
            'noise_late_true_conf': late_conf,
            'samples': all_stats[(label, 'clean_all')]['samples'],
        })

    write_csv(
        out_dir / 'early_cue_noise_summary.csv',
        rows,
        [
            'method', 'clean_all_acc', 'noise_all_acc', 'noise_cue_acc', 'noise_late_acc',
            'noise_all_acc_drop', 'noise_late_acc_drop', 'clean_all_true_conf', 'noise_all_true_conf',
            'noise_cue_true_conf', 'noise_late_true_conf', 'samples',
        ],
    )
    config = {
        'analysis': 'early_cue_noise_interference_eval',
        'dataset': 'DVSCIFAR10',
        'model': 'spiking_vgg11_bn',
        'methods': [method[0] for method in methods],
        'checkpoints': checkpoint_meta,
        'data_dir': str(Path(cli.data_dir).expanduser().resolve()),
        'T': cli.T,
        'cue_steps': cli.cue_steps,
        'noise_steps': cli.T - cli.cue_steps,
        'noise_type': cli.noise_type,
        'event_prob': cli.event_prob,
        'gaussian_std': cli.gaussian_std,
        'batch_size': cli.batch_size,
        'workers': cli.workers,
        'device': str(device),
        'seed': cli.seed,
    }
    (out_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    plot_accuracy(rows, figures_dir / 'early_cue_noise_accuracy.png')
    plot_drop(rows, figures_dir / 'early_cue_noise_accuracy_drop.png')
    print(f'Analysis complete. Results saved to: {out_dir}')


class NullRecorder:
    """Recorder-compatible object for run_sequence when spikes are not needed."""

    def clear(self):
        pass

    def stacked(self):
        return {}


if __name__ == '__main__':
    main()
