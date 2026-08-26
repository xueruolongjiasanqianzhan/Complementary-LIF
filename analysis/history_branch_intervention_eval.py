#!/usr/bin/env python3
"""Evaluate a trained LSLIF checkpoint under history-branch interventions.

This first-stage experiment does not train or modify checkpoint weights. It
compares normal inference with zeroed, batch-shuffled, and time-shifted history
terms on exactly the same DVS-CIFAR10 samples.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, ANALYSIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from modules.neuron import LSLIFNeuron  # noqa: E402
from prefix_mask_multi_neuron_suffix_similarity import (  # noqa: E402
    build_vgg11_multi,
    sync_history_flags_from_checkpoint,
)
from prefix_mask_spike_similarity import (  # noqa: E402
    build_test_loader,
    get_checkpoint_state,
    load_checkpoint,
    load_namespace_args,
    normalize_frames,
    require_file,
    run_sequence,
    write_csv,
)


class NullRecorder:
    def clear(self):
        pass

    def stacked(self):
        return {}


def parse_condition(text: str):
    """Return canonical label, neuron mode, and shift from a CLI condition."""
    value = text.strip().lower().replace('-', '_')
    if value in {'normal', 'zero', 'shuffle'}:
        return value, value, 1
    if value.startswith('time_shift_'):
        shift = int(value.rsplit('_', 1)[1])
        if shift < 1:
            raise ValueError(f'time shift must be >= 1, got {shift}')
        return f'time_shift_{shift}', 'time_shift', shift
    raise ValueError(
        f'Unsupported condition {text!r}; use normal, zero, shuffle, or time_shift_N'
    )


def configure_intervention(model: torch.nn.Module, mode: str, shift: int) -> int:
    count = 0
    for module in model.modules():
        if type(module) is LSLIFNeuron:
            module.set_history_intervention(mode, shift)
            count += 1
    if count == 0:
        raise ValueError('No standard LSLIFNeuron layers were found in the model')
    return count


def sample_metrics(logits: torch.Tensor, labels: torch.Tensor):
    summed = logits.sum(dim=0)
    labels = labels.detach().cpu().long()
    probabilities = torch.softmax(summed, dim=1)
    predictions = summed.argmax(dim=1)
    nll = F.cross_entropy(summed, labels, reduction='none')
    indices = torch.arange(labels.numel())
    return {
        'predictions': predictions,
        'correct': predictions.eq(labels),
        'nll': nll,
        'true_confidence': probabilities[indices, labels],
    }


def init_stats() -> Dict[str, float]:
    return {
        'samples': 0,
        'correct': 0,
        'nll_sum': 0.0,
        'true_confidence_sum': 0.0,
        'prediction_changes': 0,
        'normal_correct_to_wrong': 0,
        'normal_wrong_to_correct': 0,
    }


def update_stats(stats: dict, current: dict, normal: dict):
    samples = current['predictions'].numel()
    stats['samples'] += samples
    stats['correct'] += int(current['correct'].sum())
    stats['nll_sum'] += float(current['nll'].sum())
    stats['true_confidence_sum'] += float(current['true_confidence'].sum())
    changed = current['predictions'].ne(normal['predictions'])
    stats['prediction_changes'] += int(changed.sum())
    stats['normal_correct_to_wrong'] += int((normal['correct'] & ~current['correct']).sum())
    stats['normal_wrong_to_correct'] += int((~normal['correct'] & current['correct']).sum())


def build_parser():
    parser = argparse.ArgumentParser(
        description='History-branch intervention evaluation for a trained DVS-CIFAR10 LSLIF VGG11.'
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--args', required=True, dest='args_path')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--T', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument(
        '--conditions', nargs='+',
        default=['normal', 'zero', 'shuffle', 'time_shift_1', 'time_shift_2', 'time_shift_4'],
        help='normal zero shuffle and/or time_shift_N; normal is always required.',
    )
    return parser


def main(argv=None):
    cli = build_parser().parse_args(argv)
    if cli.T < 1:
        raise ValueError(f'T must be positive, got {cli.T}')
    if cli.batch_size < 2 and any(parse_condition(item)[1] == 'shuffle' for item in cli.conditions):
        raise ValueError('shuffle condition requires --batch-size >= 2')
    parsed_conditions = [parse_condition(item) for item in cli.conditions]
    labels = [item[0] for item in parsed_conditions]
    if len(labels) != len(set(labels)):
        raise ValueError(f'Duplicate conditions are not allowed: {labels}')
    if 'normal' not in labels:
        raise ValueError('--conditions must include normal')

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)

    checkpoint_path = require_file(cli.checkpoint, 'checkpoint')
    args_path = require_file(cli.args_path, 'args.txt')
    out_dir = Path(cli.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cli.device)

    checkpoint, state_dict = get_checkpoint_state(checkpoint_path, device)
    train_args = load_namespace_args(args_path)
    train_args.T = cli.T
    train_args.b = cli.batch_size
    train_args = sync_history_flags_from_checkpoint(train_args, state_dict, 'LSLIF')
    model = build_vgg11_multi(train_args, 'LSLIF', device)
    load_checkpoint(model, state_dict)
    model.eval()
    loader = build_test_loader(cli.data_dir, cli.T, cli.batch_size, cli.workers)
    stats = {label: init_stats() for label in labels}
    layer_count = 0

    for batch_idx, (frame, targets) in enumerate(loader):
        frames = normalize_frames(frame, cli.T)
        batch_results = {}
        # Normal is evaluated first and becomes the paired per-sample reference.
        ordered = sorted(parsed_conditions, key=lambda item: item[0] != 'normal')
        for label, mode, shift in ordered:
            layer_count = configure_intervention(model, mode, shift)
            _spikes, logits = run_sequence(model, NullRecorder(), frames, 0, device)
            batch_results[label] = sample_metrics(logits, targets)
        normal = batch_results['normal']
        for label in labels:
            update_stats(stats[label], batch_results[label], normal)
        if batch_idx % 20 == 0:
            print(f'processed batch {batch_idx + 1}/{len(loader)}')

    normal_accuracy = stats['normal']['correct'] / max(stats['normal']['samples'], 1)
    rows: List[dict] = []
    for label, _mode, shift in parsed_conditions:
        item = stats[label]
        samples = max(item['samples'], 1)
        accuracy = item['correct'] / samples
        rows.append({
            'condition': label,
            'shift_steps': shift if label.startswith('time_shift_') else 0,
            'samples': item['samples'],
            'accuracy': accuracy,
            'accuracy_drop_vs_normal': normal_accuracy - accuracy,
            'mean_nll': item['nll_sum'] / samples,
            'mean_true_confidence': item['true_confidence_sum'] / samples,
            'prediction_change_rate': item['prediction_changes'] / samples,
            'normal_correct_to_wrong_rate': item['normal_correct_to_wrong'] / samples,
            'normal_wrong_to_correct_rate': item['normal_wrong_to_correct'] / samples,
        })

    fields = [
        'condition', 'shift_steps', 'samples', 'accuracy', 'accuracy_drop_vs_normal',
        'mean_nll', 'mean_true_confidence', 'prediction_change_rate',
        'normal_correct_to_wrong_rate', 'normal_wrong_to_correct_rate',
    ]
    write_csv(out_dir / 'history_branch_intervention_summary.csv', rows, fields)
    config = {
        'analysis': 'history_branch_intervention_eval',
        'dataset': 'DVSCIFAR10',
        'model': 'spiking_vgg11_bn',
        'neuron_model': 'LSLIF',
        'checkpoint': str(checkpoint_path),
        'args': str(args_path),
        'checkpoint_epoch': checkpoint.get('epoch') if isinstance(checkpoint, dict) else None,
        'conditions': labels,
        'intervened_layers': layer_count,
        'T': cli.T,
        'batch_size': cli.batch_size,
        'workers': cli.workers,
        'device': str(device),
        'seed': cli.seed,
    }
    (out_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Analysis complete. Results saved to: {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
