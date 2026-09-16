#!/usr/bin/env python3
"""Decompose real LIF/LSLIF membrane potentials by source.

The script runs the repository's production ``VanillaLIFNeuron`` and
``LSLIFNeuron`` on one deterministic input sequence.  A passive ledger tracks
current input, decayed past input, reset subtraction, and the LS branch.  The
ledger never changes neuron states and is checked against the states produced
by the real forward methods at every time step.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.neuron import LSLIFNeuron, VanillaLIFNeuron  # noqa: E402
from analysis.membrane_source_ledger import NoResetLedger, SoftResetLedger  # noqa: E402


def as_scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def default_input_sequence():
    """Deterministic charge, silence, recharge, and silence protocol."""
    return [0.6] * 6 + [0.0] * 6 + [0.6] * 4 + [0.0] * 8


def run_source_analysis(
    inputs,
    tau=2.0,
    threshold=1.0,
    history_weight=0.6,
    history_power=1.0,
    tolerance=2e-5,
):
    """Run real neurons and return an exactly reconstructed source table."""
    common = dict(tau=tau, decay_input=False, v_threshold=threshold, v_reset=None)
    lif = VanillaLIFNeuron(**common)
    lslif = LSLIFNeuron(
        **common,
        history_weight=history_weight,
        history_power=history_power,
        history_mode='all',
    )
    lslif.gradient_probe_enabled = True
    lif.eval()
    lslif.eval()

    lif_main = SoftResetLedger(1.0 - 1.0 / tau, threshold)
    ls_decay = max(0.0, min(1.0, 1.0 - 1.0 / (tau + lslif.tau_eps)))
    ls_main = SoftResetLedger(ls_decay, threshold)
    ls_history = NoResetLedger(ls_decay)
    rows = []

    with torch.no_grad():
        for step, input_value in enumerate(inputs, start=1):
            x = torch.tensor([float(input_value)], dtype=torch.float32)
            lif_main.charge(input_value)
            ls_main.charge(input_value)
            ls_history.charge(input_value)

            lif_spike = as_scalar(lif(x))
            ls_spike = as_scalar(lslif(x))
            lif_pre = as_scalar(lif.last_v_pre)
            ls_pre = as_scalar(lslif.last_v_pre)

            history_weight_t = as_scalar(
                lslif._get_history_weight(lslif.n.dtype, lslif.n.device, lslif.step_count)
            )
            history_power_t = as_scalar(
                lslif._get_history_power(lslif.n.dtype, lslif.n.device)
            )
            alpha = history_weight_t / ((step + lslif.history_eps) ** history_power_t)

            lif_reconstruction = lif_main.pre_reset_total
            ls_main_pre = ls_main.pre_reset_total
            ls_history_reconstruction = ls_history.total
            ls_reconstruction = ls_main_pre + alpha * ls_history_reconstruction

            errors = {
                'lif': abs(lif_reconstruction - lif_pre),
                'ls_main': abs(ls_main_pre - (as_scalar(lslif.v) + ls_spike * threshold)),
                'ls_history': abs(ls_history_reconstruction - as_scalar(lslif.n)),
                'ls_effective': abs(ls_reconstruction - ls_pre),
            }
            if max(errors.values()) > tolerance:
                raise RuntimeError(
                    f'source reconstruction failed at step {step}: {errors}; '
                    'the neuron update may have changed'
                )

            lif_main.reset(lif_spike)
            ls_main.reset(ls_spike)
            lif_post = as_scalar(lif.v)
            ls_effective_post = as_scalar(lslif.v) + alpha * as_scalar(lslif.n)
            post_errors = {
                'lif_post': abs(lif_main.post_reset_total - lif_post),
                'ls_post': abs(
                    ls_main.post_reset_total + alpha * ls_history.total - ls_effective_post
                ),
            }
            if max(post_errors.values()) > tolerance:
                raise RuntimeError(
                    f'post-reset reconstruction failed at step {step}: {post_errors}'
                )
            row = {
                'step': step - 1,
                'input': float(input_value),
                'lif_spike': lif_spike,
                'lif_current_input': lif_main.current_input,
                'lif_past_input': lif_main.past_input,
                'lif_reset_loss': lif_main.reset_loss,
                'lif_effective_pre_reset': lif_pre,
                'lif_main_post_reset': lif_post,
                'lslif_spike': ls_spike,
                'lslif_main_current_input': ls_main.current_input,
                'lslif_main_past_input': ls_main.past_input,
                'lslif_reset_loss': ls_main.reset_loss,
                'lslif_ls_current_input': alpha * ls_history.current_input,
                'lslif_ls_past_input': alpha * ls_history.past_input,
                'lslif_effective_pre_reset': ls_pre,
                'lslif_main_post_reset': as_scalar(lslif.v),
                'lslif_effective_post_reset': ls_effective_post,
                'lslif_history_state': as_scalar(lslif.n),
                'lslif_reconstruction_error': errors['ls_effective'],
                'lif_reconstruction_error': errors['lif'],
            }
            magnitudes = [
                abs(row['lslif_main_current_input']),
                abs(row['lslif_main_past_input']),
                abs(row['lslif_reset_loss']),
                abs(row['lslif_ls_current_input']),
                abs(row['lslif_ls_past_input']),
            ]
            magnitude_total = sum(magnitudes)
            names = ['main_current', 'main_past', 'reset_loss', 'ls_current', 'ls_past']
            for name, magnitude in zip(names, magnitudes):
                row[f'lslif_{name}_magnitude_percent'] = (
                    100.0 * magnitude / magnitude_total if magnitude_total else 0.0
                )
            rows.append(row)
    return rows


def summarize(rows):
    post_reset_silent = []
    seen_reset = False
    for row in rows:
        seen_reset = seen_reset or row['lslif_spike'] > 0
        if seen_reset and row['input'] == 0.0:
            post_reset_silent.append(row)

    def mean(key):
        if not post_reset_silent:
            return None
        return float(np.mean([row[key] for row in post_reset_silent]))

    return {
        'lif_reset_count': int(sum(row['lif_spike'] > 0 for row in rows)),
        'lslif_reset_count': int(sum(row['lslif_spike'] > 0 for row in rows)),
        'post_reset_zero_input_steps': len(post_reset_silent),
        'mean_ls_past_input_contribution_after_reset': mean('lslif_ls_past_input'),
        'mean_main_past_input_contribution_after_reset': mean('lslif_main_past_input'),
        'mean_reset_loss_contribution_after_reset': mean('lslif_reset_loss'),
        'max_lif_reconstruction_error': max(row['lif_reconstruction_error'] for row in rows),
        'max_lslif_reconstruction_error': max(row['lslif_reconstruction_error'] for row in rows),
    }


def _signed_stack(ax, steps, series, colors):
    positive_bottom = np.zeros(len(steps))
    negative_bottom = np.zeros(len(steps))
    for (label, values), color in zip(series, colors):
        values = np.asarray(values)
        positive = np.maximum(values, 0.0)
        negative = np.minimum(values, 0.0)
        ax.bar(steps, positive, bottom=positive_bottom, label=label, color=color)
        ax.bar(steps, negative, bottom=negative_bottom, color=color)
        positive_bottom += positive
        negative_bottom += negative


def plot_results(rows, output_path):
    steps = [row['step'] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    _signed_stack(
        axes[0], steps,
        [
            ('current input', [row['lif_current_input'] for row in rows]),
            ('past input in main', [row['lif_past_input'] for row in rows]),
            ('reset loss', [row['lif_reset_loss'] for row in rows]),
        ],
        ['#4C78A8', '#72B7B2', '#E45756'],
    )
    axes[0].plot(steps, [row['lif_main_post_reset'] for row in rows], 'k.-', label='actual LIF after reset')
    axes[0].set_title('LIF: signed membrane-source decomposition')
    axes[0].set_ylabel('potential contribution')
    axes[0].legend(ncol=4, fontsize=8)

    _signed_stack(
        axes[1], steps,
        [
            ('main: current input', [row['lslif_main_current_input'] for row in rows]),
            ('main: past input', [row['lslif_main_past_input'] for row in rows]),
            ('main: reset loss', [row['lslif_reset_loss'] for row in rows]),
            ('LS: current input', [row['lslif_ls_current_input'] for row in rows]),
            ('LS: past input', [row['lslif_ls_past_input'] for row in rows]),
        ],
        ['#4C78A8', '#72B7B2', '#E45756', '#F2CF5B', '#54A24B'],
    )
    axes[1].plot(steps, [row['lslif_effective_post_reset'] for row in rows], 'k.-', label='actual LSLIF after reset')
    axes[1].set_title('LSLIF: reset loss is separate from LS-retained past input')
    axes[1].set_ylabel('potential contribution')
    axes[1].legend(ncol=3, fontsize=8)

    percent_keys = [
        ('main current', 'lslif_main_current_magnitude_percent'),
        ('main past', 'lslif_main_past_magnitude_percent'),
        ('reset loss', 'lslif_reset_loss_magnitude_percent'),
        ('LS current', 'lslif_ls_current_magnitude_percent'),
        ('LS past', 'lslif_ls_past_magnitude_percent'),
    ]
    axes[2].stackplot(
        steps,
        *[[row[key] for row in rows] for _, key in percent_keys],
        labels=[label for label, _ in percent_keys],
        colors=['#4C78A8', '#72B7B2', '#E45756', '#F2CF5B', '#54A24B'],
    )
    axes[2].set_title('LSLIF source magnitude percentages (absolute contributions)')
    axes[2].set(xlabel='time step', ylabel='absolute contribution (%)', ylim=(0, 100))
    axes[2].legend(ncol=5, fontsize=8, loc='upper center')
    for ax in axes:
        ax.axhline(0.0, color='black', linewidth=0.7)
        ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_csv(rows, path):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='analysis_results/membrane_source_attribution')
    parser.add_argument('--tau', type=float, default=2.0)
    parser.add_argument('--threshold', type=float, default=1.0)
    parser.add_argument('--history-weight', type=float, default=0.6)
    parser.add_argument('--history-power', type=float, default=1.0)
    parser.add_argument(
        '--inputs', type=float, nargs='+',
        help='Deterministic input values; defaults to charge/silence/recharge/silence.',
    )
    args = parser.parse_args(argv)
    inputs = args.inputs if args.inputs is not None else default_input_sequence()
    if not inputs:
        raise ValueError('input sequence must not be empty')
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_source_analysis(
        inputs,
        tau=args.tau,
        threshold=args.threshold,
        history_weight=args.history_weight,
        history_power=args.history_power,
    )
    result = {
        'configuration': {
            'inputs': inputs,
            'tau': args.tau,
            'threshold': args.threshold,
            'history_weight': args.history_weight,
            'history_power': args.history_power,
            'reset': 'natural spike-triggered soft reset',
            'percentage_definition': 'absolute contribution magnitude',
        },
        'summary': summarize(rows),
    }
    write_csv(rows, output_dir / 'membrane_source_trace.csv')
    with (output_dir / 'membrane_source_summary.json').open('w') as handle:
        json.dump(result, handle, indent=2)
    plot_results(rows, output_dir / 'membrane_source_attribution.png')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
