#!/usr/bin/env python3
"""Standalone pre-threshold membrane provenance experiment for LIF/LSLIF.

This file contains the scalar LIF/LSLIF dynamics, proportional source ledger,
plotting, and CLI.  It needs no dataset, checkpoint, or repository-local Python
module.  After a soft reset, the residual main membrane is proportionally
reassigned to its pre-reset input sources; the LS source ledger is never reset.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import FuncNorm, LogNorm
import numpy as np


class ProportionalSourceLedger:
    """Track membrane provenance by input time using proportional reset."""

    def __init__(self, decay):
        self.decay = float(decay)
        self.sources = []

    def charge(self, current_input):
        self.sources = [value * self.decay for value in self.sources]
        self.sources.append(float(current_input))

    def redistribute(self, target_total, eps=1e-12):
        source_total = self.total
        target_total = float(target_total)
        if abs(source_total) <= eps:
            if abs(target_total) <= eps:
                self.sources = [0.0 for _ in self.sources]
                return
            raise ValueError(
                f'cannot proportionally assign nonzero membrane {target_total} '
                'from zero source total'
            )
        scale = target_total / source_total
        self.sources = [value * scale for value in self.sources]

    @property
    def total(self):
        return sum(self.sources)


def default_input_sequence():
    """Two deterministic charge/reset events followed by silent observation."""
    return [1.2, 0.9, 0.0, 0.0, 1.2, 0.9, 0.0, 0.0, 0.0, 0.0]


def _percent(part, total, eps=1e-12):
    return 100.0 * part / total if abs(total) > eps else 0.0


def run_source_analysis(
    inputs,
    tau=2.0,
    threshold=1.0,
    history_weight=0.6,
    history_power=1.0,
    tolerance=2e-5,
):
    """Run standalone scalar neurons and attribute every threshold decision."""
    if tau <= 1.0:
        raise ValueError(f'tau must be greater than 1 for stable leakage, got {tau}')
    if threshold <= 0.0:
        raise ValueError(f'threshold must be positive, got {threshold}')
    decay = 1.0 - 1.0 / tau
    lif_sources = ProportionalSourceLedger(decay)
    ls_decay = decay
    ls_main_sources = ProportionalSourceLedger(ls_decay)
    ls_branch_sources = ProportionalSourceLedger(ls_decay)
    lif_v = 0.0
    ls_main_v = 0.0
    ls_history_v = 0.0
    rows = []
    source_rows = []

    for step, input_value in enumerate(inputs, start=1):
        input_value = float(input_value)
        lif_sources.charge(input_value)
        ls_main_sources.charge(input_value)
        ls_branch_sources.charge(input_value)

        lif_pre = decay * lif_v + input_value
        lif_spike = float(lif_pre >= threshold)
        lif_v = lif_pre - lif_spike * threshold
        ls_main_pre = ls_decay * ls_main_v + input_value
        ls_history_v = ls_decay * ls_history_v + input_value
        alpha = history_weight / (step ** history_power)
        ls_pre = ls_main_pre + alpha * ls_history_v
        ls_spike = float(ls_pre >= threshold)
        ls_main_v = ls_main_pre - ls_spike * threshold

        lif_values = list(lif_sources.sources)
        ls_main_values = list(ls_main_sources.sources)
        ls_branch_values = [alpha * value for value in ls_branch_sources.sources]
        ls_total_values = [
            main_value + branch_value
            for main_value, branch_value in zip(ls_main_values, ls_branch_values)
        ]
        lif_error = abs(sum(lif_values) - lif_pre)
        ls_error = abs(sum(ls_total_values) - ls_pre)
        branch_error = abs(ls_branch_sources.total - ls_history_v)
        if max(lif_error, ls_error, branch_error) > tolerance:
            raise RuntimeError(
                f'pre-threshold source reconstruction failed at step {step - 1}: '
                f'lif={lif_error}, lslif={ls_error}, ls_branch={branch_error}'
            )

        lif_current = lif_values[-1]
        lif_history = sum(lif_values[:-1])
        ls_main_current = ls_main_values[-1]
        ls_main_history = sum(ls_main_values[:-1])
        ls_branch_current = ls_branch_values[-1]
        ls_branch_history = sum(ls_branch_values[:-1])
        ls_current = ls_main_current + ls_branch_current
        ls_history = ls_main_history + ls_branch_history
        row = {
                'step': step - 1,
                'input': float(input_value),
                'threshold': float(threshold),
                'lif_spike': lif_spike,
                'lif_pre_threshold_membrane': lif_pre,
                'lif_current_input_contribution': lif_current,
                'lif_historical_input_contribution': lif_history,
                'lif_current_input_percent': _percent(lif_current, lif_pre),
                'lif_historical_input_percent': _percent(lif_history, lif_pre),
                'lslif_spike': ls_spike,
                'lslif_pre_threshold_membrane': ls_pre,
                'lslif_current_input_contribution': ls_current,
                'lslif_main_historical_contribution': ls_main_history,
                'lslif_ls_historical_contribution': ls_branch_history,
                'lslif_historical_input_contribution': ls_history,
                'lslif_current_input_percent': _percent(ls_current, ls_pre),
                'lslif_main_historical_percent': _percent(ls_main_history, ls_pre),
                'lslif_ls_historical_percent': _percent(ls_branch_history, ls_pre),
                'lslif_historical_input_percent': _percent(ls_history, ls_pre),
                'historical_share_advantage_percentage_points': (
                    _percent(ls_history, ls_pre) - _percent(lif_history, lif_pre)
                ),
                'lif_without_history': lif_current,
                'lslif_without_history': ls_current,
                'lslif_without_ls_history': ls_pre - ls_branch_history,
                'lif_reconstruction_error': lif_error,
                'lslif_reconstruction_error': ls_error,
        }
        rows.append(row)

        for source_step in range(step):
            source_rows.append({
                    'decision_step': step - 1,
                    'source_step': source_step,
                    'lif_contribution': lif_values[source_step],
                    'lslif_main_contribution': ls_main_values[source_step],
                    'lslif_ls_contribution': ls_branch_values[source_step],
                    'lslif_total_contribution': ls_total_values[source_step],
                    'lif_percent': _percent(lif_values[source_step], lif_pre),
                    'lslif_percent': _percent(ls_total_values[source_step], ls_pre),
            })

        lif_sources.redistribute(lif_v)
        ls_main_sources.redistribute(ls_main_v)
        if abs(lif_sources.total - lif_v) > tolerance:
            raise RuntimeError(f'LIF post-reset attribution failed at step {step - 1}')
        if abs(ls_main_sources.total - ls_main_v) > tolerance:
            raise RuntimeError(f'LSLIF post-reset attribution failed at step {step - 1}')
    return rows, source_rows


def summarize(rows):
    spike_rows = [row for row in rows if row['lslif_spike'] > 0]
    historical_spike_rows = [
        row for row in spike_rows if abs(row['lslif_historical_input_contribution']) > 1e-12
    ]
    ls_history_required = [
        row for row in spike_rows
        if row['lslif_without_ls_history'] < row['threshold']
    ]

    def mean(items, key):
        return float(np.mean([row[key] for row in items])) if items else None

    return {
        'lif_spike_count': int(sum(row['lif_spike'] > 0 for row in rows)),
        'lslif_spike_count': len(spike_rows),
        'lslif_spikes_with_historical_input': len(historical_spike_rows),
        'lslif_spikes_that_cross_threshold_due_to_ls_history': len(ls_history_required),
        'mean_lif_historical_percent_at_lslif_spike_steps': mean(
            spike_rows, 'lif_historical_input_percent'
        ),
        'mean_lslif_historical_percent_at_spike': mean(
            spike_rows, 'lslif_historical_input_percent'
        ),
        'mean_lslif_ls_historical_percent_at_spike': mean(
            spike_rows, 'lslif_ls_historical_percent'
        ),
        'max_lif_reconstruction_error': max(row['lif_reconstruction_error'] for row in rows),
        'max_lslif_reconstruction_error': max(row['lslif_reconstruction_error'] for row in rows),
    }


def _source_matrix(source_rows, key, steps, absolute=False):
    matrix = np.full((steps, steps), np.nan)
    for row in source_rows:
        value = row[key]
        matrix[row['source_step'], row['decision_step']] = abs(value) if absolute else value
    return matrix


def _mirrored_power_norm(vmax, gamma):
    """Expand high-value color resolution after mapping large values dark."""
    vmax = float(vmax)

    def forward(values):
        unit = np.clip(np.asarray(values) / vmax, 0.0, 1.0)
        return 1.0 - np.power(1.0 - unit, gamma)

    def inverse(values):
        unit = np.clip(np.asarray(values), 0.0, 1.0)
        return vmax * (1.0 - np.power(1.0 - unit, 1.0 / gamma))

    return FuncNorm((forward, inverse), vmin=0.0, vmax=vmax)


def _log_heatmap_data(lif_matrix, ls_matrix, decades):
    """Return shared log-scale matrices spanning ``decades`` below the maximum."""
    if decades <= 0:
        raise ValueError(f'log heatmap decades must be positive, got {decades}')
    common_max = max(np.nanmax(lif_matrix), np.nanmax(ls_matrix), 1.0)
    vmin = common_max * (10.0 ** -float(decades))

    def clipped(matrix):
        return np.where(np.isnan(matrix), np.nan, np.maximum(matrix, vmin))

    return clipped(lif_matrix), clipped(ls_matrix), LogNorm(vmin=vmin, vmax=common_max)


def plot_log_heatmap_variants(rows, source_rows, output_dir, decades_options=(2, 3, 4, 5),
                              cmap_name='viridis_r', alternate_cmap='magma_r'):
    """Write log-normalized source heatmaps plus one alternate-color candidate."""
    lif_matrix = _source_matrix(source_rows, 'lif_percent', len(rows), absolute=True)
    ls_matrix = _source_matrix(source_rows, 'lslif_percent', len(rows), absolute=True)
    output_dir = Path(output_dir)

    variants = [(int(decades), cmap_name) for decades in decades_options]
    if alternate_cmap:
        reference_decades = int(decades_options[len(decades_options) // 2])
        variants.append((reference_decades, alternate_cmap))

    for decades, variant_cmap in variants:
        lif_plot, ls_plot, norm = _log_heatmap_data(lif_matrix, ls_matrix, decades)
        cmap = plt.get_cmap(variant_cmap).copy()
        cmap.set_bad('white')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        image_lif = axes[0].imshow(lif_plot, origin='lower', aspect='auto', norm=norm, cmap=cmap)
        axes[0].set(title='LIF: contribution by input source time', xlabel='decision step', ylabel='input source step')
        axes[1].imshow(ls_plot, origin='lower', aspect='auto', norm=norm, cmap=cmap)
        axes[1].set(title='LSLIF: contribution by input source time', xlabel='decision step', ylabel='input source step')
        for ax in axes:
            ax.grid(alpha=0.18)
        fig.colorbar(
            image_lif,
            ax=axes,
            label=f'absolute share of pre-threshold membrane (%)\nlog scale, {decades} decades',
            fraction=0.025,
            pad=0.04,
        )
        fig.subplots_adjust(left=0.08, right=0.88, bottom=0.16, top=0.88, wspace=0.28)
        safe_cmap = variant_cmap.replace('/', '_')
        fig.savefig(output_dir / f'membrane_source_heatmap_log_{decades}dec_{safe_cmap}.png', dpi=180)
        plt.close(fig)


def plot_results(rows, source_rows, output_path, heatmap_gamma=0.35):
    if heatmap_gamma <= 0.0:
        raise ValueError(f'heatmap_gamma must be positive, got {heatmap_gamma}')
    steps = np.asarray([row['step'] for row in rows])
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex='col')

    axes[0, 0].plot(steps, [row['lif_pre_threshold_membrane'] for row in rows], '.-', label='LIF')
    axes[0, 0].plot(steps, [row['lslif_pre_threshold_membrane'] for row in rows], '.-', label='LSLIF')
    axes[0, 0].axhline(rows[0]['threshold'], color='black', linestyle='--', label='threshold')
    axes[0, 0].set(title='Actual pre-threshold decision membranes', ylabel='membrane')
    axes[0, 0].legend()

    width = 0.38
    axes[0, 1].bar(
        steps - width / 2,
        [row['lif_historical_input_percent'] for row in rows],
        width,
        label='LIF history',
    )
    axes[0, 1].bar(
        steps + width / 2,
        [row['lslif_historical_input_percent'] for row in rows],
        width,
        label='LSLIF history',
    )
    axes[0, 1].set(title='Historical share of the current decision membrane', ylabel='percent')
    axes[0, 1].legend()

    axes[1, 0].stackplot(
        steps,
        [row['lif_current_input_percent'] for row in rows],
        [row['lif_historical_input_percent'] for row in rows],
        labels=['current input', 'historical input'],
        colors=['#4C78A8', '#72B7B2'],
    )
    axes[1, 0].set(title='LIF decision-membrane provenance', ylabel='percent', ylim=(0, 100))
    axes[1, 0].legend(loc='upper right')

    axes[1, 1].stackplot(
        steps,
        [row['lslif_current_input_percent'] for row in rows],
        [row['lslif_main_historical_percent'] for row in rows],
        [row['lslif_ls_historical_percent'] for row in rows],
        labels=['current input', 'history in main', 'history retained by LS'],
        colors=['#4C78A8', '#72B7B2', '#54A24B'],
    )
    axes[1, 1].set(title='LSLIF decision-membrane provenance', ylabel='percent', ylim=(0, 100))
    axes[1, 1].legend(loc='upper right')

    # Signed source attributions are retained in the CSV, while the heatmaps
    # visualize their magnitudes so negative contributions are not clipped by
    # the zero-based color normalization.
    lif_matrix = _source_matrix(source_rows, 'lif_percent', len(rows), absolute=True)
    ls_matrix = _source_matrix(source_rows, 'lslif_percent', len(rows), absolute=True)
    common_max = max(np.nanmax(lif_matrix), np.nanmax(ls_matrix), 1.0)
    # Mirror the original power curve together with the palette: the nonlinear
    # resolution is now concentrated at the dark, high-magnitude end rather
    # than retaining the old light, low-value emphasis after the color reversal.
    heatmap_norm = _mirrored_power_norm(common_max, heatmap_gamma)
    heatmap_cmap = plt.get_cmap('viridis_r').copy()
    heatmap_cmap.set_bad('white')
    image_lif = axes[2, 0].imshow(
        lif_matrix,
        origin='lower',
        aspect='auto',
        norm=heatmap_norm,
        cmap=heatmap_cmap,
    )
    axes[2, 0].set(title='LIF: contribution by input source time', xlabel='decision step', ylabel='input source step')
    axes[2, 1].imshow(
        ls_matrix,
        origin='lower',
        aspect='auto',
        norm=heatmap_norm,
        cmap=heatmap_cmap,
    )
    axes[2, 1].set(title='LSLIF: contribution by input source time', xlabel='decision step', ylabel='input source step')
    colorbar_axis = fig.add_axes([0.90, 0.08, 0.012, 0.20])
    fig.colorbar(
        image_lif,
        cax=colorbar_axis,
        label=f'absolute share of pre-threshold membrane (%)\nmirrored power scale, γ={heatmap_gamma:g}',
    )
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    fig.suptitle('Proportional input provenance before each threshold decision')
    fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.94, right=0.87)
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
        '--heatmap-gamma',
        type=float,
        default=0.35,
        help='Mirrored power-law exponent; smaller values emphasize high percentages.',
    )
    parser.add_argument(
        '--heatmap-log-decades',
        type=int,
        nargs='+',
        default=[2, 3, 4, 5],
        help='Log-scale candidate ranges below the shared maximum.',
    )
    parser.add_argument('--heatmap-log-cmap', default='viridis_r')
    parser.add_argument('--heatmap-log-alternate-cmap', default='magma_r')
    parser.add_argument('--inputs', type=float, nargs='+')
    args = parser.parse_args(argv)
    inputs = args.inputs if args.inputs is not None else default_input_sequence()
    if not inputs:
        raise ValueError('input sequence must not be empty')
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, source_rows = run_source_analysis(
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
            'heatmap_gamma': args.heatmap_gamma,
            'heatmap_log_decades': args.heatmap_log_decades,
            'heatmap_log_cmap': args.heatmap_log_cmap,
            'heatmap_log_alternate_cmap': args.heatmap_log_alternate_cmap,
            'reset': 'natural spike-triggered soft reset',
            'attribution': 'proportional redistribution of residual main membrane',
        },
        'summary': summarize(rows),
    }
    write_csv(rows, output_dir / 'membrane_source_trace.csv')
    write_csv(source_rows, output_dir / 'membrane_source_by_input_time.csv')
    with (output_dir / 'membrane_source_summary.json').open('w') as handle:
        json.dump(result, handle, indent=2)
    plot_results(
        rows,
        source_rows,
        output_dir / 'membrane_source_attribution.png',
        heatmap_gamma=args.heatmap_gamma,
    )
    plot_log_heatmap_variants(
        rows,
        source_rows,
        output_dir,
        decades_options=args.heatmap_log_decades,
        cmap_name=args.heatmap_log_cmap,
        alternate_cmap=args.heatmap_log_alternate_cmap,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
