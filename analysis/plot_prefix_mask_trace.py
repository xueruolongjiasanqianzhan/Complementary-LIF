#!/usr/bin/env python3
"""Plot a prefix-mask single-neuron trace as a mechanism-focused figure."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REQUIRED_COLUMNS = (
    'time', 'input', 'masked_input',
    'lif_mem', 'lif_masked_mem', 'lif_spike', 'lif_masked_spike',
    'lslif_mem', 'lslif_masked_mem', 'lslif_spike', 'lslif_masked_spike',
)


def load_trace(path: Path):
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f'Missing required CSV columns: {missing}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'Trace CSV is empty: {path}')
    return {name: np.asarray([float(row[name]) for row in rows]) for name in REQUIRED_COLUMNS}


def infer_suffix_start(data) -> int:
    differences = np.flatnonzero(~np.isclose(data['input'], data['masked_input']))
    if differences.size == 0:
        raise ValueError('input and masked_input never differ; no prefix mask can be inferred')
    start = int(differences[-1]) + 1
    if start >= data['time'].size:
        raise ValueError('input and masked_input do not share a common suffix')
    if not np.allclose(data['input'][start:], data['masked_input'][start:]):
        raise ValueError('input and masked_input must be identical throughout the suffix')
    return start


def summarize_trace(data, suffix_start: int):
    suffix = slice(suffix_start, None)
    lif_gap = np.abs(data['lif_mem'] - data['lif_masked_mem'])
    lslif_gap = np.abs(data['lslif_mem'] - data['lslif_masked_mem'])
    lif_mismatch = ~np.isclose(data['lif_spike'], data['lif_masked_spike'])
    lslif_mismatch = ~np.isclose(data['lslif_spike'], data['lslif_masked_spike'])
    return {
        'lif_gap': lif_gap,
        'lslif_gap': lslif_gap,
        'lif_suffix_mean_gap': float(lif_gap[suffix].mean()),
        'lslif_suffix_mean_gap': float(lslif_gap[suffix].mean()),
        'lif_suffix_mismatches': int(lif_mismatch[suffix].sum()),
        'lslif_suffix_mismatches': int(lslif_mismatch[suffix].sum()),
        'suffix_steps': int(data['time'].size - suffix_start),
    }


def plot_trace(data, output: Path, dpi: int = 300):
    suffix_start = infer_suffix_start(data)
    summary = summarize_trace(data, suffix_start)
    time = data['time']
    boundary = (time[suffix_start - 1] + time[suffix_start]) / 2

    colors = {'lif': '#4C78A8', 'lslif': '#E45756', 'masked': '#6B7280'}
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1.0, 1.25, 1.35]})

    ax = axes[0]
    ax.plot(time, data['input'], color='#111827', marker='o', linewidth=1.8, markersize=4, label='Original input')
    ax.plot(time, data['masked_input'], color=colors['masked'], linestyle='--', marker='s', linewidth=1.5, markersize=3.5, label='Prefix-masked input')
    ax.set_ylabel('Input current')
    ax.set_title('(a) Counterfactual inputs: different prefix, identical suffix', loc='left', fontweight='bold')
    ax.legend(loc='lower right', ncol=2)

    ax = axes[1]
    floor = 1e-6
    ax.semilogy(time[suffix_start:], np.maximum(summary['lif_gap'][suffix_start:], floor), color=colors['lif'], marker='o', linewidth=2.2, label='LIF membrane gap')
    ax.semilogy(time[suffix_start:], np.maximum(summary['lslif_gap'][suffix_start:], floor), color=colors['lslif'], marker='o', linewidth=2.2, label='LSLIF membrane gap')
    ax.set_ylabel(r'$|V_{full}-V_{masked}|$')
    ax.set_title('(b) Prefix influence retained during the common suffix', loc='left', fontweight='bold')
    ax.grid(True, which='both', axis='y', alpha=0.25)
    ax.legend(loc='upper right')
    ratio = summary['lslif_suffix_mean_gap'] / max(summary['lif_suffix_mean_gap'], floor)
    ax.text(
        0.01, 0.06,
        f"Suffix mean gap: LIF={summary['lif_suffix_mean_gap']:.4f}, "
        f"LSLIF={summary['lslif_suffix_mean_gap']:.4f} ({ratio:.1f}x)",
        transform=ax.transAxes, fontsize=10,
        bbox={'boxstyle': 'round,pad=0.35', 'facecolor': 'white', 'edgecolor': '#D1D5DB', 'alpha': 0.92},
    )

    ax = axes[2]
    raster_rows = [
        ('LIF full', 'lif_spike', colors['lif']),
        ('LIF masked', 'lif_masked_spike', colors['masked']),
        ('LSLIF full', 'lslif_spike', colors['lslif']),
        ('LSLIF masked', 'lslif_masked_spike', '#F59E0B'),
    ]
    for y, (_label, key, color) in enumerate(raster_rows):
        spike_times = time[data[key] > 0.5]
        ax.scatter(spike_times, np.full_like(spike_times, y), marker='|', s=260, linewidths=3, color=color)
    ax.set_yticks(range(len(raster_rows)))
    ax.set_yticklabels([item[0] for item in raster_rows])
    ax.set_ylim(-0.6, len(raster_rows) - 0.4)
    ax.invert_yaxis()
    ax.set_ylabel('Spike train')
    ax.set_xlabel('Time step')
    ax.set_title('(c) Prefix-dependent changes in firing time', loc='left', fontweight='bold')
    ax.text(
        0.01, 0.04,
        f"Suffix spike mismatches: LIF={summary['lif_suffix_mismatches']}/{summary['suffix_steps']}, "
        f"LSLIF={summary['lslif_suffix_mismatches']}/{summary['suffix_steps']}",
        transform=ax.transAxes, fontsize=10,
        bbox={'boxstyle': 'round,pad=0.35', 'facecolor': 'white', 'edgecolor': '#D1D5DB', 'alpha': 0.92},
    )

    for ax in axes:
        ax.axvspan(time[0] - 0.5, boundary, color='#FEE2E2', alpha=0.32, zorder=-2)
        ax.axvspan(boundary, time[-1] + 0.5, color='#DCFCE7', alpha=0.28, zorder=-2)
        ax.axvline(boundary, color='#374151', linestyle=':', linewidth=1.3)
        ax.set_xlim(time[0] - 0.5, time[-1] + 0.5)
    axes[0].text(0.01, 0.90, 'Masked prefix', transform=axes[0].transAxes, color='#991B1B', fontweight='bold')
    axes[0].text(0.57, 0.90, 'Identical suffix', transform=axes[0].transAxes, color='#166534', fontweight='bold')

    fig.suptitle('Prefix-mask mechanism trace: LSLIF preserves more early-input influence', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    if output.suffix.lower() != '.svg':
        fig.savefig(output.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Trace CSV containing the required LIF/LSLIF columns.')
    parser.add_argument('--output', type=Path, required=True, help='Output PNG or SVG path.')
    parser.add_argument('--dpi', type=int, default=300)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    data = load_trace(args.input)
    summary = plot_trace(data, args.output, args.dpi)
    print(
        f"Wrote {args.output}; suffix mean gaps: LIF={summary['lif_suffix_mean_gap']:.6f}, "
        f"LSLIF={summary['lslif_suffix_mean_gap']:.6f}; spike mismatches: "
        f"LIF={summary['lif_suffix_mismatches']}, LSLIF={summary['lslif_suffix_mismatches']}"
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
