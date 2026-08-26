#!/usr/bin/env python3
"""Plot the summary CSV produced by history_branch_intervention_eval.py."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REQUIRED_COLUMNS = (
    'condition',
    'samples',
    'accuracy',
    'accuracy_drop_vs_normal',
    'mean_nll',
    'mean_true_confidence',
    'prediction_change_rate',
    'normal_correct_to_wrong_rate',
    'normal_wrong_to_correct_rate',
)


def load_summary(path: Path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        fieldnames = [str(name).strip() for name in (reader.fieldnames or [])]
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(
                f'Missing required history-intervention columns: {missing}. '
                f'Detected columns: {fieldnames}. Expected the '
                'history_branch_intervention_summary.csv produced by '
                'analysis/history_branch_intervention_eval.py.'
            )
        rows = [
            {str(key).strip(): value for key, value in row.items() if key is not None}
            for row in reader
        ]
    if not rows:
        raise ValueError(f'History-intervention summary is empty: {path}')
    conditions = [row['condition'].strip() for row in rows]
    if len(conditions) != len(set(conditions)):
        raise ValueError(f'Duplicate conditions in summary CSV: {conditions}')
    if 'normal' not in conditions:
        raise ValueError("Summary CSV must contain the 'normal' reference condition")
    return rows


def numeric(rows, key: str) -> np.ndarray:
    try:
        return np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Column {key!r} must contain numeric values') from exc


def annotate_bars(ax, bars, values, fmt='{:.3f}'):
    for bar, value in zip(bars, values):
        y = bar.get_height()
        offset = 4 if y >= 0 else -12
        va = 'bottom' if y >= 0 else 'top'
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, y),
            xytext=(0, offset),
            textcoords='offset points',
            ha='center',
            va=va,
            fontsize=8,
        )


def plot_summary(rows, output: Path, dpi: int = 300):
    conditions = [row['condition'].strip() for row in rows]
    x = np.arange(len(rows))
    accuracy = numeric(rows, 'accuracy')
    accuracy_drop = numeric(rows, 'accuracy_drop_vs_normal')
    nll = numeric(rows, 'mean_nll')
    confidence = numeric(rows, 'mean_true_confidence')
    prediction_change = numeric(rows, 'prediction_change_rate')
    correct_to_wrong = numeric(rows, 'normal_correct_to_wrong_rate')
    wrong_to_correct = numeric(rows, 'normal_wrong_to_correct_rate')

    colors = ['#4C78A8' if name == 'normal' else '#E45756' for name in conditions]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    bars = ax.bar(x, accuracy, color=colors, edgecolor='white')
    annotate_bars(ax, bars, accuracy)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.0, min(1.05, max(accuracy.max() + 0.12, 0.2)))
    ax.set_title('(a) Classification accuracy', loc='left', fontweight='bold')
    ax.grid(axis='y', alpha=0.25)

    ax = axes[0, 1]
    bars = ax.bar(x, accuracy_drop, color=colors, edgecolor='white')
    annotate_bars(ax, bars, accuracy_drop)
    ax.axhline(0.0, color='#111827', linewidth=1)
    ax.set_ylabel('Normal accuracy − intervention accuracy')
    ax.set_title('(b) Accuracy degradation caused by intervention', loc='left', fontweight='bold')
    ax.grid(axis='y', alpha=0.25)

    ax = axes[1, 0]
    width = 0.25
    bars_change = ax.bar(x - width, prediction_change, width, label='Prediction changed', color='#8B5CF6')
    bars_harm = ax.bar(x, correct_to_wrong, width, label='Correct → wrong', color='#DC2626')
    bars_help = ax.bar(x + width, wrong_to_correct, width, label='Wrong → correct', color='#16A34A')
    annotate_bars(ax, bars_change, prediction_change)
    annotate_bars(ax, bars_harm, correct_to_wrong)
    annotate_bars(ax, bars_help, wrong_to_correct)
    ax.set_ylabel('Fraction of test samples')
    ax.set_title('(c) Per-sample prediction transitions vs normal', loc='left', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.25)

    ax = axes[1, 1]
    width = 0.36
    bars_conf = ax.bar(x - width / 2, confidence, width, label='True-class confidence', color='#F59E0B')
    ax.set_ylabel('Mean true-class confidence', color='#92400E')
    ax.tick_params(axis='y', labelcolor='#92400E')
    ax.set_ylim(0.0, min(1.05, max(confidence.max() + 0.12, 0.2)))
    ax2 = ax.twinx()
    bars_nll = ax2.bar(x + width / 2, nll, width, label='NLL', color='#64748B')
    ax2.set_ylabel('Mean NLL', color='#334155')
    ax2.tick_params(axis='y', labelcolor='#334155')
    ax.set_title('(d) Confidence and negative log-likelihood', loc='left', fontweight='bold')
    ax.legend([bars_conf, bars_nll], ['True-class confidence', 'NLL'], fontsize=8, loc='best')

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=25, ha='right')
    fig.suptitle(
        'LSLIF history-branch interventions on one trained checkpoint',
        fontsize=16,
        fontweight='bold',
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    if output.suffix.lower() != '.svg':
        fig.savefig(output.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='history_branch_intervention_summary.csv from history_branch_intervention_eval.py',
    )
    parser.add_argument('--output', type=Path, required=True, help='Output PNG or SVG path.')
    parser.add_argument('--dpi', type=int, default=300)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = load_summary(args.input)
    plot_summary(rows, args.output, args.dpi)
    print(f'Wrote history-intervention figure: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
