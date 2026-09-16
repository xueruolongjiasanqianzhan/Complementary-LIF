#!/usr/bin/env python3
"""Visualize charge preserved by LS's non-reset branch.

This is a deliberately small causal experiment rather than a task-accuracy
experiment.  Two otherwise identical leaky integrators receive the same input.
At selected steps we force a soft reset in the main membrane, while the LS
branch is left untouched.  The resulting trace directly separates reset loss
from input masking and classification effects.
"""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ToyConfig:
    steps: int = 24
    tau: float = 2.0
    threshold: float = 1.0
    history_weight: float = 0.6
    history_power: float = 0.0
    pulse_steps: int = 24
    pulse_amplitude: float = 1.4
    reset_steps: tuple = (7, 15)


def simulate(config: ToyConfig):
    """Return matched LIF/LS traces under scheduled reset interventions."""
    if config.steps < 1 or config.tau <= 0 or config.threshold <= 0:
        raise ValueError('steps, tau, and threshold must be positive')
    reset_steps = set(config.reset_steps)
    if any(step < 0 or step >= config.steps for step in reset_steps):
        raise ValueError('reset steps must lie inside [0, steps)')

    main = 0.0
    no_reset_main = 0.0
    history = 0.0
    rows = []
    for step in range(config.steps):
        x = config.pulse_amplitude if step < config.pulse_steps else 0.0
        charged = main + (x - main) / config.tau
        no_reset_main += (x - no_reset_main) / config.tau
        history += (x - history) / config.tau
        norm = (step + 1) ** config.history_power
        branch = config.history_weight * history / norm
        forced_reset = step in reset_steps
        if forced_reset and charged < config.threshold:
            raise ValueError(
                f'forced reset at step {step} is not spike-valid: charged main '
                f'{charged:.6g} is below threshold {config.threshold:.6g}; '
                'increase the pulse or move the reset step'
            )
        post_main = charged - config.threshold if forced_reset else charged
        reset_loss = charged - post_main
        lif_retention = post_main / charged if forced_reset and charged != 0 else np.nan
        ls_pre = charged + branch
        ls_post = post_main + branch
        ls_retention = ls_post / ls_pre if forced_reset and ls_pre != 0 else np.nan
        rows.append({
            'step': step,
            'input': x,
            'forced_reset': int(forced_reset),
            'main_pre_reset': charged,
            'main_post_reset': post_main,
            'main_no_reset': no_reset_main,
            'history_state': history,
            'history_branch': branch,
            'lif_effective_post': post_main,
            'ls_effective_pre': ls_pre,
            'ls_effective_post': ls_post,
            'absolute_reset_loss': reset_loss,
            'lif_retention_ratio': lif_retention,
            'ls_retention_ratio': ls_retention,
        })
        main = post_main
    return rows


def summarize(rows: Sequence[dict]):
    reset_rows = [row for row in rows if row['forced_reset']]
    if not reset_rows:
        raise ValueError('at least one reset step is required for the summary')
    lif = np.asarray([row['lif_retention_ratio'] for row in reset_rows])
    ls = np.asarray([row['ls_retention_ratio'] for row in reset_rows])
    branch = np.asarray([row['history_branch'] for row in reset_rows])
    loss = np.asarray([row['absolute_reset_loss'] for row in reset_rows])
    return {
        'reset_count': len(reset_rows),
        'mean_lif_retention_ratio': float(lif.mean()),
        'mean_ls_retention_ratio': float(ls.mean()),
        'retention_ratio_gain': float((ls - lif).mean()),
        'mean_preserved_branch_at_reset': float(branch.mean()),
        'mean_absolute_main_reset_loss': float(loss.mean()),
    }


def write_outputs(rows: Sequence[dict], config: ToyConfig, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'reset_bypass_trace.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    with (out_dir / 'reset_bypass_summary.json').open('w') as handle:
        json.dump({'config': asdict(config), 'metrics': summary}, handle, indent=2)

    import matplotlib.pyplot as plt

    steps = [row['step'] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(steps, [row['main_no_reset'] for row in rows], '--', label='main: no-reset counterfactual')
    axes[0].plot(steps, [row['lif_effective_post'] for row in rows], label='LIF: reset main')
    axes[0].plot(steps, [row['ls_effective_post'] for row in rows], label='LS: main + non-reset branch')
    axes[0].plot(steps, [row['history_branch'] for row in rows], ':', label='preserved LS branch')
    axes[0].set_ylabel('state available after reset')
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(alpha=0.25)
    reset_rows = [row for row in rows if row['forced_reset']]
    positions = np.arange(len(reset_rows))
    width = 0.36
    axes[1].bar(positions - width / 2, [row['lif_retention_ratio'] for row in reset_rows], width, label='LIF')
    axes[1].bar(positions + width / 2, [row['ls_retention_ratio'] for row in reset_rows], width, label='LS')
    axes[1].set_xticks(positions, [str(row['step']) for row in reset_rows])
    axes[1].set_xlabel('forced-reset step')
    axes[1].set_ylabel('post / pre effective state')
    axes[1].axhline(0, color='black', linewidth=0.7)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.25)
    fig.suptitle('LS preserves a reset-independent state (matched forced-reset intervention)')
    fig.tight_layout()
    fig.savefig(out_dir / 'reset_bypass_trace.png', dpi=180)
    plt.close(fig)
    return summary


def parse_reset_steps(values: Iterable[int]):
    return tuple(sorted(set(int(value) for value in values)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='analysis_results/reset_bypass_toy')
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--tau', type=float, default=2.0)
    parser.add_argument('--threshold', type=float, default=1.0)
    parser.add_argument('--history-weight', type=float, default=0.6)
    parser.add_argument('--history-power', type=float, default=0.0)
    parser.add_argument('--pulse-steps', type=int, default=24)
    parser.add_argument('--pulse-amplitude', type=float, default=1.4)
    parser.add_argument('--reset-steps', type=int, nargs='+', default=[7, 15])
    args = parser.parse_args(argv)
    config = ToyConfig(
        steps=args.steps,
        tau=args.tau,
        threshold=args.threshold,
        history_weight=args.history_weight,
        history_power=args.history_power,
        pulse_steps=args.pulse_steps,
        pulse_amplitude=args.pulse_amplitude,
        reset_steps=parse_reset_steps(args.reset_steps),
    )
    summary = write_outputs(simulate(config), config, Path(args.out_dir))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
