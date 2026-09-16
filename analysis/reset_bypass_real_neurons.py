#!/usr/bin/env python3
"""Run the repository's actual VanillaLIFNeuron and LSLIFNeuron side by side."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.neuron import LSLIFNeuron, VanillaLIFNeuron  # noqa: E402


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().reshape(-1)[0])


def run_actual_neurons(
    steps: int = 24,
    tau: float = 2.0,
    threshold: float = 1.0,
    input_amplitude: float = 0.6,
    pulse_steps: int = 16,
    history_weight: float = 0.6,
    history_power: float = 0.0,
):
    """Execute production neuron classes and return their observed states."""
    if steps < 1 or pulse_steps < 0:
        raise ValueError('steps must be positive and pulse_steps must be non-negative')
    common = dict(tau=tau, decay_input=False, v_threshold=threshold, v_reset=None)
    lif = VanillaLIFNeuron(**common)
    ls = LSLIFNeuron(
        **common,
        history_weight=history_weight,
        history_power=history_power,
    )
    # This existing opt-in flag makes production LSLIF retain its actual fused
    # pre-reset membrane for diagnostics; it does not change the forward result.
    ls.gradient_probe_enabled = True
    lif.eval()
    ls.eval()

    rows = []
    with torch.no_grad():
        for step in range(steps):
            x_value = input_amplitude if step < pulse_steps else 0.0
            x = torch.tensor([x_value], dtype=torch.float32)
            lif_spike = lif(x)
            ls_spike = ls(x)
            history_weight_t = ls._get_history_weight(ls.n.dtype, ls.n.device, ls.step_count)
            history_power_t = ls._get_history_power(ls.n.dtype, ls.n.device)
            norm = torch.pow(
                torch.as_tensor(
                    float(ls.step_count) + ls.history_eps,
                    dtype=ls.n.dtype,
                    device=ls.n.device,
                ),
                history_power_t,
            )
            branch = history_weight_t * ls.n / norm
            rows.append({
                'step': step,
                'input': x_value,
                'lif_spike': scalar(lif_spike),
                'lif_pre_reset': scalar(lif.last_v_pre),
                'lif_main_post_reset': scalar(lif.v),
                'ls_spike': scalar(ls_spike),
                'ls_effective_pre_reset': scalar(ls.last_v_pre),
                'ls_main_post_reset': scalar(ls.v),
                'ls_history_state': scalar(ls.n),
                'ls_history_branch': scalar(branch),
                'ls_effective_post_reset': scalar(ls.v + branch),
            })
    return rows


def summarize(rows):
    lif_resets = [row for row in rows if row['lif_spike'] > 0]
    ls_resets = [row for row in rows if row['ls_spike'] > 0]

    def mean(items, key):
        return sum(item[key] for item in items) / len(items) if items else None

    return {
        'implementation': {
            'lif': 'modules.neuron.VanillaLIFNeuron',
            'lslif': 'modules.neuron.LSLIFNeuron',
        },
        'lif_spike_count': len(lif_resets),
        'lslif_spike_count': len(ls_resets),
        'mean_lif_post_reset_state': mean(lif_resets, 'lif_main_post_reset'),
        'mean_lslif_main_post_reset_state': mean(ls_resets, 'ls_main_post_reset'),
        'mean_lslif_preserved_branch_at_reset': mean(ls_resets, 'ls_history_branch'),
        'mean_lslif_effective_post_reset_state': mean(ls_resets, 'ls_effective_post_reset'),
    }


def write_outputs(rows, summary, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'actual_neurons_trace.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / 'actual_neurons_summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)

    steps = [row['step'] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, [row['lif_main_post_reset'] for row in rows], label='actual LIF: main after reset')
    ax.plot(steps, [row['ls_main_post_reset'] for row in rows], label='actual LSLIF: main after reset')
    ax.plot(steps, [row['ls_history_branch'] for row in rows], ':', label='actual LSLIF: non-reset branch')
    ax.plot(steps, [row['ls_effective_post_reset'] for row in rows], label='actual LSLIF: effective state after reset')
    lif_spike_steps = [row['step'] for row in rows if row['lif_spike'] > 0]
    ls_spike_steps = [row['step'] for row in rows if row['ls_spike'] > 0]
    ax.scatter(
        lif_spike_steps,
        [rows[step]['lif_main_post_reset'] for step in lif_spike_steps],
        marker='x',
        label='LIF spike/reset',
    )
    ax.scatter(
        ls_spike_steps,
        [rows[step]['ls_effective_post_reset'] for step in ls_spike_steps],
        marker='o',
        facecolors='none',
        label='LSLIF spike/reset',
    )
    ax.set(xlabel='time step', ylabel='state', title='Production VanillaLIFNeuron vs LSLIFNeuron')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / 'actual_neurons_trace.png', dpi=180)
    plt.close(fig)
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='analysis_results/reset_bypass_real_neurons')
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--tau', type=float, default=2.0)
    parser.add_argument('--threshold', type=float, default=1.0)
    parser.add_argument('--input-amplitude', type=float, default=0.6)
    parser.add_argument('--pulse-steps', type=int, default=16)
    parser.add_argument('--history-weight', type=float, default=0.6)
    parser.add_argument('--history-power', type=float, default=0.0)
    args = parser.parse_args(argv)
    rows = run_actual_neurons(
        steps=args.steps,
        tau=args.tau,
        threshold=args.threshold,
        input_amplitude=args.input_amplitude,
        pulse_steps=args.pulse_steps,
        history_weight=args.history_weight,
        history_power=args.history_power,
    )
    summary = summarize(rows)
    write_outputs(rows, summary, Path(args.out_dir))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
