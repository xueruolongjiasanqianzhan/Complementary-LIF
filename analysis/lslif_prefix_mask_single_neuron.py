#!/usr/bin/env python3
"""Single-neuron prefix-mask comparison for LIF and LSLIF.

The experiment feeds one non-stationary input sequence to a vanilla LIF neuron and
an LSLIF neuron, masks the input prefix, and visualizes how much the later
membrane trajectory and spike train change.  The script writes an SVG figure with
input curves, membrane curves, and spike rasters, using only Python's standard
library so it can run in minimal environments.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path


def build_input(time_steps: int, seed: int) -> list[float]:
    """Create a non-stationary 1-D input current with mild noise and pulses."""
    rng = random.Random(seed)
    x: list[float] = []
    for step in range(time_steps):
        baseline = 0.28 + 0.10 * math.sin(2.0 * math.pi * step / max(time_steps, 1) * 2.7)
        slow_ramp = 0.16 * step / max(time_steps - 1, 1)
        noise = rng.gauss(0.0, 0.035)
        x.append(max(0.0, baseline + slow_ramp + noise))

    pulse_centers = [max(2, time_steps // 8), time_steps // 3, time_steps // 2, (3 * time_steps) // 4]
    for width, center, amp in zip([2, 3, 2, 3], pulse_centers, [0.55, 0.35, 0.45, 0.40]):
        start = max(0, center - width // 2)
        end = min(time_steps, start + width)
        for idx in range(start, end):
            x[idx] += amp
    return x


def simulate_lif(x: list[float], tau: float, threshold: float) -> dict[str, list[float]]:
    decay = 1.0 - 1.0 / tau
    v = 0.0
    pre_mem, post_mem, spikes = [], [], []
    for current in x:
        m_t = v * decay + current
        spike = 1.0 if m_t >= threshold else 0.0
        v = m_t - spike * threshold
        pre_mem.append(m_t)
        post_mem.append(v)
        spikes.append(spike)
    return {"decision_mem": pre_mem, "post_reset_mem": post_mem, "spikes": spikes}


def simulate_lslif(
    x: list[float],
    tau: float,
    threshold: float,
    history_weight: float,
    history_power: float,
    history_eps: float,
) -> dict[str, list[float]]:
    decay = 1.0 - 1.0 / tau
    v = 0.0
    n = 0.0
    decision_mem, primary_mem, history_mem, post_mem, spikes = [], [], [], [], []
    for step, current in enumerate(x, start=1):
        m_t = v * decay + current
        n_t = n * decay + current
        history_term = history_weight * n_t / ((step + history_eps) ** history_power)
        total_mem = m_t + history_term
        spike = 1.0 if total_mem >= threshold else 0.0
        v = m_t - spike * threshold
        n = n_t
        primary_mem.append(m_t)
        history_mem.append(history_term)
        decision_mem.append(total_mem)
        post_mem.append(v)
        spikes.append(spike)
    return {
        "decision_mem": decision_mem,
        "primary_mem": primary_mem,
        "history_mem": history_mem,
        "post_reset_mem": post_mem,
        "spikes": spikes,
    }


def suffix_change_rate(a: list[float], b: list[float], start: int) -> float:
    suffix_len = max(len(a) - start, 1)
    return sum(1 for left, right in zip(a[start:], b[start:]) if left != right) / suffix_len


def mean_abs_delta(a: list[float], b: list[float], start: int) -> float:
    vals = [abs(left - right) for left, right in zip(a[start:], b[start:])]
    return sum(vals) / max(len(vals), 1)


def polyline(values: list[float], x0: float, y0: float, width: float, height: float, vmin: float, vmax: float) -> str:
    denom = max(vmax - vmin, 1e-9)
    points = []
    for idx, value in enumerate(values):
        x = x0 + width * idx / max(len(values) - 1, 1)
        y = y0 + height - height * (value - vmin) / denom
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def svg_text(x: float, y: float, text: str, size: int = 13, anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-family="Arial, sans-serif">{text}</text>'


def write_svg(x: list[float], masked_x: list[float], mask_steps: int, traces: dict[str, dict[str, dict[str, list[float]]]], out_file: Path, threshold: float) -> None:
    width, height = 1120, 900
    left, plot_w = 130, 900
    panel_h = 125
    panel_y = [70, 235, 400, 585]
    xmax = len(x) - 1
    mask_x = left + plot_w * (mask_steps - 0.5) / max(xmax, 1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append(svg_text(width / 2, 35, "Prefix masking exposes stronger history dependence in LSLIF", 20, "middle"))

    panels = [
        ("Input current", [(x, "original", "#111111", ""), (masked_x, "prefix masked", "#777777", "5,4")]),
        ("LIF decision membrane", [(traces["LIF"]["original"]["decision_mem"], "original", "#1f77b4", ""), (traces["LIF"]["masked"]["decision_mem"], "prefix masked", "#ff7f0e", "5,4")]),
        ("LSLIF decision membrane", [(traces["LSLIF"]["original"]["decision_mem"], "original", "#1f77b4", ""), (traces["LSLIF"]["masked"]["decision_mem"], "prefix masked", "#ff7f0e", "5,4")]),
    ]
    for i, (label, series) in enumerate(panels):
        y = panel_y[i]
        vals = [v for values, _, _, _ in series for v in values]
        if i > 0:
            vals.append(threshold)
        vmin, vmax = min(vals), max(vals)
        pad = max((vmax - vmin) * 0.12, 0.05)
        vmin, vmax = vmin - pad, vmax + pad
        parts.append(f'<rect x="{left}" y="{y}" width="{plot_w}" height="{panel_h}" fill="#fafafa" stroke="#cccccc"/>')
        parts.append(f'<rect x="{left}" y="{y}" width="{max(mask_x-left,0):.2f}" height="{panel_h}" fill="#d62728" opacity="0.08"/>')
        parts.append(f'<line x1="{mask_x:.2f}" y1="{y}" x2="{mask_x:.2f}" y2="{y+panel_h}" stroke="#888" stroke-dasharray="3,3"/>')
        parts.append(svg_text(18, y + 65, label, 14))
        for values, name, color, dash in series:
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2"{dash_attr} points="{polyline(values, left, y, plot_w, panel_h, vmin, vmax)}"/>')
        if i > 0:
            th_y = y + panel_h - panel_h * (threshold - vmin) / max(vmax - vmin, 1e-9)
            parts.append(f'<line x1="{left}" y1="{th_y:.2f}" x2="{left+plot_w}" y2="{th_y:.2f}" stroke="#d62728" stroke-dasharray="2,4"/>')
            parts.append(svg_text(left + plot_w + 8, th_y + 4, "threshold", 11))
        parts.append(svg_text(left, y - 8, f"min={vmin+pad:.2f}, max={vmax-pad:.2f}", 11))

    raster_y = panel_y[3]
    parts.append(f'<rect x="{left}" y="{raster_y}" width="{plot_w}" height="190" fill="#fafafa" stroke="#cccccc"/>')
    parts.append(f'<line x1="{mask_x:.2f}" y1="{raster_y}" x2="{mask_x:.2f}" y2="{raster_y+190}" stroke="#888" stroke-dasharray="3,3"/>')
    rows = [("LIF original", traces["LIF"]["original"]["spikes"], 25, "#1f77b4"), ("LIF masked", traces["LIF"]["masked"]["spikes"], 70, "#ff7f0e"), ("LSLIF original", traces["LSLIF"]["original"]["spikes"], 115, "#1f77b4"), ("LSLIF masked", traces["LSLIF"]["masked"]["spikes"], 160, "#ff7f0e")]
    for label, spikes, offset, color in rows:
        y = raster_y + offset
        parts.append(svg_text(18, y + 4, label, 13))
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="#dddddd"/>')
        for idx, spike in enumerate(spikes):
            if spike > 0:
                sx = left + plot_w * idx / max(xmax, 1)
                parts.append(f'<line x1="{sx:.2f}" y1="{y-16}" x2="{sx:.2f}" y2="{y+16}" stroke="{color}" stroke-width="2"/>')
    parts.append(svg_text(left + plot_w / 2, raster_y + 225, "time step", 14, "middle"))
    parts.append(svg_text(left, raster_y + 218, "0", 12, "middle"))
    parts.append(svg_text(left + plot_w, raster_y + 218, str(len(x) - 1), 12, "middle"))
    parts.append(svg_text(mask_x, raster_y + 218, f"mask={mask_steps}", 12, "middle"))
    parts.append('</svg>')
    out_file.write_text("\n".join(parts), encoding="utf-8")


def write_csv(x: list[float], masked_x: list[float], traces: dict[str, dict[str, dict[str, list[float]]]], out_file: Path) -> None:
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "input", "masked_input", "lif_mem", "lif_masked_mem", "lif_spike", "lif_masked_spike", "lslif_mem", "lslif_masked_mem", "lslif_spike", "lslif_masked_spike"])
        for i in range(len(x)):
            writer.writerow([i, x[i], masked_x[i], traces["LIF"]["original"]["decision_mem"][i], traces["LIF"]["masked"]["decision_mem"][i], traces["LIF"]["original"]["spikes"][i], traces["LIF"]["masked"]["spikes"][i], traces["LSLIF"]["original"]["decision_mem"][i], traces["LSLIF"]["masked"]["decision_mem"][i], traces["LSLIF"]["original"]["spikes"][i], traces["LSLIF"]["masked"]["spikes"][i]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize LIF vs LSLIF response after masking early input time steps.")
    parser.add_argument("--time-steps", type=int, default=40)
    parser.add_argument("--mask-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--history-weight", type=float, default=4.0)
    parser.add_argument("--history-power", type=float, default=1.0)
    parser.add_argument("--history-eps", type=float, default=1e-6)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_results/lslif_prefix_mask_single_neuron"))
    args = parser.parse_args()

    if not 0 <= args.mask_steps < args.time_steps:
        raise ValueError("--mask-steps must satisfy 0 <= mask_steps < time_steps")

    x = build_input(args.time_steps, args.seed)
    masked_x = list(x)
    for idx in range(args.mask_steps):
        masked_x[idx] = 0.0

    traces = {
        "LIF": {"original": simulate_lif(x, args.tau, args.threshold), "masked": simulate_lif(masked_x, args.tau, args.threshold)},
        "LSLIF": {"original": simulate_lslif(x, args.tau, args.threshold, args.history_weight, args.history_power, args.history_eps), "masked": simulate_lslif(masked_x, args.tau, args.threshold, args.history_weight, args.history_power, args.history_eps)},
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_file = args.out_dir / "lif_lslif_prefix_mask_comparison.svg"
    csv_file = args.out_dir / "lif_lslif_prefix_mask_traces.csv"
    write_svg(x, masked_x, args.mask_steps, traces, fig_file, args.threshold)
    write_csv(x, masked_x, traces, csv_file)

    for model in ["LIF", "LSLIF"]:
        rate = suffix_change_rate(traces[model]["original"]["spikes"], traces[model]["masked"]["spikes"], args.mask_steps)
        mem_delta = mean_abs_delta(traces[model]["original"]["decision_mem"], traces[model]["masked"]["decision_mem"], args.mask_steps)
        print(f"{model}: suffix spike change rate={rate:.3f}, suffix mean |decision_mem delta|={mem_delta:.3f}")
    print(f"saved figure: {fig_file}")
    print(f"saved traces: {csv_file}")


if __name__ == "__main__":
    main()
