#!/usr/bin/env python3
"""Single-neuron prefix-mask comparison for LIF and LSLIF.

The experiment feeds one non-stationary input sequence to a vanilla LIF neuron and
an LSLIF neuron, masks the input prefix, and visualizes how much the later
membrane trajectory and spike train change.  The script writes an SVG figure with
input curves, membrane curves, and spike rasters, using only Python's standard
library so it can run in minimal environments.

Optional paper exports require matplotlib (not installed automatically):
    python analysis/lslif_prefix_mask_single_neuron.py --paper --time-steps 40 --mask-steps 20
The original defaults remain unchanged. Paper plots show discrete samples only.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
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
    parts.append(svg_text(width / 2, 35, "Responses to prefix-masked input", 20, "middle"))

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


def write_paper_outputs(x, masked_x, mask_steps, traces, args, out_dir):
    """Export raw state differences and a discrete-time, unsmoothed figure.

    The no-reset integrator is a mechanistic reference, not a trained baseline.
    Vertical segments mark the actual pre/post reset difference at the same step.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    decay = 1.0 - 1.0 / args.tau
    reference = []
    for inputs in (x, masked_x):
        state = 0.0
        values = []
        for current in inputs:
            state = decay * state + current
            values.append(state)
        reference.append(values)
    full, masked = traces["LIF"]["original"], traces["LIF"]["masked"]
    rows = []
    for t in range(len(x)):
        pre = full["decision_mem"][t] - masked["decision_mem"][t]
        post = full["post_reset_mem"][t] - masked["post_reset_mem"][t]
        row = dict(step=t + 1, input=x[t], masked_input=masked_x[t],
                   lif_pre_full=full["decision_mem"][t], lif_pre_masked=masked["decision_mem"][t],
                   lif_post_full=full["post_reset_mem"][t], lif_post_masked=masked["post_reset_mem"][t],
                   spike_full=full["spikes"][t], spike_masked=masked["spikes"][t],
                   no_reset_full=reference[0][t], no_reset_masked=reference[1][t],
                   delta_pre=pre, delta_post=post, delta_no_reset=reference[0][t]-reference[1][t],
                   reset_abs_change=abs(post)-abs(pre))
        # Exact soft-reset identity: post difference = pre difference - theta * spike difference.
        assert math.isclose(post, pre - args.threshold * (row["spike_full"]-row["spike_masked"]), abs_tol=1e-12)
        # LSLIF's decision voltage includes history; its reset acts on primary_mem.
        for model, conditions in traces.items():
            for state_name in conditions["original"]:
                original = conditions["original"][state_name][t]
                masked_value = conditions["masked"][state_name][t]
                prefix = model.lower() + "_" + state_name
                row[prefix + "_full"] = original
                row[prefix + "_masked"] = masked_value
                row[prefix + "_delta"] = original - masked_value
        rows.append(row)
    with (out_dir / "reset_state_traces.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    suffix = rows[mask_steps:]
    events = [r for r in suffix if r["spike_full"] or r["spike_masked"]]
    summary = dict(suffix_spike_change_rate=suffix_change_rate(full["spikes"], masked["spikes"], mask_steps),
                   suffix_reset_events=len(events),
                   reset_contractions=sum(r["reset_abs_change"] < -1e-12 for r in events),
                   reset_expansions=sum(r["reset_abs_change"] > 1e-12 for r in events),
                   reset_unchanged=sum(abs(r["reset_abs_change"]) <= 1e-12 for r in events),
                   reset_event_definition="A suffix time step where either paired LIF trajectory spikes; simultaneous spikes count once.",
                   suffix_first_step=mask_steps + 1,
                   absolute_tolerance=1e-12,
                   delta_definition="full minus masked",
                   note="Counts describe this sequence only; spike agreement is not proof of lost information or impaired accuracy.")
    # Compare the same primary state across the LSLIF reset, excluding history.
    lslif_events = [r for r in suffix if r["lslif_spikes_full"] or r["lslif_spikes_masked"]]
    lslif_changes = [abs(r["lslif_post_reset_mem_delta"]) - abs(r["lslif_primary_mem_delta"])
                    for r in lslif_events]
    summary["lslif_primary_reset"] = dict(
        suffix_reset_events=len(lslif_events),
        reset_contractions=sum(v < -1e-12 for v in lslif_changes),
        reset_expansions=sum(v > 1e-12 for v in lslif_changes),
        reset_unchanged=sum(abs(v) <= 1e-12 for v in lslif_changes),
        reset_event_definition="A suffix step where either LSLIF trajectory spikes; simultaneous spikes count once.")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    style = {"font.family": "DejaVu Serif", "font.size": 9, "axes.spines.top": False,
             "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.1), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.5, 0.65]}, layout="constrained")
        steps = list(range(1, len(x)+1))
        blue, orange, green = "#0072B2", "#D55E00", "#009E73"
        axes[0].plot(steps, x, color=blue, linestyle="none", marker="o", ms=4,
                     mfc="none", label="Full input")
        axes[0].plot(steps, masked_x, color=orange, linestyle="none", marker=".", ms=3,
                     label="Prefix masked")
        axes[0].set_ylabel("Input current")
        axes[0].set_title("(a) Identical input after the masked prefix", loc="left", fontsize=10)
        axes[0].legend(frameon=False, ncol=2, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.2))
        axes[1].plot(steps, [r["delta_no_reset"] for r in rows], color=green,
                     linestyle="none", marker="x", ms=3, label="No-reset integrator")
        axes[1].plot(steps, [r["delta_pre"] for r in rows], linestyle="none", marker="o", ms=3,
                     mfc="white", mec=blue, label="LIF before reset")
        axes[1].plot(steps, [r["delta_post"] for r in rows], linestyle="none", marker=".", ms=5,
                     color=orange, label="LIF after reset")
        for i, row in enumerate(rows):
            if row["spike_full"] or row["spike_masked"]:
                axes[1].plot([i+1, i+1], [row["delta_pre"], row["delta_post"]], color=orange, lw=1.2)
        axes[1].axhline(0, color="0.65", lw=0.6)
        axes[1].set_ylabel("State difference\n(full − masked)")
        axes[1].set_title("(b) State differences before and after soft reset", loc="left", fontsize=10)
        axes[1].legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.15))
        for y, key, color in [(1, "spike_full", blue), (0, "spike_masked", orange)]:
            ts = [r["step"] for r in rows if r[key]]
            axes[2].vlines(ts, y-0.23, y+0.23, color=color, lw=1.2)
        axes[2].set_yticks([0, 1], ["Masked", "Full"])
        axes[2].set_ylim(-0.5, 1.5)
        axes[2].set_title("(c) LIF spike output", loc="left", fontsize=10)
        axes[2].set_xlabel("Time step")
        for ax in axes:
            if mask_steps:
                ax.axvspan(0.5, mask_steps+0.5, color="0.9", zorder=-5)
                ax.axvline(mask_steps+0.5, color="0.5", lw=0.8, ls="--")
            ax.set_xlim(0.5, len(x)+0.5)
        fig.suptitle("State differences under prefix masking", fontsize=12)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(out_dir / ("prefix_reset_mechanism."+ext), dpi=300)
        plt.close(fig)
    print(json.dumps(summary, indent=2))


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
    parser.add_argument("--paper", action="store_true", help="Add mechanism figure (requires matplotlib), raw states and configuration.")
    parser.add_argument("--input-scale", type=float, default=1.0, help="Scale the generated current without altering its timing.")
    args = parser.parse_args()
    if args.time_steps < 2 or not math.isfinite(args.tau) or args.tau <= 1:
        parser.error("time-steps must be >= 2 and tau must be finite and > 1")
    if not math.isfinite(args.threshold) or args.threshold <= 0:
        parser.error("threshold must be finite and > 0")
    if not all(math.isfinite(v) for v in (args.input_scale, args.history_weight, args.history_power, args.history_eps)) or args.history_eps <= 0 or args.input_scale < 0:
        parser.error("parameters must be finite; history-eps > 0 and input-scale >= 0")
    if args.paper:
        # Each paper run is isolated even when an existing output root is supplied.
        args.out_dir = args.out_dir / ("paper_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))

    if not 0 <= args.mask_steps < args.time_steps:
        raise ValueError("--mask-steps must satisfy 0 <= mask_steps < time_steps")

    x = [args.input_scale * v for v in build_input(args.time_steps, args.seed)]
    masked_x = list(x)
    for idx in range(args.mask_steps):
        masked_x[idx] = 0.0

    traces = {
        "LIF": {"original": simulate_lif(x, args.tau, args.threshold), "masked": simulate_lif(masked_x, args.tau, args.threshold)},
        "LSLIF": {"original": simulate_lslif(x, args.tau, args.threshold, args.history_weight, args.history_power, args.history_eps), "masked": simulate_lslif(masked_x, args.tau, args.threshold, args.history_weight, args.history_power, args.history_eps)},
    }

    args.out_dir.mkdir(parents=True, exist_ok=not args.paper)
    fig_file = args.out_dir / "lif_lslif_prefix_mask_comparison.svg"
    csv_file = args.out_dir / "lif_lslif_prefix_mask_traces.csv"
    write_svg(x, masked_x, args.mask_steps, traces, fig_file, args.threshold)
    write_csv(x, masked_x, traces, csv_file)

    if args.paper:
        write_paper_outputs(x, masked_x, args.mask_steps, traces, args, args.out_dir)

    for model in ["LIF", "LSLIF"]:
        rate = suffix_change_rate(traces[model]["original"]["spikes"], traces[model]["masked"]["spikes"], args.mask_steps)
        mem_delta = mean_abs_delta(traces[model]["original"]["decision_mem"], traces[model]["masked"]["decision_mem"], args.mask_steps)
        print(f"{model}: suffix spike change rate={rate:.3f}, suffix mean |decision_mem delta|={mem_delta:.3f}")
    print(f"saved figure: {fig_file}")
    print(f"saved traces: {csv_file}")


if __name__ == "__main__":
    main()
