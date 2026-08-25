#!/usr/bin/env python3
"""Compare firing rates around the best observed epochs of two training runs.

A run directory may contain a resumed/partial ``metrics.csv`` and ``args.txt``;
only epochs that are actually present in that directory are considered. Plotting
uses Matplotlib for publication-quality PNG and SVG output.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EPOCH_RE = re.compile(
    r"epoch=(?P<epoch>\d+),\s*train_loss=.*?"
    r"test_acc=(?P<test_acc>[-+0-9.eE]+),\s*"
    r"max_test_acc=(?P<max_test_acc>[-+0-9.eE]+),.*?"
    r"test_spike_rate_global=(?P<global_rate>[-+0-9.eE]+),\s*"
    r"escape_time=.*?(?=(?:\r?\n|$))",
)
LAYERS_RE = re.compile(
    r"^test_spike_rate_layers=(?:OrderedDict\()?"
    r"(?P<payload>\{.*\}|\[.*\])"
    r"\)?\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    test_acc: float
    max_test_acc: float | None
    global_rate: float
    layer_rates: dict[str, float]


@dataclass
class RunData:
    path: Path
    summary: dict
    records: dict[int, EpochRecord]
    warnings: list[str]

    @property
    def model_name(self) -> str:
        return str(self.summary.get("neuron_model", "unknown"))


def _same_record(left: EpochRecord, right: EpochRecord) -> bool:
    return (
        math.isclose(left.test_acc, right.test_acc, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(left.global_rate, right.global_rate, rel_tol=0.0, abs_tol=1e-12)
        and left.layer_rates == right.layer_rates
    )


def _insert_record(records: dict[int, EpochRecord], record: EpochRecord, source: str, warnings: list[str]) -> None:
    previous = records.get(record.epoch)
    if previous is None:
        records[record.epoch] = record
    elif _same_record(previous, record):
        warnings.append(f"Removed an exactly duplicated epoch {record.epoch} from {source}.")
    else:
        raise ValueError(f"Conflicting records for epoch {record.epoch} in {source}.")


def _read_metrics_csv(path: Path, warnings: list[str]) -> dict[int, EpochRecord]:
    records: dict[int, EpochRecord] = {}
    if not path.is_file():
        return records
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"epoch", "test_acc", "test_spike_rate_global"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} is missing required columns: {sorted(required)}")
        for row in reader:
            if not row.get("epoch", "").strip():
                continue
            max_acc = row.get("max_test_acc", "").strip()
            record = EpochRecord(
                epoch=int(float(row["epoch"])),
                test_acc=float(row["test_acc"]),
                max_test_acc=float(max_acc) if max_acc else None,
                global_rate=float(row["test_spike_rate_global"]),
                layer_rates={},
            )
            _insert_record(records, record, str(path), warnings)
    return records


def _parse_layers(segment: str, path: Path, epoch: int) -> dict[str, float]:
    match = LAYERS_RE.search(segment)
    if not match:
        return {}
    try:
        value = ast.literal_eval(match.group("payload"))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse layer firing rates for epoch {epoch} in {path}: {exc}") from exc
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, list):
        try:
            items = dict(value).items()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"OrderedDict entries for epoch {epoch} in {path} are not valid name/rate pairs."
            ) from exc
    else:
        raise ValueError(
            f"Layer firing rates for epoch {epoch} in {path} are neither a dictionary nor an OrderedDict entry list."
        )
    try:
        return {str(name): float(rate) for name, rate in items}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Layer firing rates for epoch {epoch} in {path} contain a non-numeric rate.") from exc


def _read_args_txt(path: Path, warnings: list[str]) -> dict[int, EpochRecord]:
    records: dict[int, EpochRecord] = {}
    if not path.is_file():
        return records
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(EPOCH_RE.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[match.start():end]
        epoch = int(match.group("epoch"))
        record = EpochRecord(
            epoch=epoch,
            test_acc=float(match.group("test_acc")),
            max_test_acc=float(match.group("max_test_acc")),
            global_rate=float(match.group("global_rate")),
            layer_rates=_parse_layers(segment, path, epoch),
        )
        _insert_record(records, record, str(path), warnings)
    return records


def load_run(path: Path) -> RunData:
    if not path.is_dir():
        raise ValueError(f"Run path is not a directory: {path}")
    warnings: list[str] = []
    summary_path = path / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    csv_records = _read_metrics_csv(path / "metrics.csv", warnings)
    text_records = _read_args_txt(path / "args.txt", warnings)
    epochs = sorted(set(csv_records) | set(text_records))
    records: dict[int, EpochRecord] = {}
    for epoch in epochs:
        csv_record = csv_records.get(epoch)
        text_record = text_records.get(epoch)
        if csv_record and text_record:
            if not math.isclose(csv_record.test_acc, text_record.test_acc, abs_tol=1e-9):
                raise ValueError(f"test_acc differs between metrics.csv and args.txt at epoch {epoch} in {path}.")
            if not math.isclose(csv_record.global_rate, text_record.global_rate, abs_tol=1e-9):
                raise ValueError(f"global firing rate differs between metrics.csv and args.txt at epoch {epoch} in {path}.")
            records[epoch] = EpochRecord(
                epoch=epoch,
                test_acc=csv_record.test_acc,
                max_test_acc=csv_record.max_test_acc,
                global_rate=csv_record.global_rate,
                layer_rates=text_record.layer_rates,
            )
        else:
            records[epoch] = csv_record or text_record  # type: ignore[assignment]
    if not records:
        raise ValueError(f"No epoch firing-rate records found in {path}.")
    if not any(record.layer_rates for record in records.values()):
        raise ValueError(f"No per-layer firing rates found in {path / 'args.txt'}.")
    observed = sorted(records)
    if observed[0] > 1:
        warnings.append(
            f"Partial/resumed log detected: observed epochs {observed[0]}-{observed[-1]}; "
            "only epochs in this directory are analyzed."
        )
    best = max(records.values(), key=lambda item: (item.test_acc, -item.epoch))
    historical = max((item.max_test_acc or item.test_acc) for item in records.values())
    if historical > best.test_acc + 1e-12:
        warnings.append(
            f"Historical max_test_acc={historical:.6f} exceeds the best observed test_acc={best.test_acc:.6f}; "
            "the best observed epoch is used."
        )
    return RunData(path=path, summary=summary, records=records, warnings=warnings)


def select_window(run: RunData, size: int) -> tuple[EpochRecord, list[EpochRecord]]:
    if size < 1 or size % 2 == 0:
        raise ValueError("--window-size must be a positive odd integer.")
    ordered = [run.records[epoch] for epoch in sorted(run.records)]
    best_index = max(range(len(ordered)), key=lambda i: (ordered[i].test_acc, -ordered[i].epoch))
    best = ordered[best_index]
    actual_size = min(size, len(ordered))
    start = best_index - actual_size // 2
    start = max(0, min(start, len(ordered) - actual_size))
    window = ordered[start:start + actual_size]
    if actual_size < size:
        run.warnings.append(f"Only {actual_size} observed epochs are available for the requested {size}-epoch window.")
    elif best_index < size // 2 or best_index + size // 2 >= len(ordered):
        run.warnings.append(f"Best observed epoch {best.epoch} is at a boundary; used a one-sided window.")
    min_acc = min(item.test_acc for item in window)
    if best.test_acc - min_acc >= 0.20:
        run.warnings.append(
            f"Accuracy changes by at least 0.20 inside the selected window around epoch {best.epoch}; "
            "the mean may mix normal and collapsed states."
        )
    return best, window


def _available_layers(run: RunData, window: Iterable[EpochRecord]) -> set[str]:
    layer_sets = [set(record.layer_rates) for record in window]
    return set.intersection(*layer_sets) if layer_sets else set()


def select_layers(
    ls_run: RunData,
    ls_window: list[EpochRecord],
    baseline_run: RunData,
    baseline_window: list[EpochRecord],
    requested: tuple[str | None, str | None, str | None],
) -> tuple[str, str, str]:
    common = _available_layers(ls_run, ls_window) & _available_layers(baseline_run, baseline_window)
    if not common:
        raise ValueError("The selected windows have no common per-layer firing-rate names.")
    if any(requested):
        if not all(requested):
            raise ValueError("Specify all of --shallow-layer, --middle-layer, and --deep-layer together.")
        missing = [name for name in requested if name not in common]
        if missing:
            raise ValueError(f"Requested layers are unavailable in both selected windows: {missing}")
        return requested  # type: ignore[return-value]
    resnet = ("layer1.0.relu1", "layer3.0.relu1", "layer4.1.relu2")
    if all(name in common for name in resnet):
        return resnet
    first_record = next(record for record in ls_window if record.layer_rates)
    ordered_common = [name for name in first_record.layer_rates if name in common]
    if len(ordered_common) < 3:
        raise ValueError("At least three common neuron layers are required for automatic selection.")
    indices = [round((len(ordered_common) - 1) * fraction) for fraction in (0.25, 0.50, 0.75)]
    return tuple(ordered_common[index] for index in indices)  # type: ignore[return-value]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def compute_rates(window: list[EpochRecord], layers: tuple[str, str, str]) -> dict[str, float]:
    return {
        "Global": _mean(record.global_rate for record in window),
        "Shallow": _mean(record.layer_rates[layers[0]] for record in window),
        "Middle": _mean(record.layer_rates[layers[1]] for record in window),
        "Deep": _mean(record.layer_rates[layers[2]] for record in window),
    }


def _write_csvs(
    output: Path,
    groups: list[tuple[str, RunData, EpochRecord, list[EpochRecord], dict[str, float]]],
    layers: tuple[str, str, str],
) -> None:
    scope_layers = {"Global": "global", "Shallow": layers[0], "Middle": layers[1], "Deep": layers[2]}
    with (output / "spike_rate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "group", "neuron_model", "observed_epoch_start", "observed_epoch_end",
            "best_observed_epoch", "best_observed_acc", "selected_epochs", "scope", "layer", "mean_firing_rate",
        ])
        for label, run, best, window, rates in groups:
            observed = sorted(run.records)
            selected = ";".join(str(item.epoch) for item in window)
            for scope in ("Global", "Shallow", "Middle", "Deep"):
                writer.writerow([
                    label, run.model_name, observed[0], observed[-1], best.epoch, best.test_acc,
                    selected, scope.lower(), scope_layers[scope], rates[scope],
                ])
    ls_rates = groups[0][4]
    baseline_rates = groups[1][4]
    with (output / "spike_rate_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scope", "layer", "ls_mean_firing_rate", "baseline_mean_firing_rate", "difference", "relative_change"])
        for scope in ("Global", "Shallow", "Middle", "Deep"):
            difference = ls_rates[scope] - baseline_rates[scope]
            relative = difference / baseline_rates[scope] if baseline_rates[scope] else float("nan")
            writer.writerow([scope.lower(), scope_layers[scope], ls_rates[scope], baseline_rates[scope], difference, relative])


def _write_plots(
    output_dir: Path,
    ls_rates: dict[str, float],
    baseline_rates: dict[str, float],
    groups: list[tuple[str, RunData, EpochRecord, list[EpochRecord], dict[str, float]]],
    layers: tuple[str, str, str],
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError("Plotting requires matplotlib. Install it with: python -m pip install -r analysis/requirements.txt")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    if fig_width <= 0 or fig_height <= 0:
        raise ValueError("--fig-width and --fig-height must be positive.")
    if dpi < 72:
        raise ValueError("--dpi must be at least 72.")

    scopes = ("Global", "Shallow", "Middle", "Deep")
    ls_values = [ls_rates[scope] for scope in scopes]
    baseline_values = [baseline_rates[scope] for scope in scopes]
    x = list(range(len(scopes)))
    bar_width = 0.34
    ls_model, baseline_model = groups[0][1].model_name, groups[1][1].model_name
    with plt.rc_context({
        "font.family": "DejaVu Serif",
        "font.size": 22,
        "axes.labelsize": 24,
        "axes.titlesize": 30,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 22,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    }):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
        ls_bars = ax.bar(
            [value - bar_width / 2 for value in x], ls_values, bar_width,
            label=f"LS: {ls_model}", color="#4C78A8", edgecolor="#334155", linewidth=0.7,
        )
        baseline_bars = ax.bar(
            [value + bar_width / 2 for value in x], baseline_values, bar_width,
            label=f"Baseline: {baseline_model}", color="#F2A65A", edgecolor="#7C2D12", linewidth=0.7,
        )
        maximum = max(ls_values + baseline_values + [1e-6])
        ax.set_ylim(0, maximum * 1.30)
        ax.set_ylabel("Mean firing rate")
        ax.set_xticks(x)
        ax.set_xticklabels(scopes)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#475569")
        ax.spines["bottom"].set_color("#475569")
        ax.set_title("Mean firing-rate comparison", fontweight="bold", pad=18)
        ax.legend(
            loc="upper right", ncol=1, frameon=True, fancybox=False,
            edgecolor="#94A3B8", facecolor="white", framealpha=0.95,
            handlelength=1.8, borderpad=0.8, labelspacing=0.7,
        )
        ax.bar_label(ls_bars, labels=[f"{value:.2%}" for value in ls_values], padding=6, fontsize=19)
        ax.bar_label(
            baseline_bars, labels=[f"{value:.2%}" for value in baseline_values],
            padding=6, fontsize=19,
        )
        fig.savefig(output_dir / "mean_spike_rate_comparison.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(output_dir / "mean_spike_rate_comparison.svg", bbox_inches="tight")
        plt.close(fig)


def _config_warnings(ls_run: RunData, baseline_run: RunData) -> list[str]:
    warnings = []
    keys = ("dataset", "model", "seed", "time_steps", "batch_size", "optimizer", "base_lr", "lr_scheduler", "weight_decay")
    for key in keys:
        left, right = ls_run.summary.get(key), baseline_run.summary.get(key)
        if left is not None and right is not None and left != right:
            warnings.append(f"Configuration mismatch for {key}: LS={left!r}, baseline={right!r}.")
    return warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ls-run", type=Path, required=True, help="LS experiment directory")
    parser.add_argument("--baseline-run", type=Path, required=True, help="non-LS experiment directory")
    parser.add_argument("--output-dir", type=Path, default=Path("spike_rate_comparison"))
    parser.add_argument("--window-size", type=int, default=5, help="odd number of epochs around the best observed accuracy")
    parser.add_argument("--fig-width", type=float, default=14.0, help="figure width in inches")
    parser.add_argument("--fig-height", type=float, default=8.0, help="figure height in inches")
    parser.add_argument("--dpi", type=int, default=300, help="PNG dots per inch; SVG remains vector-based")
    parser.add_argument("--shallow-layer")
    parser.add_argument("--middle-layer")
    parser.add_argument("--deep-layer")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ls_run = load_run(args.ls_run)
        baseline_run = load_run(args.baseline_run)
        ls_best, ls_window = select_window(ls_run, args.window_size)
        baseline_best, baseline_window = select_window(baseline_run, args.window_size)
        layers = select_layers(
            ls_run, ls_window, baseline_run, baseline_window,
            (args.shallow_layer, args.middle_layer, args.deep_layer),
        )
        ls_rates = compute_rates(ls_window, layers)
        baseline_rates = compute_rates(baseline_window, layers)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        groups = [
            ("LS", ls_run, ls_best, ls_window, ls_rates),
            ("Baseline", baseline_run, baseline_best, baseline_window, baseline_rates),
        ]
        _write_csvs(args.output_dir, groups, layers)
        _write_plots(
            args.output_dir, ls_rates, baseline_rates, groups, layers,
            args.fig_width, args.fig_height, args.dpi,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Representative layers: shallow={layers[0]}, middle={layers[1]}, deep={layers[2]}")
    for label, run, best, window, rates in groups:
        observed = sorted(run.records)
        print(f"\n{label} ({run.model_name})")
        print(f"  run: {run.path}")
        print(f"  observed epochs: {observed[0]}-{observed[-1]}")
        print(f"  best observed epoch/accuracy: {best.epoch} / {best.test_acc:.4%}")
        print(f"  selected epochs: {', '.join(str(item.epoch) for item in window)}")
        for scope in ("Global", "Shallow", "Middle", "Deep"):
            print(f"  {scope.lower()} mean firing rate: {rates[scope]:.4%}")
        for warning in run.warnings:
            print(f"  warning: {warning}")
    for warning in _config_warnings(ls_run, baseline_run):
        print(f"warning: {warning}")
    print(f"\nSaved analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
