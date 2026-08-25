#!/usr/bin/env python3
"""Compare firing rates around the best observed epochs of two training runs.

The script intentionally uses only the Python standard library.  A run directory
may contain a resumed/partial ``metrics.csv`` and ``args.txt``; only epochs that
are actually present in that directory are considered.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import struct
import sys
import zlib
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


FONT = {
    "0": ("111", "101", "101", "101", "111"), "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"), "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"), "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"), "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"), "9": ("111", "101", "111", "001", "111"),
    ".": ("000", "000", "000", "000", "010"), "%": ("101", "001", "010", "100", "101"),
    "-": ("000", "000", "111", "000", "000"), " ": ("000",) * 5,
    "G": ("111", "100", "101", "101", "111"), "S": ("111", "100", "111", "001", "111"),
    "M": ("10001", "11011", "10101", "10001", "10001"), "D": ("110", "101", "101", "101", "110"),
    "L": ("100", "100", "100", "100", "111"), "B": ("110", "101", "110", "101", "110"),
}


def _draw_text(pixels: list[list[tuple[int, int, int]]], x: int, y: int, value: str, scale: int = 2) -> None:
    for char in value.upper():
        glyph = FONT.get(char, FONT[" "])
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "1":
                    for dy in range(scale):
                        for dx in range(scale):
                            yy, xx = y + row * scale + dy, x + column * scale + dx
                            if 0 <= yy < len(pixels) and 0 <= xx < len(pixels[0]):
                                pixels[yy][xx] = (30, 30, 30)
        x += (len(glyph[0]) + 1) * scale


def _write_png(path: Path, ls_rates: dict[str, float], baseline_rates: dict[str, float]) -> None:
    width, height = 920, 560
    pixels = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]
    scopes = ("Global", "Shallow", "Middle", "Deep")
    maximum = max(max(ls_rates.values()), max(baseline_rates.values()), 1e-6) * 1.18
    left, top, bottom = 80, 55, 455
    for y in range(top, bottom + 1):
        pixels[y][left] = (70, 70, 70)
    for x in range(left, width - 30):
        pixels[bottom][x] = (70, 70, 70)
    group_width = (width - left - 50) // 4
    colors = ((60, 126, 210), (235, 133, 54))
    for index, scope in enumerate(scopes):
        center = left + group_width * index + group_width // 2
        for offset, rate, color in ((-34, ls_rates[scope], colors[0]), (8, baseline_rates[scope], colors[1])):
            bar_height = round((bottom - top) * rate / maximum)
            x0, x1 = center + offset, center + offset + 28
            y0 = bottom - bar_height
            for y in range(y0, bottom):
                for x in range(x0, x1):
                    pixels[y][x] = color
            _draw_text(pixels, x0 - 8, max(15, y0 - 18), f"{rate * 100:.1f}%", 2)
        _draw_text(pixels, center - 10, bottom + 16, scope[0], 3)
    _draw_text(pixels, 110, 505, "LS", 3)
    for y in range(500, 520):
        for x in range(80, 102): pixels[y][x] = colors[0]
    _draw_text(pixels, 225, 505, "B", 3)
    for y in range(500, 520):
        for x in range(195, 217): pixels[y][x] = colors[1]
    raw = b"".join(b"\x00" + bytes(channel for pixel in row for channel in pixel) for row in pixels)
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


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
        _write_png(args.output_dir / "mean_spike_rate_comparison.png", ls_rates, baseline_rates)
    except (OSError, ValueError, json.JSONDecodeError, csv.Error) as exc:
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
