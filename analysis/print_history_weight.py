#!/usr/bin/env python3
"""Print trained history_weight values from old or new LSLIF checkpoints.

Usage:
    python analysis/print_history_weight.py /path/to/checkpoint.pth

The script supports both known learnable-history encodings in this repository:

* new/bounded: beta = lo + (hi - lo) * sigmoid(history_weight_raw)
* old/positive: beta = softplus(history_weight_raw)

When checkpoint metadata or CLI overrides do not make the encoding unambiguous,
``--mode auto`` prints both decoded values instead of silently guessing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable


DEFAULT_HISTORY_WEIGHT = 1.0
DEFAULT_HISTORY_WEIGHT_LO = -0.8
DEFAULT_HISTORY_WEIGHT_HI = 0.8
KNOWN_CHECKPOINT_STATE_KEYS = ("net", "model", "state_dict", "module", "network")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print final trained history_weight values from a .pth checkpoint."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint .pth file.")
    parser.add_argument(
        "--mode",
        choices=("auto", "new", "old", "both"),
        default="auto",
        help=(
            "Decoding mode. new uses bounded sigmoid, old uses softplus, both prints both. "
            "auto uses checkpoint args when available; otherwise prints both for learned raw tensors."
        ),
    )
    parser.add_argument(
        "--history-weight-lo",
        type=float,
        default=None,
        help="Override lower bound for new/bounded checkpoints. Defaults to checkpoint args or -0.8.",
    )
    parser.add_argument(
        "--history-weight-hi",
        type=float,
        default=None,
        help="Override upper bound for new/bounded checkpoints. Defaults to checkpoint args or 0.8.",
    )
    parser.add_argument(
        "--fixed-history-weight",
        type=float,
        default=None,
        help=(
            "Value to print when no history_weight_raw tensor is present. "
            "If omitted, checkpoint args.history_weight is used when available, otherwise 1.0."
        ),
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Print all decoded values. By default only a preview is shown for large tensors.",
    )
    parser.add_argument(
        "--max-preview",
        type=int,
        default=32,
        help="Maximum values to preview when --show-values is not set. Default: %(default)s.",
    )
    return parser.parse_args()


def torch_load_cpu(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_args(checkpoint: Any) -> Any:
    if not isinstance(checkpoint, dict):
        return None
    for key in ("args", "train_args", "config", "cfg"):
        if key in checkpoint:
            return checkpoint[key]
    return None


def looks_like_state_dict(value: Any) -> bool:
    import torch

    return isinstance(value, dict) and any(isinstance(v, torch.Tensor) for v in value.values())


def extract_state_dict(checkpoint: Any) -> tuple[dict[str, Any], dict[str, Any], Any]:
    metadata: dict[str, Any] = {}
    args_obj = extract_args(checkpoint)

    if looks_like_state_dict(checkpoint):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict):
        state_dict = None
        for key in KNOWN_CHECKPOINT_STATE_KEYS:
            candidate = checkpoint.get(key)
            if looks_like_state_dict(candidate):
                state_dict = candidate
                break
        if state_dict is None:
            import torch

            tensor_items = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
            if tensor_items:
                state_dict = tensor_items
            else:
                raise TypeError("Could not find a tensor state_dict in checkpoint.")
        for key in ("epoch", "max_test_acc", "best_acc", "acc", "test_acc"):
            if key in checkpoint:
                metadata[key] = checkpoint[key]
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    return state_dict, metadata, args_obj


def find_history_weight_raw_items(state_dict: dict[str, Any]) -> list[tuple[str, Any]]:
    return [
        (name, value.detach().float().cpu())
        for name, value in state_dict.items()
        if _is_tensor(value) and "history_weight_raw" in name
    ]


def _is_tensor(value: Any) -> bool:
    import torch

    return isinstance(value, torch.Tensor)


def decode_new(raw: Any, lo: float, hi: float) -> Any:
    import torch

    return lo + (hi - lo) * torch.sigmoid(raw)


def decode_old(raw: Any) -> Any:
    import torch.nn.functional as F

    return F.softplus(raw)


def format_values(values: Any, show_values: bool, max_preview: int) -> str:
    flat = values.flatten()
    shown = flat if show_values else flat[: max(0, max_preview)]
    rendered = ", ".join(f"{float(v):.8g}" for v in shown)
    if not show_values and flat.numel() > shown.numel():
        rendered += f", ... ({flat.numel()} total)"
    return f"[{rendered}]"


def print_summary(label: str, values: Any, show_values: bool, max_preview: int) -> None:
    flat = values.flatten()
    print(f"{label}:")
    print(f"  shape: {tuple(values.shape)}")
    print(f"  min: {float(flat.min()):.8g}")
    print(f"  max: {float(flat.max()):.8g}")
    print(f"  mean: {float(flat.mean()):.8g}")
    print(f"  values: {format_values(flat, show_values, max_preview)}")


def any_key_contains(keys: Iterable[str], text: str) -> bool:
    return any(text in key for key in keys)


def choose_auto_mode(state_dict: dict[str, Any], args_obj: Any, lo_arg: float | None, hi_arg: float | None) -> str:
    if lo_arg is not None or hi_arg is not None:
        return "new"
    if get_attr_or_key(args_obj, "history_weight_lo") is not None or get_attr_or_key(args_obj, "history_weight_hi") is not None:
        return "new"
    if any_key_contains(state_dict.keys(), "history_weight_lo") or any_key_contains(state_dict.keys(), "history_weight_hi"):
        return "new"
    return "both"


def main() -> None:
    args = parse_args()
    checkpoint = torch_load_cpu(args.checkpoint)
    state_dict, metadata, args_obj = extract_state_dict(checkpoint)
    raw_items = find_history_weight_raw_items(state_dict)

    lo = args.history_weight_lo
    if lo is None:
        lo = get_attr_or_key(args_obj, "history_weight_lo", DEFAULT_HISTORY_WEIGHT_LO)
    hi = args.history_weight_hi
    if hi is None:
        hi = get_attr_or_key(args_obj, "history_weight_hi", DEFAULT_HISTORY_WEIGHT_HI)
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        raise ValueError("history_weight_hi must be larger than history_weight_lo")

    mode = args.mode
    if mode == "auto":
        mode = choose_auto_mode(state_dict, args_obj, args.history_weight_lo, args.history_weight_hi)

    print(f"checkpoint: {args.checkpoint}")
    for key, value in metadata.items():
        print(f"{key}: {value}")

    if not raw_items:
        fixed = args.fixed_history_weight
        source = "--fixed-history-weight"
        if fixed is None:
            fixed = get_attr_or_key(args_obj, "history_weight", DEFAULT_HISTORY_WEIGHT)
            source = "checkpoint args.history_weight" if args_obj is not None else "default"
        print("history_weight_raw: not found")
        print("learnable_history_weight: false or not saved")
        print(f"history_weight: {float(fixed):.8g} ({source})")
        return

    print(f"history_weight_raw tensors: {len(raw_items)}")
    print(f"decode mode: {mode}")
    if mode in {"new", "both"}:
        print(f"new/bounded formula: beta = {lo:.8g} + ({hi:.8g} - {lo:.8g}) * sigmoid(raw)")
    if mode in {"old", "both"}:
        print("old/positive formula: beta = softplus(raw)")

    decoded_groups: dict[str, list[Any]] = {}
    for name, raw in raw_items:
        print("=" * 88)
        print(name)
        print(f"raw shape: {tuple(raw.shape)}")
        print(f"raw min/max/mean: {float(raw.min()):.8g} / {float(raw.max()):.8g} / {float(raw.mean()):.8g}")
        if mode in {"new", "both"}:
            values = decode_new(raw, lo, hi)
            decoded_groups.setdefault("history_weight_new_bounded", []).append(values.flatten())
            print_summary("history_weight_new_bounded", values, args.show_values, args.max_preview)
        if mode in {"old", "both"}:
            values = decode_old(raw)
            decoded_groups.setdefault("history_weight_old_softplus", []).append(values.flatten())
            print_summary("history_weight_old_softplus", values, args.show_values, args.max_preview)

    if len(raw_items) > 1:
        print("=" * 88)
        print("overall")
        for label, tensors in decoded_groups.items():
            import torch

            print_summary(label, torch.cat(tensors), args.show_values, args.max_preview)


if __name__ == "__main__":
    main()
