#!/usr/bin/env python3
"""Inspect learned LSLIF auxiliary branch coefficients in a checkpoint.

LSLIF stores the learnable auxiliary branch coefficient as ``history_weight_raw``.
The effective coefficient used by the neuron is bounded by:

    beta = lo + (hi - lo) * sigmoid(history_weight_raw)

where ``lo`` and ``hi`` should match the training arguments
``-history_weight_lo`` and ``-history_weight_hi``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


DEFAULT_HISTORY_WEIGHT_LO = -0.8
DEFAULT_HISTORY_WEIGHT_HI = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the effective LSLIF auxiliary history-branch coefficient "
            "from checkpoint['net'] or a plain state_dict."
        )
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to checkpoint_latest.pth/checkpoint_max.pth or a plain state_dict .pth file.",
    )
    parser.add_argument(
        "--history-weight-lo",
        type=float,
        default=DEFAULT_HISTORY_WEIGHT_LO,
        help="Lower bound used during training by -history_weight_lo. Default: %(default)s.",
    )
    parser.add_argument(
        "--history-weight-hi",
        type=float,
        default=DEFAULT_HISTORY_WEIGHT_HI,
        help="Upper bound used during training by -history_weight_hi. Default: %(default)s.",
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Print every effective beta value for each tensor. Useful for per-step coefficients.",
    )
    parser.add_argument(
        "--max-preview",
        type=int,
        default=16,
        help="Number of values to preview when --show-values is not set. Default: %(default)s.",
    )
    parser.add_argument(
        "--assume-fixed-history-weight",
        type=float,
        default=None,
        help=(
            "Value to report when no learned history_weight_raw tensors are found. "
            "Use this if the model was trained without -history_learn_weight."
        ),
    )
    return parser.parse_args()


def load_state_dict(checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata: dict[str, Any] = {}

    if isinstance(checkpoint, dict) and "net" in checkpoint:
        state_dict = checkpoint["net"]
        for key in ("epoch", "max_test_acc"):
            if key in checkpoint:
                metadata[key] = checkpoint[key]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported state_dict type: {type(state_dict)!r}")

    return state_dict, metadata


def effective_history_weight(raw: Any, lo: float, hi: float) -> Any:
    import torch

    return lo + (hi - lo) * torch.sigmoid(raw.detach().float())


def sign_label(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def format_values(values: Any, show_all: bool, max_preview: int) -> str:
    flat = values.flatten()
    shown = flat if show_all else flat[: max(0, max_preview)]
    as_list = [float(v) for v in shown]
    if show_all or flat.numel() <= shown.numel():
        return str(as_list)
    return f"{as_list} ... ({flat.numel()} total)"


def print_tensor_report(name: str, raw: Any, beta: Any, show_values: bool, max_preview: int) -> None:
    flat = beta.flatten()
    n = flat.numel()
    n_pos = int((flat > 0).sum().item())
    n_neg = int((flat < 0).sum().item())
    n_zero = int((flat == 0).sum().item())

    print("=" * 88)
    print(name)
    print(f"  raw shape: {tuple(raw.shape)}")
    print(f"  beta shape: {tuple(beta.shape)}")
    print(f"  beta min/max/mean: {float(flat.min()):.8g} / {float(flat.max()):.8g} / {float(flat.mean()):.8g}")
    print(f"  sign counts: positive={n_pos}/{n}, negative={n_neg}/{n}, zero={n_zero}/{n}")
    print(f"  beta preview: {format_values(flat, show_values, max_preview)}")


def main() -> None:
    args = parse_args()
    if args.history_weight_hi <= args.history_weight_lo:
        raise ValueError("--history-weight-hi must be larger than --history-weight-lo")

    state_dict, metadata = load_state_dict(args.checkpoint)
    raw_items = [(name, value) for name, value in state_dict.items() if "history_weight_raw" in name]

    print(f"checkpoint: {args.checkpoint}")
    for key, value in metadata.items():
        print(f"{key}: {value}")
    print(
        "effective beta transform: "
        f"beta = {args.history_weight_lo} + "
        f"({args.history_weight_hi} - {args.history_weight_lo}) * sigmoid(history_weight_raw)"
    )

    if not raw_items:
        print("\nNo history_weight_raw tensors were found in this checkpoint.")
        print("This usually means the run did not enable -history_learn_weight, so beta was fixed.")
        if args.assume_fixed_history_weight is not None:
            fixed = float(args.assume_fixed_history_weight)
            print(f"Assumed fixed beta: {fixed:.8g} ({sign_label(fixed)})")
        else:
            print("Pass --assume-fixed-history-weight VALUE if you know the fixed -history_weight used for training.")
        return

    all_beta = []
    print(f"\nFound {len(raw_items)} learned history_weight_raw tensor(s).")
    for name, raw in raw_items:
        beta = effective_history_weight(raw, args.history_weight_lo, args.history_weight_hi)
        all_beta.append(beta.flatten())
        print_tensor_report(name, raw, beta, args.show_values, args.max_preview)

    import torch

    merged = torch.cat(all_beta)
    n = merged.numel()
    print("=" * 88)
    print("overall")
    print(f"  beta min/max/mean: {float(merged.min()):.8g} / {float(merged.max()):.8g} / {float(merged.mean()):.8g}")
    print(
        "  sign counts: "
        f"positive={int((merged > 0).sum().item())}/{n}, "
        f"negative={int((merged < 0).sum().item())}/{n}, "
        f"zero={int((merged == 0).sum().item())}/{n}"
    )


if __name__ == "__main__":
    main()
