#!/usr/bin/env python3
"""Compare temporal input gradients from LS and non-LS checkpoints.

The command performs exactly one evaluation forward/backward pass per checkpoint;
it does not train or update either model.
"""

import argparse
import ast
import importlib.util
import json
import random
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    # Executing ``python analysis/analyze_temporal_gradient.py`` otherwise puts
    # only ``analysis/`` on sys.path, so project modules such as ``utils`` and
    # ``models`` cannot be imported.
    sys.path.insert(0, str(REPOSITORY_ROOT))

RUNTIME_MODULES = ("torch", "torchvision", "spikingjelly", "numpy", "matplotlib")


def parse_namespace(text):
    """Parse the first ``Namespace(...)`` in an args log without executing it."""
    start = text.find("Namespace(")
    if start < 0:
        return {}
    depth = 0
    quote = None
    escaped = False
    end = None
    for offset, character in enumerate(text[start:]):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in ("'", '"'):
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                end = start + offset + 1
                break
    if end is None:
        raise ValueError("Unterminated Namespace record.")
    expression = ast.parse(text[start:end], mode="eval").body
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError("Invalid Namespace record.")
    if expression.func.id != "Namespace" or expression.args:
        raise ValueError("Invalid Namespace record.")
    return {keyword.arg: ast.literal_eval(keyword.value) for keyword in expression.keywords}


def load_config(run_dir):
    run_dir = Path(run_dir)
    config = {}
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(summary, dict):
            config.update(summary.get("args", summary))
    args_path = run_dir / "args.txt"
    if args_path.exists():
        # Namespace is authoritative because it records the actual resumed command.
        config.update(parse_namespace(args_path.read_text(encoding="utf-8")))
    if not config:
        raise ValueError(f"No readable configuration found in {run_dir}.")
    return config


def evenly_spaced_indices(total, maximum):
    if total <= 0 or maximum <= 0:
        raise ValueError("Neuron counts must be positive.")
    if total <= maximum:
        return list(range(total))
    if maximum == 1:
        return [total // 2]
    return [round(index * (total - 1) / (maximum - 1)) for index in range(maximum)]


def _checkpoint_path(run_dir, name):
    path = Path(run_dir) / name
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _validate_pair(ls_config, baseline_config):
    for key in ("dataset", "model", "T"):
        if ls_config.get(key) != baseline_config.get(key):
            raise ValueError(
                f"LS and baseline must use the same {key}: "
                f"{ls_config.get(key)!r} != {baseline_config.get(key)!r}.")
    if ls_config.get("dataset") != "DVSCIFAR10":
        raise ValueError("Temporal-gradient analysis currently supports dataset=DVSCIFAR10.")
    for key in ("loss_lambda", "mse_n_reg", "loss_means", "label_smoothing"):
        if ls_config.get(key, 0) != baseline_config.get(key, 0):
            raise ValueError(f"Loss setting {key} differs between the two experiments.")


def _require_runtime():
    missing = [name for name in RUNTIME_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Missing runtime dependencies: " + ", ".join(missing) +
            ". Install the training environment and analysis/requirements.txt.")


def _install_numpy_legacy_aliases(np):
    """Restore aliases required by this repository's older data dependencies."""
    for name, value in (("object", object), ("bool", bool), ("int", int)):
        if name not in np.__dict__:
            setattr(np, name, value)
    if "typeDict" not in np.__dict__ and "sctypeDict" in np.__dict__:
        np.typeDict = np.sctypeDict


def _surrogate(config, surrogate_sj, surrogate_self):
    name = config.get("surrogate", "rectangle")
    if name == "sigmoid":
        return surrogate_sj.Sigmoid()
    if name == "rectangle":
        return surrogate_self.Rectangle()
    if name == "triangle":
        return surrogate_sj.PiecewiseQuadratic()
    raise ValueError(f"Unsupported surrogate: {name}")


def _build_model(config, device):
    from models import spiking_resnet, spiking_vgg_bn, vgg_model
    from modules import neuron
    from modules import surrogate as surrogate_self
    from spikingjelly.clock_driven import surrogate as surrogate_sj

    neuron_name = config.get("neuron_model")
    neurons = {"LIF": neuron.VanillaLIFNeuron, "LSLIF": neuron.LSLIFNeuron}
    if neuron_name not in neurons:
        raise ValueError(f"Only LIF and LSLIF checkpoints are supported, got {neuron_name!r}.")
    kwargs = {
        "tau": config.get("tau", 2.0),
        "decay_input": config.get("decay_input", False),
        "v_threshold": config.get("v_threshold", 1.0),
        "detach_reset": config.get("detach_reset", False),
        "surrogate_function": _surrogate(config, surrogate_sj, surrogate_self),
        "history_weight": config.get("history_weight", 1.0),
        "history_power": config.get("history_power", 1.0),
        "history_eps": config.get("history_eps", 1e-6),
        "history_learn_weight": config.get("history_learn_weight", False),
        "history_weight_lo": config.get("history_weight_lo"),
        "history_weight_hi": config.get("history_weight_hi"),
        "history_weight_per_step": config.get("history_weight_per_step", False),
        "history_max_steps": config["T"],
        "history_learn_power": config.get("history_learn_power", False),
        "history_mode": config.get("history_mode", "all"),
    }
    model_name = config["model"]
    common = dict(neuron=neurons[neuron_name], num_classes=10,
                  neuron_dropout=config.get("drop_rate", 0.0), c_in=2, **kwargs)
    if model_name.startswith("spiking_resnet") and model_name in spiking_resnet.__dict__:
        model = spiking_resnet.__dict__[model_name](fc_hw=1, **common)
    elif model_name.startswith("spiking_vgg") and model_name in spiking_vgg_bn.__dict__:
        model = spiking_vgg_bn.__dict__[model_name](fc_hw=48, **common)
    elif model_name in vgg_model.__dict__:
        model = vgg_model.__dict__[model_name](fc_hw=48, **common)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(device)


def _load_weights(model, checkpoint_path, device, torch):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("net", checkpoint)
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def _load_test_batch(config, data_dir, batch_size, batch_index, workers, seed, torch):
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from utils.augmentation import Resize, ToPILImage, ToTensor
    from utils.cifar10_dvs import CIFAR10DVS

    transform = transforms.Compose([ToPILImage(), Resize(48), ToTensor()])
    dataset = CIFAR10DVS(str(data_dir), train=False, use_frame=True,
                         frames_num=config["T"], split_by="number",
                         normalization=None, transform=transform)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, generator=generator)
    for index, batch in enumerate(loader):
        if index == batch_index:
            return batch
    raise ValueError(f"batch-index {batch_index} exceeds the test loader ({len(loader)} batches).")


def _training_loss(outputs, labels, config, torch, time_steps=None):
    import torch.nn.functional as functional

    time_steps = config["T"] if time_steps is None else time_steps
    repeated_labels = torch.cat([labels for _ in range(time_steps)], dim=0)
    cross_entropy = functional.cross_entropy(
        outputs, repeated_labels, label_smoothing=config.get("label_smoothing", 0.0))
    loss_lambda = config.get("loss_lambda", 0.0)
    if loss_lambda <= 0:
        return cross_entropy
    if config.get("mse_n_reg", False):
        target = functional.one_hot(repeated_labels, num_classes=10).to(outputs.dtype)
    else:
        target = torch.zeros_like(outputs).fill_(config.get("loss_means", 1.0))
    return (1.0 - loss_lambda) * cross_entropy + loss_lambda * functional.mse_loss(outputs, target)


def _time_frame(frames, time_step, expected_steps):
    """Return one time slice from the list produced by the DVS transforms."""
    if isinstance(frames, (list, tuple)):
        if len(frames) != expected_steps:
            raise ValueError(
                f"Expected {expected_steps} frame tensors, received {len(frames)}.")
        return frames[time_step]
    # Retain support for an explicitly time-major tensor supplied by a custom
    # collate function. The repository's default DVS transform returns a list.
    if frames.shape[0] != expected_steps:
        raise ValueError(
            f"Expected time-major frames with first dimension {expected_steps}, "
            f"received shape {tuple(frames.shape)}.")
    return frames[time_step]


def _gradient_matrix(config, checkpoint, batch, layer_name, sample_index,
                     gradient_target, gradient_source, aggregation, device, torch):
    from spikingjelly.clock_driven import functional

    model = _build_model(config, device)
    _load_weights(model, checkpoint, device, torch)
    modules = dict(model.named_modules())
    if layer_name not in modules:
        candidates = [name for name, module in modules.items()
                      if module.__class__.__name__ in ("VanillaLIFNeuron", "LSLIFNeuron")]
        raise ValueError(f"Layer {layer_name!r} not found. Available neuron layers: {candidates}")
    captured = []

    def retain(value):
        if not value.requires_grad:
            raise RuntimeError(f"Selected value at {layer_name} does not require gradients.")
        value.retain_grad()
        captured.append(value)

    def capture_input(_module, inputs):
        value = inputs[0]
        retain(value)

    def capture_state(module, _inputs, _output):
        if not hasattr(module, "last_v_pre"):
            raise RuntimeError(f"Neuron layer {layer_name} does not expose its pre-spike membrane.")
        retain(module.last_v_pre)

    target_module = modules[layer_name]
    if gradient_source == "state":
        # LSLIF only retains its fused membrane while this explicit diagnostic
        # flag is enabled, so ordinary training has no extra graph reference.
        target_module.gradient_probe_enabled = True
        handle = target_module.register_forward_hook(capture_state)
    else:
        handle = target_module.register_forward_pre_hook(capture_input)
    frames, labels = batch
    if sample_index >= labels.shape[0]:
        raise ValueError(f"sample-index {sample_index} is outside batch size {labels.shape[0]}.")
    labels = labels.to(device)
    model.eval()
    model.zero_grad(set_to_none=True)
    functional.reset_net(model)
    outputs = torch.cat([
        model(_time_frame(frames, t, config["T"]).float().to(device))
        for t in range(config["T"])
    ], dim=0)
    if gradient_target == "final":
        # A final-step objective measures genuine BPTT propagation: gradients
        # can reach earlier inputs only through the neuron's temporal state.
        loss = _training_loss(outputs[-labels.shape[0]:], labels, config, torch, time_steps=1)
    else:
        loss = _training_loss(outputs, labels, config, torch)
    loss.backward()
    handle.remove()
    if gradient_source == "state":
        target_module.gradient_probe_enabled = False
        target_module.last_v_pre = None
    if len(captured) != config["T"] or any(value.grad is None for value in captured):
        raise RuntimeError(f"Expected {config['T']} temporal gradients, captured {len(captured)}.")
    if aggregation == "batch-mean-abs":
        columns = [value.grad.detach().abs().mean(dim=0).reshape(-1).cpu() for value in captured]
    else:
        columns = [value.grad[sample_index].detach().reshape(-1).cpu() for value in captured]
    matrix = torch.stack(columns, dim=1)
    functional.reset_net(model)
    return matrix, float(loss.detach().cpu())


def _display_cmap(normalization):
    return "Blues" if normalization == "per-neuron" else "RdBu_r"


def _plot(ls_matrix, baseline_matrix, indices, layer, output_dir, args, np, plt):
    from matplotlib.colors import SymLogNorm

    combined = np.concatenate([np.abs(ls_matrix).ravel(), np.abs(baseline_matrix).ravel()])
    limit = float(np.percentile(combined, args.gradient_percentile))
    if limit <= 0:
        limit = float(combined.max()) if combined.size and combined.max() > 0 else 1.0
    plt.rcParams.update({"font.family": "serif", "font.size": 16, "axes.titleweight": "bold"})
    difference = ls_matrix - baseline_matrix
    figure, axes = plt.subplots(1, 3, figsize=(args.fig_width, args.fig_height),
                                sharex=True, sharey=True, constrained_layout=True)
    norm = None
    cmap = _display_cmap(args.normalization)
    image_limits = {"vmin": -limit, "vmax": limit}
    if args.normalization == "per-neuron":
        # A sequential white-to-blue map makes zero gradients white and uses
        # progressively darker blue for stronger propagation, matching the
        # conventional normalized-gradient heatmap style without neon colors.
        image_limits = {"vmin": 0.0, "vmax": 1.0}
    elif args.color_scale == "symlog":
        # Temporal gradients often span several orders of magnitude. A shared
        # symmetric-log scale reveals later steps without normalizing columns
        # independently or destroying the LS/non-LS magnitude comparison.
        norm = SymLogNorm(linthresh=max(limit * 1e-3, 1e-30), linscale=1.0,
                          vmin=-limit, vmax=limit, base=10, clip=True)
    images = []
    for axis, matrix, title in zip(
            axes[:2], (baseline_matrix, ls_matrix), ("Non-LS (LIF)", "LS (LSLIF)")):
        image_kwargs = {"norm": norm} if norm is not None else image_limits
        image = axis.imshow(matrix, aspect="auto", cmap=cmap,
                            interpolation="nearest", origin="upper", **image_kwargs)
        images.append(image)
        axis.set_title(title, pad=14)
        axis.set_xlabel("Time step")
        axis.set_xticks(range(matrix.shape[1]))
        axis.set_xticklabels(range(1, matrix.shape[1] + 1))
    if args.normalization == "per-neuron":
        difference_limit = 1.0
    else:
        difference_limit = float(np.percentile(np.abs(difference), args.gradient_percentile))
        if difference_limit <= 0:
            difference_limit = float(np.abs(difference).max()) or 1.0
    difference_image = axes[2].imshow(
        difference, aspect="auto", cmap="RdBu_r", vmin=-difference_limit,
        vmax=difference_limit, interpolation="nearest", origin="upper")
    axes[2].set_title("Difference (LS − Non-LS)", pad=14)
    axes[2].set_xlabel("Time step")
    axes[2].set_xticks(range(difference.shape[1]))
    axes[2].set_xticklabels(range(1, difference.shape[1] + 1))
    axes[0].set_ylabel("Sampled neuron index")
    target_label = "final-step loss" if args.gradient_target == "final" else "all-step loss"
    source_label = "membrane" if args.gradient_source == "state" else "input"
    figure.suptitle(f"Temporal {source_label}-gradient propagation at {layer} ({target_label})",
                    fontsize=20, fontweight="bold")
    colorbar = figure.colorbar(images[0], ax=axes[:2], shrink=0.88, pad=0.02)
    colorbar.set_label("Per-neuron normalized |gradient|" if args.normalization == "per-neuron"
                       else f"{source_label.capitalize()} gradient")
    difference_colorbar = figure.colorbar(difference_image, ax=axes[2], shrink=0.88, pad=0.02)
    difference_colorbar.set_label(
        "Normalized gradient difference (LS − Non-LS)"
        if args.normalization == "per-neuron" else "Gradient difference (LS − Non-LS)")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "temporal_gradient_comparison.png", dpi=args.dpi,
                   bbox_inches="tight", facecolor="white")
    figure.savefig(output_dir / "temporal_gradient_comparison.svg", bbox_inches="tight",
                   facecolor="white")
    plt.close(figure)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ls-run", required=True, type=Path)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, help="Override the dataset path recorded in args.txt.")
    parser.add_argument("--layer", default="layer3.6", help="Common neuron layer to inspect.")
    parser.add_argument("--output-dir", type=Path, default=Path("gradient_analysis"))
    parser.add_argument("--checkpoint-name", default="checkpoint_max.pth")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-neurons", type=int, default=512)
    parser.add_argument("--gradient-percentile", type=float, default=99.0)
    parser.add_argument("--gradient-target", choices=("final", "all"), default="final",
                        help="Backpropagate final-step loss (default) or the training loss at every step.")
    parser.add_argument("--gradient-source", choices=("state", "input"), default="state",
                        help="Inspect the pre-spike membrane state (default) or layer input.")
    parser.add_argument("--aggregation", choices=("batch-mean-abs", "sample-signed"),
                        default="batch-mean-abs",
                        help="Aggregate absolute gradients over the fixed batch (default) or one sample.")
    parser.add_argument("--normalization", choices=("per-neuron", "none"), default="per-neuron",
                        help="Normalize each neuron's temporal profile for the display (default).")
    parser.add_argument("--color-scale", choices=("symlog", "linear"), default="symlog",
                        help="Shared signed color scale; symlog reveals small temporal gradients.")
    parser.add_argument("--device", help="Default: cuda:0 when available, otherwise cpu.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--fig-width", type=float, default=21.0)
    parser.add_argument("--fig-height", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if min(args.batch_size, args.max_neurons, args.fig_width, args.fig_height, args.dpi) <= 0:
        raise ValueError("Batch size, neuron count, figure dimensions, and DPI must be positive.")
    if not 0 < args.gradient_percentile <= 100:
        raise ValueError("gradient-percentile must be in (0, 100].")
    ls_config, baseline_config = load_config(args.ls_run), load_config(args.baseline_run)
    _validate_pair(ls_config, baseline_config)
    data_dir = args.data_dir or ls_config.get("data_dir")
    if not data_dir:
        raise ValueError("No dataset path found; pass --data-dir.")
    _require_runtime()
    import numpy as np
    _install_numpy_legacy_aliases(np)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    batch = _load_test_batch(ls_config, data_dir, args.batch_size, args.batch_index,
                             args.num_workers, args.seed, torch)
    ls_checkpoint = _checkpoint_path(args.ls_run, args.checkpoint_name)
    baseline_checkpoint = _checkpoint_path(args.baseline_run, args.checkpoint_name)
    baseline_full, baseline_loss = _gradient_matrix(
        baseline_config, baseline_checkpoint, batch, args.layer, args.sample_index,
        args.gradient_target, args.gradient_source, args.aggregation, device, torch)
    ls_full, ls_loss = _gradient_matrix(
        ls_config, ls_checkpoint, batch, args.layer, args.sample_index,
        args.gradient_target, args.gradient_source, args.aggregation, device, torch)
    if ls_full.shape != baseline_full.shape:
        raise ValueError(f"Gradient shapes differ: LS {tuple(ls_full.shape)}, baseline {tuple(baseline_full.shape)}.")
    indices = evenly_spaced_indices(ls_full.shape[0], args.max_neurons)
    ls_raw = ls_full[indices].numpy()
    baseline_raw = baseline_full[indices].numpy()
    if args.normalization == "per-neuron":
        ls_scale = np.max(np.abs(ls_raw), axis=1, keepdims=True)
        baseline_scale = np.max(np.abs(baseline_raw), axis=1, keepdims=True)
        ls_matrix = np.abs(ls_raw) / np.maximum(ls_scale, 1e-30)
        baseline_matrix = np.abs(baseline_raw) / np.maximum(baseline_scale, 1e-30)
    else:
        ls_matrix, baseline_matrix = ls_raw, baseline_raw
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "temporal_gradients.npz",
                        ls_gradient_raw=ls_raw, baseline_gradient_raw=baseline_raw,
                        gradient_difference_raw=ls_raw - baseline_raw,
                        ls_gradient_display=ls_matrix, baseline_gradient_display=baseline_matrix,
                        gradient_difference_display=ls_matrix - baseline_matrix,
                        neuron_indices=np.asarray(indices), layer=np.asarray(args.layer),
                        sample_index=np.asarray(args.sample_index), batch_index=np.asarray(args.batch_index),
                        gradient_target=np.asarray(args.gradient_target),
                        gradient_source=np.asarray(args.gradient_source),
                        aggregation=np.asarray(args.aggregation),
                        normalization=np.asarray(args.normalization),
                        color_scale=np.asarray(args.color_scale),
                        ls_loss=np.asarray(ls_loss), baseline_loss=np.asarray(baseline_loss))
    _plot(ls_matrix, baseline_matrix, indices, args.layer, args.output_dir, args, np, plt)
    print(f"Saved temporal-gradient comparison to {args.output_dir}")
    print(f"Layer={args.layer}, neurons={len(indices)}, time_steps={ls_matrix.shape[1]}, "
          f"baseline_loss={baseline_loss:.6f}, ls_loss={ls_loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
