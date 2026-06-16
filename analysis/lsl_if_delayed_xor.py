#!/usr/bin/env python3
"""Delayed spiking XOR (ROX) experiment for LSLIF.

This file mirrors the checked-in DH-SNN delayed-XOR experiment without modifying
``DH-SNN/delayed_xor``.  It intentionally keeps only the outputs that the DH-SNN
script has: console training logs, an in-memory ``acc_list`` return value, the
forward-pass readout sequence (``d2_output``), and optional best-model saving.
No additional figures or analysis artifacts are generated here.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from modules.neuron import LSLIFNeuron
from modules.surrogate import Rectangle


@dataclass(frozen=True)
class DelayedXORConfig:
    time_steps: int = 200
    channel_rates: Tuple[float, float] = (0.2, 0.6)
    noise_rate: float = 0.01
    channel_size: int = 20
    coding_time: int = 10
    test_time: int = 1
    batch_size: int = 500
    hidden_dims: int = 16
    output_dim: int = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def xor_label_matrix(device: torch.device) -> torch.Tensor:
    label = torch.zeros(2, 2, device=device, dtype=torch.long)
    label[1, 0] = 1
    label[0, 1] = 1
    return label


def get_batch(cfg: DelayedXORConfig, device: torch.device):
    """Generate the delayed spiking XOR dataset with DH-SNN-equivalent controls."""
    values = torch.rand(cfg.batch_size, cfg.time_steps, cfg.channel_size, device=device) <= cfg.noise_rate
    targets = torch.zeros(cfg.time_steps, cfg.batch_size, device=device, dtype=torch.int64)
    rates = torch.tensor(cfg.channel_rates, device=device, dtype=torch.float32)

    init_pattern = torch.randint(len(cfg.channel_rates), size=(cfg.batch_size,), device=device)
    prob_matrix = torch.ones(cfg.coding_time, cfg.channel_size, cfg.batch_size, device=device) * rates[init_pattern]
    add_patterns = torch.bernoulli(prob_matrix).permute(2, 0, 1).bool()
    values[:, : cfg.coding_time, :] = values[:, : cfg.coding_time, :] | add_patterns

    position = torch.randint(cfg.test_time, size=(cfg.batch_size,), device=device)
    pattern = torch.randint(len(cfg.channel_rates), size=(cfg.batch_size,), device=device)
    label_t = xor_label_matrix(device)[init_pattern, pattern]
    prob_matrix = torch.ones(cfg.coding_time, cfg.channel_size, cfg.batch_size, device=device) * rates[pattern]
    add_patterns = torch.bernoulli(prob_matrix).permute(2, 0, 1).bool()

    for i in range(cfg.batch_size):
        start = cfg.time_steps - (int(position[i].item()) + 1) * cfg.coding_time
        end = cfg.time_steps - int(position[i].item()) * cfg.coding_time
        values[i, start:end, :] = values[i, start:end, :] | add_patterns[i]
        targets[start:, i] = label_t[i]

    return values.float(), targets.transpose(0, 1).contiguous(), position


class LIFStep(nn.Module):
    """Minimal vanilla LIF step for the protocol-matched local control."""

    def __init__(self, tau: float = 2.0, threshold: float = 1.0):
        super().__init__()
        self.tau = float(tau)
        self.threshold = float(threshold)
        self.surrogate = Rectangle()
        self.v = None

    def reset(self):
        self.v = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.v is None or self.v.shape != x.shape or self.v.device != x.device:
            self.v = torch.zeros_like(x, dtype=torch.float32)
        decay = max(0.0, min(1.0, 1.0 - 1.0 / self.tau))
        mem = self.v * decay + x.float()
        spike = self.surrogate(mem - self.threshold)
        self.v = mem - spike * self.threshold
        return spike.to(dtype=x.dtype)


class DelayedXORNet(nn.Module):
    def __init__(self, cfg: DelayedXORConfig, neuron: str, args: argparse.Namespace):
        super().__init__()
        self.cfg = cfg
        self.input = nn.Linear(cfg.channel_size, cfg.hidden_dims, bias=True)
        if neuron == "lslif":
            self.neuron = LSLIFNeuron(
                tau=args.tau,
                v_threshold=args.v_threshold,
                detach_reset=args.detach_reset,
                history_weight=args.history_weight,
                history_power=args.history_power,
                history_eps=args.history_eps,
                history_learn_weight=args.history_learn_weight,
                history_weight_lo=args.history_weight_lo,
                history_weight_hi=args.history_weight_hi,
                history_weight_per_step=args.history_weight_per_step,
                history_max_steps=cfg.time_steps,
                history_learn_power=args.history_learn_power,
                history_mode=args.history_mode,
            )
        elif neuron == "lif":
            self.neuron = LIFStep(tau=args.tau, threshold=args.v_threshold)
        else:
            raise ValueError(f"Unsupported neuron: {neuron}")
        self.readout = nn.Linear(cfg.hidden_dims, cfg.output_dim, bias=True)
        self.criterion = nn.CrossEntropyLoss()

    def reset_state(self):
        self.neuron.reset()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, position: torch.Tensor):
        batch_size, seq_num, _ = inputs.shape
        self.reset_state()
        d2_output = torch.zeros(batch_size, seq_num, self.cfg.output_dim)
        loss = inputs.new_tensor(0.0)
        correct = 0
        total = 0
        for t in range(seq_num):
            hidden_current = self.input(inputs[:, t, :])
            hidden_spike = self.neuron(hidden_current)
            mem_layer2 = self.readout(hidden_spike)
            d2_output[:, t, :] = mem_layer2.detach().cpu()

            active = t > (self.cfg.time_steps - (position + 1) * self.cfg.coding_time)
            if bool(active.any()):
                # Match DH-SNN: softmax is applied before CrossEntropyLoss.
                output = F.softmax(mem_layer2, dim=1)
                loss = loss + self.criterion(output[active], targets[active, t].long())
                predicted = torch.max(output[active].detach(), dim=1).indices.cpu().t()
                labels = targets[active, t].cpu()
                correct += int((predicted == labels).sum().item())
                total += int(labels.numel())
        return loss, d2_output, correct, total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Protocol-matched LSLIF delayed spiking XOR experiment.")
    p.add_argument("--neuron", choices=["lslif", "lif"], default="lslif")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batches-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--hidden-dims", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-best", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("analysis/results/delayed_xor_lslif/model"))
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--v-threshold", type=float, default=1.0)
    p.add_argument("--detach-reset", action="store_true")
    p.add_argument("--history-weight", type=float, default=1.0)
    p.add_argument("--history-power", type=float, default=1.0)
    p.add_argument("--history-eps", type=float, default=1e-6)
    p.add_argument("--history-learn-weight", action="store_true")
    p.add_argument("--history-weight-lo", type=float, default=-0.8)
    p.add_argument("--history-weight-hi", type=float, default=0.8)
    p.add_argument("--history-weight-per-step", action="store_true")
    p.add_argument("--history-learn-power", action="store_true")
    p.add_argument("--history-mode", choices=["all", "post_spike"], default="all")
    return p.parse_args()


def train(model: DelayedXORNet, cfg: DelayedXORConfig, args: argparse.Namespace, device: torch.device):
    acc_list = []
    best_loss = 150
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    if args.save_best:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0.0
        sum_correct = 0
        model.train()

        for _ in range(args.batches_per_epoch):
            data, target, position = get_batch(cfg, device)
            optimizer.zero_grad()
            loss, output, correct, total = model(data, target, position)
            loss.backward()
            train_loss_sum += loss.item()
            optimizer.step()
            sum_correct += correct
            sum_sample += total

        scheduler.step()
        acc_list.append(train_acc)
        print('lr: ', optimizer.param_groups[0]["lr"])

        if args.save_best and train_loss_sum < best_loss * args.batches_per_epoch:
            best_loss = train_loss_sum / args.batches_per_epoch
            name = f'{args.neuron}_sfnn_time{cfg.time_steps}'
            torch.save(model, args.output_dir / f'{name}{str(best_loss)[:7]}-srnn-shd.pth')

        print(
            'log_internel:{:3d}, epoch: {:3d}, Train Loss: {:.4f}, Acc: {:.3f}'.format(
                args.batches_per_epoch,
                epoch,
                train_loss_sum / args.batches_per_epoch,
                sum_correct / sum_sample,
            ),
            flush=True,
        )

    return acc_list


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print("device:", device)
    cfg = DelayedXORConfig(batch_size=args.batch_size, hidden_dims=args.hidden_dims)
    model = DelayedXORNet(cfg, args.neuron, args).to(device)
    train(model, cfg, args, device)


if __name__ == "__main__":
    main()
