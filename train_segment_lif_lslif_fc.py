# -*- coding: utf-8 -*-
"""被试内 EEG segment 数据上的单层全连接 SNN baseline：LIF vs LSLIF。

本文件是一个自包含实验脚本：不依赖项目内的 logs、model 或 modules 文件。
默认数据路径沿用 train_segment.py 的 Self-data/new-segment 被试内划分；如需换路径，
只需要修改命令行参数 --train-dir / --test-dir 或下方 DEFAULT_* 常量。

输入数据约定：.npy 文件为 dict，包含：
  - fea: [N, 61, 750]
  - label: [N]
脚本会把每个样本按 750 个原始采样点展开为 SNN 时间步：
  [B, 1, 61, 750] -> [750, B, 61]
并分别训练 LIF 与 LSLIF 两个单层全连接网络进行对比。
"""

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset


DEFAULT_TRAIN_DIR = "/home/guyue/zhao/PythonProject/dataset/Self-data/kFold/kFold(61x750)/new-segment/train/"
DEFAULT_TEST_DIR = "/home/guyue/zhao/PythonProject/dataset/Self-data/kFold/kFold(61x750)/new-segment/test/"
DEFAULT_OUTPUT_DIR = "all_result/simple_snn_fc/segment_lif_vs_lslif/"


class RectangleSpike(torch.autograd.Function):
    """与项目 modules.surrogate.Rectangle 一致的矩形替代梯度。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_x = None
        if ctx.needs_input_grad[0]:
            (x,) = ctx.saved_tensors
            alpha = ctx.alpha
            mask = x.abs() <= (alpha / 2.0)
            grad_x = grad_output * mask.to(grad_output.dtype) / alpha
        return grad_x, None


def rectangle_spike(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return RectangleSpike.apply(x, alpha)


class StatefulNeuron(nn.Module):
    def reset(self):
        raise NotImplementedError


class LIFNeuron(StatefulNeuron):
    """项目 VanillaLIFNeuron 的轻量自包含版本。"""

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = None,
        detach_reset: bool = False,
        tau_eps: float = 1e-6,
    ):
        super().__init__()
        self.tau = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.detach_reset = bool(detach_reset)
        self.tau_eps = float(tau_eps)
        self.v = None

    def reset(self):
        self.v = None

    def _ensure_state(self, x: torch.Tensor):
        if self.v is None or self.v.shape != x.shape or self.v.device != x.device:
            self.v = torch.zeros_like(x, dtype=torch.float32, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)
        tau_eff = torch.as_tensor(self.tau, device=x.device, dtype=torch.float32)
        if self.decay_input:
            self.v = self.v + (x_f - self.v) / (tau_eff + self.tau_eps)
        else:
            decay = torch.clamp(1.0 - 1.0 / (tau_eff + self.tau_eps), 0.0, 1.0)
            self.v = self.v * decay + x_f

        th = torch.as_tensor(self.v_threshold, device=x.device, dtype=torch.float32)
        spike = rectangle_spike(self.v - th)
        reset_spike = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - reset_spike * th
        else:
            self.v = torch.where(reset_spike.bool(), torch.as_tensor(self.v_reset, device=x.device), self.v)
        return spike.to(dtype=x.dtype)


class LSLIFNeuron(StatefulNeuron):
    """项目 LSLIFNeuron 的核心机制自包含版本。

    主膜电位 v 正常泄漏积分并在发放后重置；辅助膜 n 用相同泄漏积分但不发放、
    不重置。发放前融合 M_t = v_t + beta * n_t / step_t ** power。
    """

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = None,
        detach_reset: bool = False,
        tau_eps: float = 1e-6,
        history_weight: float = 1.0,
        history_power: float = 1.0,
        history_eps: float = 1e-6,
        history_mode: str = "all",
    ):
        super().__init__()
        self.tau = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.detach_reset = bool(detach_reset)
        self.tau_eps = float(tau_eps)
        self.history_weight = float(history_weight)
        self.history_power = float(history_power)
        self.history_eps = float(history_eps)
        self.history_mode = history_mode.lower()
        if self.history_mode not in {"all", "post_spike"}:
            raise ValueError("history_mode must be 'all' or 'post_spike' in this standalone script")
        self.v = None
        self.n = None
        self.has_fired = None
        self.step_count = 0

    def reset(self):
        self.v = None
        self.n = None
        self.has_fired = None
        self.step_count = 0

    def _ensure_state(self, x: torch.Tensor):
        if self.v is None or self.v.shape != x.shape or self.v.device != x.device:
            self.v = torch.zeros_like(x, dtype=torch.float32, device=x.device)
            self.n = torch.zeros_like(x, dtype=torch.float32, device=x.device)
            self.has_fired = torch.zeros_like(x, dtype=torch.bool, device=x.device)
            self.step_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_state(x)
        x_f = x.to(torch.float32)
        tau_eff = torch.as_tensor(self.tau, device=x.device, dtype=torch.float32)
        if self.decay_input:
            m_t = self.v + (x_f - self.v) / (tau_eff + self.tau_eps)
            n_t = self.n + (x_f - self.n) / (tau_eff + self.tau_eps)
        else:
            decay = torch.clamp(1.0 - 1.0 / (tau_eff + self.tau_eps), 0.0, 1.0)
            m_t = self.v * decay + x_f
            n_t = self.n * decay + x_f

        self.step_count += 1
        step_t = torch.as_tensor(float(self.step_count), device=x.device, dtype=torch.float32)
        norm = torch.pow(step_t + self.history_eps, torch.as_tensor(self.history_power, device=x.device))
        history_term = self.history_weight * (n_t / norm)
        if self.history_mode == "post_spike":
            history_term = history_term * self.has_fired.to(dtype=history_term.dtype)
        total_mem = m_t + history_term

        th = torch.as_tensor(self.v_threshold, device=x.device, dtype=torch.float32)
        spike = rectangle_spike(total_mem - th)
        reset_spike = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = m_t - reset_spike * th
        else:
            self.v = torch.where(reset_spike.bool(), torch.as_tensor(self.v_reset, device=x.device), m_t)
        self.n = n_t
        self.has_fired = torch.logical_or(self.has_fired, reset_spike.bool())
        return spike.to(dtype=x.dtype)


class EEGSegmentDataset(Dataset):
    def __init__(self, path: str):
        data = np.load(path, allow_pickle=True).item()
        self.fea = data["fea"]
        self.label = data["label"]

    def __len__(self):
        return len(self.fea)

    def __getitem__(self, index: int):
        fea = self.fea[index].astype(np.float32)  # [61, 750]
        label = int(self.label[index])
        fea = np.expand_dims(fea, axis=0)  # [1, 61, 750]，保持 train_segment.py 数据形状习惯
        return torch.from_numpy(fea), torch.tensor(label, dtype=torch.long)


class SingleLayerFCSNN(nn.Module):
    def __init__(self, neuron_type: str, input_dim: int = 61, num_classes: int = 2, **neuron_kwargs):
        super().__init__()
        self.neuron_type = neuron_type.upper()
        self.fc = nn.Linear(input_dim, num_classes)
        if self.neuron_type == "LIF":
            self.neuron = LIFNeuron(**{k: v for k, v in neuron_kwargs.items() if not k.startswith("history_")})
        elif self.neuron_type == "LSLIF":
            self.neuron = LSLIFNeuron(**neuron_kwargs)
        else:
            raise ValueError(f"Unsupported neuron_type: {neuron_type}")

    def reset(self):
        self.neuron.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 61, 750] -> [750, B, 61]
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"Expected input shape [B,1,61,750], got {tuple(x.shape)}")
        x_tb = x.squeeze(1).permute(2, 0, 1).contiguous()
        spike_sum = None
        for t in range(x_tb.size(0)):
            current = self.fc(x_tb[t])
            spike_t = self.neuron(current)
            spike_sum = spike_t if spike_sum is None else spike_sum + spike_t
        return spike_sum / x_tb.size(0)


def reset_net(model: nn.Module):
    for module in model.modules():
        if module is not model and hasattr(module, "reset"):
            module.reset()


def sorted_fold_files(data_dir: str, prefix: str) -> List[str]:
    files = [f for f in os.listdir(data_dir) if f.startswith(prefix) and f.endswith(".npy")]
    files.sort(key=lambda name: int(name[len(prefix):-4]))
    return files


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Metrics:
    loss: float
    acc: float
    recall: float
    specificity: float
    precision: float
    f1: float
    auc: float


def compute_metrics(losses: List[float], labels: List[int], preds: List[int], probs_pos: List[float]) -> Metrics:
    labels_np = np.asarray(labels)
    preds_np = np.asarray(preds)
    cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auc = roc_auc_score(labels_np, np.asarray(probs_pos)) if len(np.unique(labels_np)) == 2 else 0.0
    except ValueError:
        auc = 0.0
    return Metrics(
        loss=float(np.mean(losses)) if losses else 0.0,
        acc=float((preds_np == labels_np).mean()) if len(labels_np) else 0.0,
        recall=float(recall_score(labels_np, preds_np, zero_division=0)),
        specificity=float(specificity),
        precision=float(precision_score(labels_np, preds_np, zero_division=0)),
        f1=float(f1_score(labels_np, preds_np, zero_division=0)),
        auc=float(auc),
    )


def run_one_epoch(model, dataloader, criterion, optimizer, device, train: bool) -> Metrics:
    model.train(train)
    losses, labels_all, preds_all, probs_all = [], [], [], []
    for inputs, labels in dataloader:
        reset_net(model)
        inputs = inputs.to(device)
        labels = labels.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
        probs = F.softmax(outputs.detach(), dim=1)
        losses.append(loss.item())
        labels_all.extend(labels.detach().cpu().tolist())
        preds_all.extend(outputs.detach().argmax(1).cpu().tolist())
        probs_all.extend(probs[:, 1].cpu().tolist())
    return compute_metrics(losses, labels_all, preds_all, probs_all)


def train_one_fold(args, fold_idx: int, neuron_type: str, train_path: str, test_path: str, device: torch.device) -> Dict[str, float]:
    train_dataset = EEGSegmentDataset(train_path)
    test_dataset = EEGSegmentDataset(test_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = SingleLayerFCSNN(
        neuron_type=neuron_type,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        tau=args.tau,
        decay_input=args.decay_input,
        v_threshold=args.v_threshold,
        v_reset=None,
        detach_reset=args.detach_reset,
        history_weight=args.history_weight,
        history_power=args.history_power,
        history_eps=args.history_eps,
        history_mode=args.history_mode,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best = None
    best_state = None
    rows = []
    for epoch in range(args.epochs):
        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        test_metrics = run_one_epoch(model, test_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        row = {
            "fold": fold_idx,
            "neuron": neuron_type,
            "epoch": epoch + 1,
            **{f"train_{k}": v for k, v in asdict(train_metrics).items()},
            **{f"test_{k}": v for k, v in asdict(test_metrics).items()},
        }
        rows.append(row)
        if best is None or test_metrics.acc > best["test_acc"]:
            best = row.copy()
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(
            f"[{neuron_type}][fold {fold_idx}][{epoch + 1:03d}/{args.epochs}] "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.acc:.4f} "
            f"test_loss={test_metrics.loss:.4f} test_acc={test_metrics.acc:.4f} "
            f"test_f1={test_metrics.f1:.4f} test_auc={test_metrics.auc:.4f}"
        )

    fold_dir = os.path.join(args.output_dir, neuron_type, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    with open(os.path.join(fold_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if args.save_model and best_state is not None:
        torch.save(best_state, os.path.join(fold_dir, "best_model.pth"))
    with open(os.path.join(fold_dir, "best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    return best


def parse_args():
    parser = argparse.ArgumentParser(description="Single-file EEG segment LIF vs LSLIF single-layer FC SNN experiment")
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--neurons", nargs="+", default=["LIF", "LSLIF"], choices=["LIF", "LSLIF"])
    parser.add_argument("--batch-size", type=int, default=124)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--step-size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--hidden-dim", type=int, default=128, help="Deprecated; ignored because the model is now single-layer FC.")
    parser.add_argument("--input-dim", type=int, default=61)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--v-threshold", type=float, default=1.0)
    parser.add_argument("--decay-input", action="store_true")
    parser.add_argument("--detach-reset", action="store_true")
    parser.add_argument("--history-weight", type=float, default=1.0)
    parser.add_argument("--history-power", type=float, default=1.0)
    parser.add_argument("--history-eps", type=float, default=1e-6)
    parser.add_argument("--history-mode", default="all", choices=["all", "post_spike"])
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-last", action="store_true", default=True)
    parser.add_argument("--no-drop-last", dest="drop_last", action="store_false")
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    print("torch version:", torch.__version__)
    print("device:", device)
    print("start time:", time.strftime("%Y-%m-%d %H:%M:%S"))

    train_files = sorted_fold_files(args.train_dir, "train")
    test_files = sorted_fold_files(args.test_dir, "test")
    num_folds = min(args.folds, len(train_files), len(test_files))
    if num_folds <= 0:
        raise RuntimeError(f"No folds found in train_dir={args.train_dir} and test_dir={args.test_dir}")

    summary_rows = []
    for fold_idx in range(num_folds):
        train_path = os.path.join(args.train_dir, train_files[fold_idx])
        test_path = os.path.join(args.test_dir, test_files[fold_idx])
        print(f"\nfold {fold_idx}: train={train_path} test={test_path}")
        for neuron_type in args.neurons:
            best = train_one_fold(args, fold_idx, neuron_type, train_path, test_path, device)
            summary_rows.append(best)

    summary_path = os.path.join(args.output_dir, "summary_best.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    for neuron_type in args.neurons:
        accs = [row["test_acc"] for row in summary_rows if row["neuron"] == neuron_type]
        f1s = [row["test_f1"] for row in summary_rows if row["neuron"] == neuron_type]
        aucs = [row["test_auc"] for row in summary_rows if row["neuron"] == neuron_type]
        print(
            f"{neuron_type} best summary: "
            f"acc={np.mean(accs):.4f}±{np.std(accs):.4f}, "
            f"f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}, "
            f"auc={np.mean(aucs):.4f}±{np.std(aucs):.4f}"
        )
    print("summary saved to:", summary_path)


if __name__ == "__main__":
    main()
