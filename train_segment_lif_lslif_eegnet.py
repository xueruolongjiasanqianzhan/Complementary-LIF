# -*- coding: utf-8 -*-
"""被试内 EEG segment 数据上的 chunk-based Spiking EEGNet：LIF vs LSLIF。

输入 ``[B, 1, 61, 750]`` 沿采样轴切成 30 个互不重叠的 25 点 chunk。每个
chunk 直接作为一个 SNN 时间步输入共享权重的 EEGNet，不做脉冲编码，也不在
chunk 之间维护卷积缓存。LIF/LSLIF 状态在同一样本的 chunk 之间保留；所有时间
步的脉冲特征先求平均，最后由普通 ``nn.Linear`` 分类。

训练顺序以 fold 为外层循环、神经元类型为内层循环，因此默认执行顺序为：
fold0-LIF、fold0-LSLIF、fold1-LIF、fold1-LSLIF……，便于逐折公平比较。
"""

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset


DEFAULT_TRAIN_DIR = "/home/guyue/zhao/PythonProject/dataset/Self-data/kFold/kFold(61x750)/new-segment/train/"
DEFAULT_TEST_DIR = "/home/guyue/zhao/PythonProject/dataset/Self-data/kFold/kFold(61x750)/new-segment/test/"
DEFAULT_OUTPUT_DIR = "all_result/spiking_eegnet/segment_lif_vs_lslif/"


class RectangleSpike(torch.autograd.Function):
    """二值发放，反向传播使用宽度为 alpha 的矩形替代梯度。"""

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
            mask = x.abs() <= (ctx.alpha / 2.0)
            grad_x = grad_output * mask.to(grad_output.dtype) / ctx.alpha
        return grad_x, None


def rectangle_spike(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return RectangleSpike.apply(x, alpha)


class StatefulNeuron(nn.Module):
    def reset(self):
        raise NotImplementedError


class LIFNeuron(StatefulNeuron):
    def __init__(self, tau=2.0, decay_input=False, v_threshold=1.0,
                 v_reset=None, detach_reset=False, tau_eps=1e-6):
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

    def _ensure_state(self, x):
        if self.v is None or self.v.shape != x.shape or self.v.device != x.device:
            self.v = torch.zeros_like(x, dtype=torch.float32)

    def forward(self, x):
        self._ensure_state(x)
        x_f = x.float()
        tau = torch.as_tensor(self.tau, device=x.device, dtype=torch.float32)
        if self.decay_input:
            self.v = self.v + (x_f - self.v) / (tau + self.tau_eps)
        else:
            decay = torch.clamp(1.0 - 1.0 / (tau + self.tau_eps), 0.0, 1.0)
            self.v = self.v * decay + x_f
        threshold = torch.as_tensor(self.v_threshold, device=x.device, dtype=torch.float32)
        spike = rectangle_spike(self.v - threshold)
        reset_spike = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - reset_spike * threshold
        else:
            reset_value = torch.as_tensor(self.v_reset, device=x.device, dtype=torch.float32)
            self.v = torch.where(reset_spike.bool(), reset_value, self.v)
        return spike.to(x.dtype)


class LSLIFNeuron(StatefulNeuron):
    """带不重置辅助膜的 LSLIF；辅助历史按 SNN 时间步幂次归一化。"""

    def __init__(self, tau=2.0, decay_input=False, v_threshold=1.0,
                 v_reset=None, detach_reset=False, tau_eps=1e-6,
                 history_weight=1.0, history_power=1.0, history_eps=1e-6,
                 history_mode="all"):
        super().__init__()
        if history_mode not in {"all", "post_spike"}:
            raise ValueError("history_mode must be 'all' or 'post_spike'")
        self.tau = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.detach_reset = bool(detach_reset)
        self.tau_eps = float(tau_eps)
        self.history_weight = float(history_weight)
        self.history_power = float(history_power)
        self.history_eps = float(history_eps)
        self.history_mode = history_mode
        self.v = self.n = self.has_fired = None
        self.step_count = 0

    def reset(self):
        self.v = self.n = self.has_fired = None
        self.step_count = 0

    def _ensure_state(self, x):
        if self.v is None or self.v.shape != x.shape or self.v.device != x.device:
            self.v = torch.zeros_like(x, dtype=torch.float32)
            self.n = torch.zeros_like(x, dtype=torch.float32)
            self.has_fired = torch.zeros_like(x, dtype=torch.bool)
            self.step_count = 0

    def forward(self, x):
        self._ensure_state(x)
        x_f = x.float()
        tau = torch.as_tensor(self.tau, device=x.device, dtype=torch.float32)
        if self.decay_input:
            m_t = self.v + (x_f - self.v) / (tau + self.tau_eps)
            n_t = self.n + (x_f - self.n) / (tau + self.tau_eps)
        else:
            decay = torch.clamp(1.0 - 1.0 / (tau + self.tau_eps), 0.0, 1.0)
            m_t = self.v * decay + x_f
            n_t = self.n * decay + x_f
        self.step_count += 1
        norm = (float(self.step_count) + self.history_eps) ** self.history_power
        history = self.history_weight * n_t / norm
        if self.history_mode == "post_spike":
            history = history * self.has_fired.to(history.dtype)
        threshold = torch.as_tensor(self.v_threshold, device=x.device, dtype=torch.float32)
        spike = rectangle_spike(m_t + history - threshold)
        reset_spike = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = m_t - reset_spike * threshold
        else:
            reset_value = torch.as_tensor(self.v_reset, device=x.device, dtype=torch.float32)
            self.v = torch.where(reset_spike.bool(), reset_value, m_t)
        self.n = n_t
        self.has_fired = torch.logical_or(self.has_fired, reset_spike.bool())
        return spike.to(x.dtype)


def build_neuron(neuron_type: str, **kwargs):
    if neuron_type == "LIF":
        kwargs = {key: value for key, value in kwargs.items() if not key.startswith("history_")}
        return LIFNeuron(**kwargs)
    if neuron_type == "LSLIF":
        return LSLIFNeuron(**kwargs)
    raise ValueError(f"Unsupported neuron_type: {neuron_type}")


class EEGSegmentDataset(Dataset):
    def __init__(self, path: str):
        data = np.load(path, allow_pickle=True).item()
        self.fea = data["fea"]
        self.label = data["label"]

    def __len__(self):
        return len(self.fea)

    def __getitem__(self, index):
        fea = self.fea[index].astype(np.float32)
        return torch.from_numpy(fea[None]), torch.tensor(int(self.label[index]), dtype=torch.long)


class SpikingEEGNet(nn.Module):
    """Chunk-based EEGNet-like SNN，支持 tiny 和 full 两种容量。"""

    def __init__(self, neuron_type, channels=61, num_classes=2, chunk_size=25,
                 architecture="tiny", f1=4, depth_multiplier=1, f2=16,
                 temporal_kernel=15, separable_kernel=7, dropout=0.25,
                 **neuron_kwargs):
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if temporal_kernel <= 0 or temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be a positive odd number")
        if separable_kernel <= 0 or separable_kernel % 2 == 0:
            raise ValueError("separable_kernel must be a positive odd number")
        if architecture not in {"tiny", "full"}:
            raise ValueError("architecture must be 'tiny' or 'full'")
        self.architecture = architecture
        self.chunk_size = int(chunk_size)
        self.channels = int(channels)
        spatial_filters = int(f1) * int(depth_multiplier)

        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(f1, spatial_filters, (channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(spatial_filters),
        )
        self.neuron1 = build_neuron(neuron_type, **neuron_kwargs)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        if architecture == "full":
            self.separable = nn.Sequential(
                nn.Conv2d(spatial_filters, spatial_filters, (1, separable_kernel),
                          padding=(0, separable_kernel // 2), groups=spatial_filters, bias=False),
                nn.Conv2d(spatial_filters, f2, 1, bias=False),
                nn.BatchNorm2d(f2),
            )
            self.neuron2 = build_neuron(neuron_type, **neuron_kwargs)
            classifier_features = f2
        else:
            self.separable = None
            self.neuron2 = None
            classifier_features = spatial_filters
        self.output_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout2 = nn.Dropout(dropout)
        self.classifier = nn.Linear(classifier_features, num_classes)

    def _step(self, chunk):
        x = self.temporal(chunk)
        x = self.spatial(x)
        x = self.dropout1(self.pool1(self.neuron1(x)))
        if self.separable is not None:
            x = self.neuron2(self.separable(x))
        x = self.dropout2(self.output_pool(x))
        return torch.flatten(x, 1)

    def forward(self, x):
        if x.dim() != 4 or x.size(1) != 1 or x.size(2) != self.channels:
            raise ValueError(f"Expected [B,1,{self.channels},L], got {tuple(x.shape)}")
        length = x.size(-1)
        if length <= 0:
            raise ValueError("EEG sample length must be positive")
        steps = math.ceil(length / self.chunk_size)
        feature_sum = None
        weight_sum = 0.0
        for step in range(steps):
            start = step * self.chunk_size
            end = min(start + self.chunk_size, length)
            chunk = x[..., start:end]
            valid_weight = (end - start) / self.chunk_size
            if chunk.size(-1) < self.chunk_size:
                chunk = F.pad(chunk, (0, self.chunk_size - chunk.size(-1)))
            feature = self._step(chunk)
            feature_sum = feature * valid_weight if feature_sum is None else feature_sum + feature * valid_weight
            weight_sum += valid_weight
        mean_feature = feature_sum / weight_sum
        return self.classifier(mean_feature)


def reset_net(model):
    for module in model.modules():
        if module is not model and hasattr(module, "reset"):
            module.reset()


def sorted_fold_files(data_dir: str, prefix: str) -> List[str]:
    files = [name for name in os.listdir(data_dir) if name.startswith(prefix) and name.endswith(".npy")]
    files.sort(key=lambda name: int(name[len(prefix):-4]))
    return files


def set_seed(seed):
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


def compute_metrics(losses, labels, preds, probs_pos):
    labels_np, preds_np = np.asarray(labels), np.asarray(preds)
    tn, fp, fn, tp = confusion_matrix(labels_np, preds_np, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
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


def run_one_epoch(model, dataloader, criterion, optimizer, device, train):
    model.train(train)
    losses, labels_all, preds_all, probs_all = [], [], [], []
    for inputs, labels in dataloader:
        reset_net(model)
        inputs, labels = inputs.to(device), labels.to(device)
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


def train_one_fold(args, fold_idx, neuron_type, train_path, test_path, device) -> Dict[str, float]:
    train_loader = DataLoader(EEGSegmentDataset(train_path), batch_size=args.batch_size,
                              shuffle=True, drop_last=args.drop_last)
    test_loader = DataLoader(EEGSegmentDataset(test_path), batch_size=args.batch_size,
                             shuffle=False, drop_last=False)
    model = SpikingEEGNet(
        neuron_type=neuron_type, channels=args.channels, num_classes=args.num_classes,
        architecture=args.architecture,
        chunk_size=args.chunk_size, f1=args.f1, depth_multiplier=args.depth_multiplier,
        f2=args.f2, temporal_kernel=args.temporal_kernel,
        separable_kernel=args.separable_kernel, dropout=args.dropout, tau=args.tau,
        decay_input=args.decay_input, v_threshold=args.v_threshold, v_reset=None,
        detach_reset=args.detach_reset, history_weight=args.history_weight,
        history_power=args.history_power, history_eps=args.history_eps,
        history_mode=args.history_mode,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    best = best_state = None
    rows = []
    for epoch in range(args.epochs):
        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device, True)
        test_metrics = run_one_epoch(model, test_loader, criterion, optimizer, device, False)
        scheduler.step()
        row = {"fold": fold_idx, "neuron": neuron_type, "epoch": epoch + 1,
               **{f"train_{key}": value for key, value in asdict(train_metrics).items()},
               **{f"test_{key}": value for key, value in asdict(test_metrics).items()}}
        rows.append(row)
        if best is None or test_metrics.acc > best["test_acc"]:
            best = row.copy()
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(f"[{neuron_type}][fold {fold_idx}][{epoch + 1:03d}/{args.epochs}] "
              f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.acc:.4f} "
              f"test_loss={test_metrics.loss:.4f} test_acc={test_metrics.acc:.4f} "
              f"test_f1={test_metrics.f1:.4f} test_auc={test_metrics.auc:.4f}")
    fold_dir = os.path.join(args.output_dir, neuron_type, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    with open(os.path.join(fold_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if args.save_model and best_state is not None:
        torch.save(best_state, os.path.join(fold_dir, "best_model.pth"))
    with open(os.path.join(fold_dir, "best.json"), "w", encoding="utf-8") as file:
        json.dump(best, file, ensure_ascii=False, indent=2)
    return best


def parse_args():
    parser = argparse.ArgumentParser(description="Chunk-based Spiking EEGNet LIF vs LSLIF experiment")
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
    parser.add_argument("--channels", type=int, default=61)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--architecture", default="tiny", choices=["tiny", "full"],
                        help="tiny removes the second separable-convolution/spiking block")
    parser.add_argument("--f1", type=int, default=4)
    parser.add_argument("--depth-multiplier", type=int, default=1)
    parser.add_argument("--f2", type=int, default=16)
    parser.add_argument("--temporal-kernel", type=int, default=15)
    parser.add_argument("--separable-kernel", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.25)
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
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)
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
        # Neuron loop intentionally stays inside the fold loop: LIF/LSLIF alternate by fold.
        for neuron_type in args.neurons:
            summary_rows.append(train_one_fold(args, fold_idx, neuron_type,
                                               train_path, test_path, device))
    with open(os.path.join(args.output_dir, "summary_best.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    for neuron_type in args.neurons:
        selected = [row for row in summary_rows if row["neuron"] == neuron_type]
        for metric in ("test_acc", "test_f1", "test_auc"):
            values = [row[metric] for row in selected]
            print(f"{neuron_type} {metric}={np.mean(values):.4f}±{np.std(values):.4f}")


if __name__ == "__main__":
    main()
