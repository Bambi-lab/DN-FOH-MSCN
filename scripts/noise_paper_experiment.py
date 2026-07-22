"""Noise robustness experiments for the FOH-MSCN paper revision.

This script is designed for the remote server. It trains one model per
invocation and can aggregate all completed runs into paper-ready CSV tables.

Examples:
  python noise_paper_experiment.py train --model foh --seed 42
  python noise_paper_experiment.py train --model dn_foh --seed 42 --init-checkpoint auto
  python noise_paper_experiment.py aggregate
"""

import argparse
import csv
import datetime as _dt
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "optical_classification" / "noise_paper"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "results" / "optical_classification" / "cache_enhanced"
DEFAULT_OPTICAL_DIR = Path(os.environ.get("OPTICAL_DIR", str(PROJECT_ROOT / "data" / "foh_enhanced")))

sys.path.insert(0, str(SCRIPT_PATH.parent))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "optical_classification"))

import dataset_optical as ds_mod  # noqa: E402
from dataset_optical import (  # noqa: E402
    CATEGORIES,
    IndexDataset,
    compute_class_weights,
    compute_sample_weights,
    get_split_indices,
    load_phase_gammatone_data,
    load_raw_data_v2,
    scan_dataset,
)
from gammatone_features import extract_gammatone_spectrum  # noqa: E402


SNR_ORDER = ["clean", "30", "20", "10", "5", "0"]
LOW_SNR_KEYS = ["10", "5", "0"]
MODEL_LABELS = {
    "foh": "FOH-MSCN",
    "av2": "Exp-A-v2",
    "expb": "Exp-B",
    "aug_foh": "FOH-MSCN+Aug",
    "denoise_foh": "FOH-MSCN+Denoiser",
    "dn_foh": "DN-FOH-MSCN",
}


def format_time(seconds):
    return str(_dt.timedelta(seconds=int(round(seconds))))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_snr_token(token):
    token = str(token).strip().lower()
    if token in {"clean", "none", "inf", "infinite"}:
        return None
    return float(token)


def snr_key(snr):
    if snr is None:
        return "clean"
    value = float(snr)
    if value.is_integer():
        return str(int(value))
    return str(value)


def add_awgn_numpy(signal, snr_db, rng):
    if snr_db is None:
        return signal
    sig_power = np.mean(signal ** 2, axis=-1, keepdims=True)
    sig_power = np.maximum(sig_power, 1e-12)
    noise_power = sig_power / (10.0 ** (float(snr_db) / 10.0))
    noise = rng.standard_normal(signal.shape).astype(np.float32) * np.sqrt(noise_power)
    return signal + noise


def add_awgn_torch(x, snr_levels):
    if not snr_levels:
        return x
    out = x.clone()
    choices = torch.randint(0, len(snr_levels), (x.size(0),), device=x.device)
    for choice_idx, snr in enumerate(snr_levels):
        mask = choices == choice_idx
        if not torch.any(mask) or snr is None:
            continue
        selected = x[mask]
        sig_power = selected.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12)
        noise_power = sig_power / (10.0 ** (float(snr) / 10.0))
        out[mask] = selected + torch.randn_like(selected) * noise_power.sqrt()
    return out


class MSConvBlock(nn.Module):
    def __init__(self, in_channels, out_per_branch=32, out_proj=128):
        super().__init__()
        self.branch_s = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(out_per_branch),
            nn.GELU(),
        )
        self.branch_m = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(out_per_branch),
            nn.GELU(),
        )
        self.branch_l = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=31, stride=2, padding=15, bias=False),
            nn.BatchNorm1d(out_per_branch),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Conv1d(out_per_branch * 3, out_proj, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_proj),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(torch.cat([self.branch_s(x), self.branch_m(x), self.branch_l(x)], dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).unsqueeze(-1)


class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        w = torch.softmax(self.score(x), dim=1)
        return (x * w).sum(dim=1)


class FOHMSCN(nn.Module):
    def __init__(
        self,
        num_classes=5,
        in_channels=3,
        msconv_per_branch=32,
        msconv_proj=128,
        gru_hidden=64,
        gru_layers=2,
        dropout=0.4,
    ):
        super().__init__()
        self.input_norm = nn.InstanceNorm1d(in_channels)
        self.msconv = MSConvBlock(in_channels, msconv_per_branch, msconv_proj)
        self.stage1 = nn.Sequential(
            nn.MaxPool1d(4),
            nn.Conv1d(msconv_proj, msconv_proj, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(msconv_proj),
            nn.GELU(),
            nn.MaxPool1d(4),
        )
        self.stage2 = nn.Sequential(
            nn.Conv1d(msconv_proj, msconv_proj, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(msconv_proj),
            nn.GELU(),
            nn.MaxPool1d(4),
        )
        self.chan_attn = ChannelAttention(msconv_proj, reduction=8)
        self.bigru = nn.GRU(
            input_size=msconv_proj,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.temp_pool = TemporalAttentionPool(gru_hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.msconv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.chan_attn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bigru(x)
        x = self.temp_pool(x)
        return self.classifier(x)


class OpticalCNNBiGRU_v2(nn.Module):
    def __init__(self, num_classes=5, in_channels=3):
        super().__init__()
        self.input_norm = nn.InstanceNorm1d(in_channels)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.gru = nn.GRU(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        return self.classifier(out[:, -1, :])


class CNNModel(nn.Module):
    def __init__(self, input_shape=(128, 139), num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        with torch.no_grad():
            x = torch.zeros(1, 1, *input_shape)
            x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
            flatten_size = x.view(1, -1).size(1)
        self.fc1 = nn.Linear(flatten_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class LightweightResidualDenoiser(nn.Module):
    def __init__(self, channels=3, hidden=24, gate_hidden=12):
        super().__init__()
        self.noise_net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=15, padding=7, groups=channels, bias=False),
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=31, padding=15, groups=hidden, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
        )
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )
        self._init_identity()

    def _init_identity(self):
        last = self.noise_net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        gate_linear = self.gate[-2]
        nn.init.zeros_(gate_linear.weight)
        nn.init.constant_(gate_linear.bias, -2.0)

    def forward(self, x):
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        gate = self.gate(torch.cat([mean, std], dim=1)).view(-1, 1, 1)
        estimated_noise = self.noise_net(x)
        return x - gate * estimated_noise


class DenoisedFOHMSCN(nn.Module):
    def __init__(self, denoise_hidden=24):
        super().__init__()
        self.denoiser = LightweightResidualDenoiser(channels=3, hidden=denoise_hidden)
        self.backbone = FOHMSCN()

    def forward(self, x, return_denoised=False):
        x_denoised = self.denoiser(x)
        logits = self.backbone(x_denoised)
        if return_denoised:
            return logits, x_denoised
        return logits


class DeterministicNoisyDiffDataset(Dataset):
    def __init__(self, X, y, indices, snr_db=None, noise_seed=42):
        self.X = X
        self.y = y
        self.indices = np.asarray(indices)
        self.snr_db = snr_db
        self.noise_seed = int(noise_seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        x = np.array(self.X[i], dtype=np.float32)
        if self.snr_db is not None:
            snr_offset = int(round(float(self.snr_db) * 100))
            rng = np.random.default_rng(self.noise_seed + i * 1009 + snr_offset)
            x = add_awgn_numpy(x, self.snr_db, rng)
        return torch.from_numpy(x), int(self.y[i])


class DeterministicGammatoneNoisyDataset(Dataset):
    def __init__(self, file_paths, labels, indices, snr_db=None, noise_seed=42):
        self.file_paths = file_paths
        self.labels = labels
        self.indices = np.asarray(indices)
        self.snr_db = snr_db
        self.noise_seed = int(noise_seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        mat = sio.loadmat(self.file_paths[i])["Vo3x3"]
        sig = np.stack([mat[:, 0], mat[:, 1], mat[:, 2]], axis=0).astype(np.float32)
        if self.snr_db is not None:
            snr_offset = int(round(float(self.snr_db) * 100))
            rng = np.random.default_rng(self.noise_seed + i * 1009 + snr_offset)
            sig = add_awgn_numpy(sig, self.snr_db, rng)
        v1, v2, v3 = sig[0], sig[1], sig[2]
        phi = np.arctan2(np.sqrt(3) * (v3 - v2), 2 * v1 - v2 - v3)
        spec, _ = extract_gammatone_spectrum(phi.astype(np.float64), sr=12000)
        return torch.from_numpy(spec.astype(np.float32)).unsqueeze(0), int(self.labels[i])


def configure_dataset_paths(optical_dir, cache_dir):
    ds_mod.OPTICAL_DIR = str(optical_dir)
    ds_mod.CACHE_DIR = str(cache_dir)
    os.environ["OPTICAL_DIR"] = str(optical_dir)
    os.environ["CACHE_DIR"] = str(cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)


def make_model(model_key, input_shape=None, denoise_hidden=24):
    if model_key in {"foh", "aug_foh"}:
        return FOHMSCN()
    if model_key == "av2":
        return OpticalCNNBiGRU_v2()
    if model_key == "expb":
        return CNNModel(input_shape=input_shape or (128, 139))
    if model_key in {"denoise_foh", "dn_foh"}:
        return DenoisedFOHMSCN(denoise_hidden=denoise_hidden)
    raise ValueError(f"Unsupported model: {model_key}")


def load_checkpoint_for_variant(model, model_key, checkpoint_path, device):
    if not checkpoint_path:
        return None
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"[WARN] Checkpoint not found: {checkpoint_path}", flush=True)
        return None
    state = torch.load(checkpoint_path, map_location=device)
    if model_key in {"dn_foh", "denoise_foh"}:
        model.backbone.load_state_dict(state)
        loaded_as = "backbone"
    else:
        model.load_state_dict(state)
        loaded_as = "full"
    print(f"Loaded {loaded_as} checkpoint: {checkpoint_path}", flush=True)
    return str(checkpoint_path)


def resolve_auto_checkpoint(args):
    if args.init_checkpoint != "auto":
        return args.init_checkpoint
    if args.model not in {"dn_foh", "denoise_foh", "aug_foh"}:
        return None
    candidate = Path(args.results_root) / f"foh_seed{args.seed}" / "model_best.pth"
    return str(candidate)


def make_train_loader(X, y, train_idx, batch_size, num_workers, device, unsqueeze_channel=False):
    sample_weights = compute_sample_weights(y, train_idx)
    sampler = WeightedRandomSampler(sample_weights, len(train_idx), replacement=True)
    train_ds = IndexDataset(X, y, train_idx, unsqueeze_channel=unsqueeze_channel)
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def make_eval_loader(
    model_key,
    X,
    y,
    file_paths,
    labels,
    indices,
    snr,
    batch_size,
    num_workers,
    device,
    noise_seed,
):
    if model_key == "expb":
        if snr is None and X is not None:
            ds = IndexDataset(X, y, indices, unsqueeze_channel=True)
        else:
            ds = DeterministicGammatoneNoisyDataset(file_paths, labels, indices, snr_db=snr, noise_seed=noise_seed)
    else:
        ds = DeterministicNoisyDiffDataset(X, y, indices, snr_db=snr, noise_seed=noise_seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def predict_all(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(1).cpu().numpy()
            preds.append(pred)
            labels.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def evaluate_once(
    model,
    model_key,
    X,
    y,
    file_paths,
    labels,
    test_idx,
    snr,
    batch_size,
    num_workers,
    device,
    noise_seed,
):
    loader = make_eval_loader(
        model_key,
        X,
        y,
        file_paths,
        labels,
        test_idx,
        snr,
        batch_size,
        num_workers,
        device,
        noise_seed,
    )
    y_pred, y_true = predict_all(model, loader, device)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CATEGORIES))),
        target_names=CATEGORIES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report": report,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def score_model(model, data_ctx, select_snr_levels, args, device):
    scores = []
    for snr in select_snr_levels:
        result = evaluate_once(
            model,
            args.model,
            data_ctx["X"],
            data_ctx["y"],
            data_ctx["file_paths"],
            data_ctx["labels"],
            data_ctx["test_idx"],
            snr,
            args.eval_batch_size,
            args.eval_num_workers,
            device,
            args.noise_seed,
        )
        scores.append(result["accuracy"])
    return float(np.mean(scores)), scores


def set_backbone_trainable(model, trainable):
    if not hasattr(model, "backbone"):
        return
    for param in model.backbone.parameters():
        param.requires_grad = trainable


def train_epoch(model, model_key, loader, criterion, recon_criterion, optimizer, device, train_snr_levels, lambda_recon):
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_rec = 0.0
    correct = 0
    count = 0
    for xb_clean, yb in loader:
        xb_clean = xb_clean.to(device)
        yb = yb.to(device)
        xb_input = add_awgn_torch(xb_clean, train_snr_levels) if train_snr_levels else xb_clean
        optimizer.zero_grad()
        if model_key in {"dn_foh", "denoise_foh"}:
            logits, xb_denoised = model(xb_input, return_denoised=True)
            loss_cls = criterion(logits, yb)
            loss_rec = recon_criterion(xb_denoised, xb_clean)
            loss = loss_cls + lambda_recon * loss_rec
        else:
            logits = model(xb_input)
            loss_cls = criterion(logits, yb)
            loss_rec = torch.tensor(0.0, device=device)
            loss = loss_cls
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
        total_cls += float(loss_cls.item())
        total_rec += float(loss_rec.item())
        correct += int((logits.argmax(1) == yb).sum().item())
        count += int(yb.size(0))
    n_batches = max(len(loader), 1)
    return {
        "loss": total_loss / n_batches,
        "loss_cls": total_cls / n_batches,
        "loss_recon": total_rec / n_batches,
        "train_acc": correct / max(count, 1),
    }


def load_data_for_training(model_key, args):
    configure_dataset_paths(args.optical_dir, args.cache_dir)
    file_paths, labels = scan_dataset(str(args.optical_dir))
    if model_key == "expb":
        X, y = load_phase_gammatone_data(file_paths, labels)
    else:
        X, y = load_raw_data_v2(file_paths, labels)
    train_idx, test_idx = get_split_indices(y)
    return {
        "file_paths": file_paths,
        "labels": labels,
        "X": X,
        "y": y,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


def default_train_snr_levels(model_key):
    if model_key in {"dn_foh", "aug_foh"}:
        return [None, 30.0, 20.0, 10.0, 5.0, 0.0]
    return []


def default_select_snr_levels(model_key):
    if model_key in {"dn_foh", "aug_foh"}:
        return [10.0, 5.0, 0.0]
    return [None]


def run_train(args):
    set_seed(args.seed)
    args.results_root = Path(args.results_root)
    args.optical_dir = Path(args.optical_dir)
    args.cache_dir = Path(args.cache_dir)
    run_name = args.run_name or f"{args.model}_seed{args.seed}"
    output_dir = args.results_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model} ({MODEL_LABELS[args.model]})", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Output: {output_dir}", flush=True)

    data_ctx = load_data_for_training(args.model, args)
    train_idx = data_ctx["train_idx"]
    test_idx = data_ctx["test_idx"]
    X = data_ctx["X"]
    y = data_ctx["y"]
    print(f"Data: X={X.shape}, train={len(train_idx)}, test={len(test_idx)}", flush=True)

    if args.sanity_run:
        train_idx = train_idx[: min(len(train_idx), 240)]
        test_idx = test_idx[: min(len(test_idx), 80)]
        data_ctx["train_idx"] = train_idx
        data_ctx["test_idx"] = test_idx
        args.epochs = min(args.epochs, 2)
        args.patience = 99
        args.eval_every = 1
        print("[SANITY] using reduced split and <=2 epochs", flush=True)

    train_snr_levels = [parse_snr_token(x) for x in args.train_snr_levels] if args.train_snr_levels else default_train_snr_levels(args.model)
    select_snr_levels = [parse_snr_token(x) for x in args.select_snr_levels] if args.select_snr_levels else default_select_snr_levels(args.model)
    eval_snr_levels = [parse_snr_token(x) for x in args.eval_snr_levels]
    print(f"Train SNR: {[snr_key(x) for x in train_snr_levels] or ['clean-only']}", flush=True)
    print(f"Select SNR: {[snr_key(x) for x in select_snr_levels]}", flush=True)
    print(f"Eval SNR: {[snr_key(x) for x in eval_snr_levels]}", flush=True)

    input_shape = None
    if args.model == "expb":
        input_shape = (X.shape[1], X.shape[2])
    model = make_model(args.model, input_shape=input_shape, denoise_hidden=args.denoise_hidden).to(device)
    loaded_checkpoint = load_checkpoint_for_variant(model, args.model, resolve_auto_checkpoint(args), device)

    unsqueeze_channel = args.model == "expb"
    train_loader = make_train_loader(
        X,
        y,
        train_idx,
        args.batch_size,
        args.num_workers,
        device,
        unsqueeze_channel=unsqueeze_channel,
    )
    class_weights = compute_class_weights(y).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    recon_criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.min_lr,
    )

    total_params = count_params(model)
    denoiser_params = count_params(model.denoiser) if hasattr(model, "denoiser") else 0
    print(f"Params: total={total_params:,}, denoiser={denoiser_params:,}", flush=True)

    best_score = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        set_backbone_trainable(model, epoch >= args.freeze_backbone_epochs)
        stats = train_epoch(
            model,
            args.model,
            train_loader,
            criterion,
            recon_criterion,
            optimizer,
            device,
            train_snr_levels,
            args.lambda_recon,
        )
        scheduler.step()
        do_eval = ((epoch + 1) % args.eval_every == 0) or epoch == 0 or epoch == args.epochs - 1
        if do_eval:
            score, select_accs = score_model(model, data_ctx, select_snr_levels, args, device)
        else:
            score, select_accs = best_score, []

        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "selection_score": score,
            "selection_accs": select_accs,
            **stats,
        }
        history.append(row)

        if do_eval and score > best_score:
            best_score = score
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "model_best.pth")
        elif do_eval:
            patience_counter += 1

        if do_eval or (epoch + 1) % 5 == 0:
            acc_str = ", ".join(f"{snr_key(s)}={a * 100:.2f}%" for s, a in zip(select_snr_levels, select_accs))
            print(
                f"Epoch {epoch + 1:3d}/{args.epochs} "
                f"loss={stats['loss']:.4f} cls={stats['loss_cls']:.4f} rec={stats['loss_recon']:.4f} "
                f"train={stats['train_acc'] * 100:.2f}% select={score * 100:.2f}% [{acc_str}] "
                f"best={best_score * 100:.2f}%@{best_epoch} {format_time(time.time() - t0)}",
                flush=True,
            )

        if patience_counter >= args.patience:
            print(f"Early stop at epoch {epoch + 1} (patience={args.patience})", flush=True)
            break

    training_time = format_time(time.time() - t0)
    best_path = output_dir / "model_best.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    print("Final SNR evaluation...", flush=True)
    snr_results = {}
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    for snr in eval_snr_levels:
        key = snr_key(snr)
        result = evaluate_once(
            model,
            args.model,
            data_ctx["X"],
            data_ctx["y"],
            data_ctx["file_paths"],
            data_ctx["labels"],
            data_ctx["test_idx"],
            snr,
            args.eval_batch_size,
            args.eval_num_workers,
            device,
            args.noise_seed,
        )
        np.savez_compressed(
            predictions_dir / f"predictions_snr_{key}.npz",
            y_true=result["y_true"],
            y_pred=result["y_pred"],
        )
        snr_results[key] = {
            "accuracy": round(result["accuracy"], 6),
            "macro_f1": round(result["macro_f1"], 6),
            "report": result["report"],
        }
        print(f"  {key:>5}: acc={result['accuracy'] * 100:.2f}% mf1={result['macro_f1'] * 100:.2f}%", flush=True)

    metrics = {
        "model_key": args.model,
        "model_label": MODEL_LABELS[args.model],
        "seed": args.seed,
        "run_name": run_name,
        "training_time": training_time,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_selection_score": round(float(best_score), 6),
        "n_params": total_params,
        "denoiser_params": denoiser_params,
        "loaded_checkpoint": loaded_checkpoint,
        "hyperparams": {
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "lambda_recon": args.lambda_recon,
            "denoise_hidden": args.denoise_hidden,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "train_snr_levels": [snr_key(x) for x in train_snr_levels],
            "select_snr_levels": [snr_key(x) for x in select_snr_levels],
            "eval_snr_levels": [snr_key(x) for x in eval_snr_levels],
            "noise_seed": args.noise_seed,
            "optical_dir": str(args.optical_dir),
            "cache_dir": str(args.cache_dir),
        },
        "snr_results": snr_results,
        "history": history,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(output_dir / "snr_results.json", "w", encoding="utf-8") as f:
        json.dump(snr_results, f, indent=2, ensure_ascii=False)
    with open(output_dir / "run_command.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(sys.argv) + "\n")
    print(f"Saved: {output_dir / 'metrics.json'}", flush=True)


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return "", ""
    if arr.size == 1:
        return round(float(arr.mean()), 6), ""
    return round(float(arr.mean()), 6), round(float(arr.std(ddof=1)), 6)


def collect_runs(results_root):
    runs = []
    for metrics_path in sorted(Path(results_root).glob("*/metrics.json")):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        if str(metrics.get("run_name", metrics_path.parent.name)).startswith("sanity"):
            continue
        metrics["_dir"] = str(metrics_path.parent)
        runs.append(metrics)
    return runs


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_aggregate(args):
    results_root = Path(args.results_root)
    out_dir = results_root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(results_root)
    if not runs:
        raise SystemExit(f"No metrics.json found under {results_root}")

    by_model = {}
    by_model_seed = {}
    for run in runs:
        model_key = run["model_key"]
        seed = str(run["seed"])
        by_model.setdefault(model_key, []).append(run)
        by_model_seed[(model_key, seed)] = run

    snr_rows = []
    f1_rows = []
    for model_key in MODEL_LABELS:
        model_runs = by_model.get(model_key, [])
        if not model_runs:
            continue
        row_acc = {"model": MODEL_LABELS[model_key], "model_key": model_key, "n_seeds": len(model_runs)}
        row_f1 = {"model": MODEL_LABELS[model_key], "model_key": model_key, "n_seeds": len(model_runs)}
        for snr in SNR_ORDER:
            acc_values = []
            f1_values = []
            for run in model_runs:
                result = run.get("snr_results", {}).get(snr)
                if result:
                    acc_values.append(result["accuracy"])
                    f1_values.append(result["macro_f1"])
            acc_mean, acc_std = mean_std(acc_values)
            f1_mean, f1_std = mean_std(f1_values)
            row_acc[f"{snr}_mean"] = acc_mean
            row_acc[f"{snr}_std"] = acc_std
            row_f1[f"{snr}_mean"] = f1_mean
            row_f1[f"{snr}_std"] = f1_std
        snr_rows.append(row_acc)
        f1_rows.append(row_f1)

    table_fields = ["model", "model_key", "n_seeds"]
    for snr in SNR_ORDER:
        table_fields.extend([f"{snr}_mean", f"{snr}_std"])
    write_csv(out_dir / "snr_accuracy_table.csv", snr_rows, table_fields)
    write_csv(out_dir / "snr_macro_f1_table.csv", f1_rows, table_fields)

    clean_rows = []
    for model_key, model_runs in by_model.items():
        acc_values = [run["snr_results"]["clean"]["accuracy"] for run in model_runs if "clean" in run.get("snr_results", {})]
        f1_values = [run["snr_results"]["clean"]["macro_f1"] for run in model_runs if "clean" in run.get("snr_results", {})]
        acc_mean, acc_std = mean_std(acc_values)
        f1_mean, f1_std = mean_std(f1_values)
        clean_rows.append(
            {
                "model": MODEL_LABELS.get(model_key, model_key),
                "model_key": model_key,
                "n_seeds": len(model_runs),
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "macro_f1_mean": f1_mean,
                "macro_f1_std": f1_std,
                "params": model_runs[0].get("n_params", ""),
                "denoiser_params": model_runs[0].get("denoiser_params", ""),
            }
        )
    write_csv(
        out_dir / "clean_performance_table.csv",
        clean_rows,
        ["model", "model_key", "n_seeds", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std", "params", "denoiser_params"],
    )

    per_class_rows = []
    for model_key, model_runs in by_model.items():
        for snr in LOW_SNR_KEYS:
            for cls in CATEGORIES:
                f1_values = []
                recall_values = []
                precision_values = []
                for run in model_runs:
                    report = run.get("snr_results", {}).get(snr, {}).get("report", {})
                    if cls in report:
                        f1_values.append(report[cls]["f1-score"])
                        recall_values.append(report[cls]["recall"])
                        precision_values.append(report[cls]["precision"])
                f1_mean, f1_std = mean_std(f1_values)
                recall_mean, recall_std = mean_std(recall_values)
                precision_mean, precision_std = mean_std(precision_values)
                per_class_rows.append(
                    {
                        "model": MODEL_LABELS.get(model_key, model_key),
                        "model_key": model_key,
                        "snr": snr,
                        "class": cls,
                        "f1_mean": f1_mean,
                        "f1_std": f1_std,
                        "recall_mean": recall_mean,
                        "recall_std": recall_std,
                        "precision_mean": precision_mean,
                        "precision_std": precision_std,
                    }
                )
    write_csv(
        out_dir / "low_snr_per_class_table.csv",
        per_class_rows,
        [
            "model",
            "model_key",
            "snr",
            "class",
            "f1_mean",
            "f1_std",
            "recall_mean",
            "recall_std",
            "precision_mean",
            "precision_std",
        ],
    )

    mcnemar_rows = compute_mcnemar_rows(by_model_seed)
    write_csv(out_dir / "mcnemar_dn_vs_av2.csv", mcnemar_rows, ["seed", "snr", "b", "c", "chi2", "p_value"])

    with open(out_dir / "aggregate_noise_paper.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_runs": len(runs),
                "runs": [
                    {
                        "model_key": run["model_key"],
                        "model_label": run["model_label"],
                        "seed": run["seed"],
                        "dir": run["_dir"],
                    }
                    for run in runs
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Aggregated {len(runs)} runs into {out_dir}", flush=True)


def compute_mcnemar_rows(by_model_seed):
    rows = []
    try:
        from scipy.stats import chi2 as chi2_dist
    except Exception:
        chi2_dist = None
    seeds = sorted({seed for model, seed in by_model_seed if model == "dn_foh"})
    for seed in seeds:
        dn = by_model_seed.get(("dn_foh", seed))
        av2 = by_model_seed.get(("av2", seed))
        if not dn or not av2:
            continue
        for snr in LOW_SNR_KEYS:
            dn_pred = Path(dn["_dir"]) / "predictions" / f"predictions_snr_{snr}.npz"
            av2_pred = Path(av2["_dir"]) / "predictions" / f"predictions_snr_{snr}.npz"
            if not dn_pred.exists() or not av2_pred.exists():
                continue
            dn_data = np.load(dn_pred)
            av2_data = np.load(av2_pred)
            y_true = dn_data["y_true"]
            dn_correct = dn_data["y_pred"] == y_true
            av2_correct = av2_data["y_pred"] == y_true
            b = int(np.logical_and(dn_correct, np.logical_not(av2_correct)).sum())
            c = int(np.logical_and(np.logical_not(dn_correct), av2_correct).sum())
            chi2 = ((abs(b - c) - 1) ** 2) / max(b + c, 1)
            p_value = float(chi2_dist.sf(chi2, 1)) if chi2_dist is not None else ""
            rows.append({"seed": seed, "snr": snr, "b": b, "c": c, "chi2": round(float(chi2), 6), "p_value": p_value})
    return rows


def add_common_train_args(parser):
    parser.add_argument("--model", required=True, choices=sorted(MODEL_LABELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--optical-dir", default=str(DEFAULT_OPTICAL_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--init-checkpoint", default=None, help="Checkpoint path, 'auto', or omitted.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-recon", type=float, default=0.05)
    parser.add_argument("--denoise-hidden", type=int, default=24)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--train-snr-levels", nargs="+", default=None)
    parser.add_argument("--select-snr-levels", nargs="+", default=None)
    parser.add_argument("--eval-snr-levels", nargs="+", default=SNR_ORDER)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--sanity-run", action="store_true")


def apply_model_defaults(args):
    if args.batch_size is None:
        args.batch_size = 64 if args.model == "expb" else 16 if args.model == "av2" else 8
    if args.epochs is None:
        args.epochs = 80 if args.model in {"foh", "av2", "expb"} else 35
    if args.lr is None:
        args.lr = 1e-3 if args.model in {"foh", "av2", "expb"} else 3e-4
    if args.model in {"dn_foh", "denoise_foh", "aug_foh"} and args.init_checkpoint is None:
        args.init_checkpoint = "auto"
    return args


def build_parser():
    parser = argparse.ArgumentParser(description="Noise robustness experiments for FOH-MSCN")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train")
    add_common_train_args(train_parser)

    agg_parser = sub.add_parser("aggregate")
    agg_parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        args = apply_model_defaults(args)
        run_train(args)
    elif args.command == "aggregate":
        run_aggregate(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
