"""
Experiment C: FOH-MSCN — Multi-Scale Channel-aware Network for 3×3 fiber-optic hydrophone differential interference signals

Input: 3 differential channels (V1-V2, V2-V3, V1-V3), shape=(batch, 3, 36000)
Architecture:
  InstanceNorm1d → MSConv(multi-scale branches) → CTA(channel attention) → BiGRU → temporal attention pooling → classification head

Data: shipsear_optical_enhanced (enhanced simulation)
Cache: cache_enhanced/raw_v2_diff3_X.npy (pre-existing, reused directly)
Output: results/optical_classification/exp_C_foh_mscn_enhanced/
"""

import sys
import json
import argparse
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_optical import (
    scan_dataset, load_raw_data_v2, get_split_indices, get_split_by_mode,
    compute_class_weights, compute_sample_weights,
    IndexDataset, CATEGORIES, OPTICAL_DIR as _DEFAULT_OPTICAL_DIR
)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# Network modules
# ============================================================

class MSConvBlock(nn.Module):
    """
    Multi-scale 1D convolution block — captures short (k=7), medium (k=15), and long (k=31) temporal features simultaneously.
    """
    def __init__(self, in_channels, out_per_branch=32, out_proj=128):
        super().__init__()
        self.branch_s = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(out_per_branch), nn.GELU()
        )
        self.branch_m = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(out_per_branch), nn.GELU()
        )
        self.branch_l = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=31, stride=2, padding=15, bias=False),
            nn.BatchNorm1d(out_per_branch), nn.GELU()
        )
        concat_ch = out_per_branch * 3
        self.proj = nn.Sequential(
            nn.Conv1d(concat_ch, out_proj, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_proj), nn.GELU()
        )

    def forward(self, x):
        s = self.branch_s(x)
        m = self.branch_m(x)
        l = self.branch_l(x)
        x = torch.cat([s, m, l], dim=1)
        return self.proj(x)


class ChannelAttention(nn.Module):
    """
    Squeeze-and-excitation style channel attention — adaptively emphasizes important feature channels.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(),
            nn.Linear(mid, channels), nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T)
        w = self.se(x).unsqueeze(-1)  # (B, C, 1)
        return x * w


class TemporalAttentionPool(nn.Module):
    """
    Temporal attention pooling — weighted sum over all timesteps (instead of taking the last timestep).
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, D)
        w = torch.softmax(self.score(x), dim=1)  # (B, T, 1)
        return (x * w).sum(dim=1)                 # (B, D)


class FOHMSCN(nn.Module):
    """
    FOH-MSCN: Multi-Scale Channel-aware Network for FOH signals

    Input:  (batch, 3, 36000) — 3 differential channels
    Output: (batch, num_classes)

    Temporal compression path:
      MSConv(stride=2) → (B,128,18000)
      MaxPool(4)       → (B,128,4500)
      Conv+MaxPool(4)  → (B,128,1125)
      Conv+MaxPool(4)  → (B,128,281)
      ChannelAttn      → (B,128,281)
      BiGRU            → (B,281,128)
      TemporalAttnPool → (B,128)
      FC               → (B,5)
    """
    def __init__(self, num_classes=5, in_channels=3,
                 msconv_per_branch=32, msconv_proj=128,
                 gru_hidden=64, gru_layers=2, dropout=0.4):
        super().__init__()

        self.input_norm = nn.InstanceNorm1d(in_channels)

        # Multi-scale conv block: (B,3,36000) → (B,128,18000)
        self.msconv = MSConvBlock(in_channels, msconv_per_branch, msconv_proj)

        # Compression stage 1: (B,128,18000) → (B,128,4500) → (B,128,1125)
        self.stage1 = nn.Sequential(
            nn.MaxPool1d(4),
            nn.Conv1d(msconv_proj, msconv_proj, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(msconv_proj), nn.GELU(),
            nn.MaxPool1d(4),
        )

        # Compression stage 2: (B,128,1125) → (B,128,281)
        self.stage2 = nn.Sequential(
            nn.Conv1d(msconv_proj, msconv_proj, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(msconv_proj), nn.GELU(),
            nn.MaxPool1d(4),
        )

        # Channel attention
        self.chan_attn = ChannelAttention(msconv_proj, reduction=8)

        # BiGRU: input_size=128, hidden=gru_hidden, bidirectional → output=gru_hidden*2
        self.bigru = nn.GRU(
            input_size=msconv_proj,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )
        gru_out_dim = gru_hidden * 2  # 128

        # Temporal attention pooling
        self.temp_pool = TemporalAttentionPool(gru_out_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, 3, 36000)
        x = self.input_norm(x)           # InstanceNorm
        x = self.msconv(x)               # (B, 128, 18000)
        x = self.stage1(x)               # (B, 128, 1125)
        x = self.stage2(x)               # (B, 128, 281)
        x = self.chan_attn(x)            # (B, 128, 281)
        x = x.permute(0, 2, 1)          # (B, 281, 128)
        x, _ = self.bigru(x)             # (B, 281, 128)
        x = self.temp_pool(x)            # (B, 128)
        x = self.classifier(x)           # (B, 5)
        return x


# ============================================================
# Utility functions
# ============================================================

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(round(seconds))))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Training and evaluation
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, count = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y_batch).sum().item()
        count += y_batch.size(0)
    return total_loss / len(loader), correct / count


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, count = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            total_loss += criterion(out, y_batch).item()
            correct += (out.argmax(1) == y_batch).sum().item()
            count += y_batch.size(0)
    return total_loss / len(loader), correct / count


def predict_all(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ============================================================
# Main function
# ============================================================

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ── Optional data path overrides (for E2 ideal simulation) ───────────────────
    import dataset_optical as _ds_mod
    if args.optical_dir:
        _ds_mod.OPTICAL_DIR = args.optical_dir
    if args.cache_dir:
        _ds_mod.CACHE_DIR = args.cache_dir
        import os; os.makedirs(args.cache_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Experiment C: FOH-MSCN main experiment")
    print(f"  OPTICAL_DIR: {_ds_mod.OPTICAL_DIR}")
    print(f"  CACHE_DIR:   {_ds_mod.CACHE_DIR}")
    print("=" * 55)

    file_paths, labels = scan_dataset(_ds_mod.OPTICAL_DIR)
    X, y = load_raw_data_v2(file_paths, labels)   # reuse diff-3ch cache
    print(f"Data shape: X={X.shape}, y={y.shape}")

    split_mode = getattr(args, 'split_mode', '80-20')
    split = get_split_by_mode(y, split_mode)
    train_idx = split['train']; val_idx = split['val']; test_idx = split['test']
    print(f"Split mode: {split_mode}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx) if val_idx is not None else 'N/A'}, Test: {len(test_idx)}")

    # sanity run: only use first N samples
    if args.sanity_run:
        print("\n[SANITY RUN] mode (2 epochs, small subset)")
        N = 200
        train_idx = train_idx[:N]
        if val_idx is not None:
            val_idx = val_idx[:50]
        test_idx = test_idx[:50]
        args.epochs = 2
        args.patience = 999

    class_weights = compute_class_weights(y).to(device)
    sample_weights = compute_sample_weights(y, train_idx)
    sampler = WeightedRandomSampler(sample_weights, len(train_idx), replacement=True)

    train_ds = IndexDataset(X, y, train_idx)
    test_ds  = IndexDataset(X, y, test_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.num_workers,
                              pin_memory=(device.type == 'cuda'))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=(device.type == 'cuda'))

    val_loader = None
    if val_idx is not None:
        val_ds = IndexDataset(X, y, val_idx)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=(device.type == 'cuda'))

    # ── Model ──────────────────────────────────────────────────
    model = FOHMSCN(num_classes=5, in_channels=3,
                    msconv_per_branch=args.msconv_per_branch,
                    msconv_proj=args.msconv_proj,
                    gru_hidden=args.gru_hidden,
                    gru_layers=args.gru_layers,
                    dropout=args.dropout).to(device)

    n_params = count_params(model)
    print(f"\nModel parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs // 2, eta_min=1e-5)

    # ── Training loop ──────────────────────────────────────────────
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    patience_counter = 0
    monitor_loader = val_loader if val_loader is not None else test_loader
    monitor_name = "val" if val_loader is not None else "test"

    print(f"\nTraining started (batch={args.batch_size}, lr={args.lr}, epochs={args.epochs}, "
          f"monitor={monitor_name})\n")
    t0 = time.time()

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_epoch(model, monitor_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(va_loss)
        val_accs.append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / 'model_best.pth')
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch < 3:
            elapsed = format_time(time.time() - t0)
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
                  f"{monitor_name.capitalize()} Loss {va_loss:.4f} Acc {va_acc:.4f} | "
                  f"Best {best_val_acc:.4f} | LR {lr_now:.2e} | {elapsed}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

    training_time = format_time(time.time() - t0)
    print(f"\nTraining time: {training_time}")

    # ── Final evaluation ──────────────────────────────────────────────
    model.load_state_dict(torch.load(output_dir / 'model_best.pth'))
    y_pred, y_true = predict_all(model, test_loader, device)
    final_acc = np.mean(y_pred == y_true)
    print(f"\nFinal test accuracy: {final_acc * 100:.2f}%")

    report_str = classification_report(y_true, y_pred, target_names=CATEGORIES, digits=4)
    print("\nClassification report:")
    print(report_str)

    # ── Save results ──────────────────────────────────────────────
    # 1. classification_report.txt
    with open(output_dir / 'classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"Experiment C: FOH-MSCN enhanced simulation\n")
        f.write(f"Training time: {training_time}\n")
        f.write(f"Parameters: {n_params:,}\n")
        f.write(f"Final test accuracy: {final_acc * 100:.2f}%\n")
        f.write(f"Best val accuracy: {best_val_acc * 100:.2f}%\n\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  batch_size={args.batch_size}, lr={args.lr}, epochs={args.epochs}\n")
        f.write(f"  gru_hidden={args.gru_hidden}, gru_layers={args.gru_layers}\n")
        f.write(f"  msconv_per_branch={args.msconv_per_branch}, msconv_proj={args.msconv_proj}\n\n")
        f.write(report_str)

    # 2. metrics.json
    from sklearn.metrics import classification_report as cr
    cr_dict = cr(y_true, y_pred, target_names=CATEGORIES, output_dict=True)
    metrics = {
        "experiment": "E1_FOH_MSCN_enhanced",
        "final_test_acc": round(float(final_acc), 6),
        "best_val_acc": round(float(best_val_acc), 6),
        "training_time": training_time,
        "n_params": n_params,
        "epochs_trained": len(train_losses),
        "hyperparams": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
            "gru_hidden": args.gru_hidden,
            "gru_layers": args.gru_layers,
            "msconv_per_branch": args.msconv_per_branch,
            "msconv_proj": args.msconv_proj,
            "dropout": args.dropout,
        },
        "per_class": {
            cat: {
                "precision": round(cr_dict[cat]["precision"], 4),
                "recall":    round(cr_dict[cat]["recall"],    4),
                "f1":        round(cr_dict[cat]["f1-score"],  4),
                "support":   cr_dict[cat]["support"],
            }
            for cat in CATEGORIES
        },
        "macro_avg": {k: round(v, 4) for k, v in cr_dict["macro avg"].items()},
        "weighted_avg": {k: round(v, 4) for k, v in cr_dict["weighted avg"].items()},
    }
    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"metrics.json saved")

    # 3. confusion_matrix.png
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'FOH-MSCN Confusion Matrix (enhanced sim {final_acc*100:.2f}%)')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()

    # 4. training_curves.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_losses, label='Train Loss', linewidth=1.5)
    ax1.plot(val_losses,   label='Val Loss', linewidth=1.5)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.set_title('Loss Curves'); ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label='Train Acc', linewidth=1.5)
    ax2.plot(val_accs,   label='Val Acc', linewidth=1.5)
    ax2.axhline(y=0.90, color='red', linestyle='--', alpha=0.5, label='Target 90%')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.set_title('Accuracy Curves'); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\nAll results saved to: {output_dir}")
    print(f"\n{'='*55}")
    print(f"  Final accuracy: {final_acc*100:.2f}%")
    print(f"  Class C recall: {cr_dict['C']['recall']:.4f}")
    print(f"  Parameters:     {n_params:,}")
    print(f"{'='*55}")

    return final_acc, best_val_acc


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FOH-MSCN enhanced simulation main experiment (E1)')

    # Run mode
    parser.add_argument('--sanity-run', action='store_true',
                        help='Sanity run: 2 epochs on 200 samples')
    parser.add_argument('--split-mode', type=str, default='80-20',
                        choices=['80-20', '70-10-20'],
                        help='Data split mode: 80-20 (traditional) or 70-10-20 (with validation set)')

    # Training hyperparameters
    parser.add_argument('--batch-size',  type=int,   default=8,     help='Batch size (default 8; increase to 16 if VRAM permits)')
    parser.add_argument('--lr',          type=float, default=1e-3,  help='Learning rate')
    parser.add_argument('--epochs',      type=int,   default=100,   help='Maximum training epochs')
    parser.add_argument('--patience',    type=int,   default=15,    help='Early stopping patience')
    parser.add_argument('--seed',        type=int,   default=42,    help='Random seed')
    parser.add_argument('--num-workers', type=int,   default=0,     help='DataLoader workers')

    # Model hyperparameters
    parser.add_argument('--msconv-per-branch', type=int,   default=32,  help='MSConv channels per branch')
    parser.add_argument('--msconv-proj',       type=int,   default=128, help='MSConv projection channels')
    parser.add_argument('--gru-hidden',        type=int,   default=64,  help='GRU hidden size')
    parser.add_argument('--gru-layers',        type=int,   default=2,   help='GRU number of layers')
    parser.add_argument('--dropout',           type=float, default=0.4, help='Dropout rate')

    # Output
    parser.add_argument('--output-dir', type=str,
                        default='./results/exp_C_foh_mscn_enhanced',
                        help='Results output directory')
    # Data path overrides (for E2 ideal simulation)
    parser.add_argument('--optical-dir', type=str, default=None,
                        help='Override OPTICAL_DIR (e.g. switch to ideal simulation)')
    parser.add_argument('--cache-dir',   type=str, default=None,
                        help='Override CACHE_DIR')

    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("FOH-MSCN — Enhanced simulation ship classification (E1)")
    print("=" * 55)
    print(f"Config: batch={args.batch_size} lr={args.lr} epochs={args.epochs} "
          f"patience={args.patience} seed={args.seed}")
    print(f"Model: msconv_per_branch={args.msconv_per_branch} "
          f"msconv_proj={args.msconv_proj} "
          f"gru_hidden={args.gru_hidden} layers={args.gru_layers} "
          f"dropout={args.dropout}")

    main(args)
