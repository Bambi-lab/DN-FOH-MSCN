"""
dataset_optical.py
Unified fiber-optic hydrophone dataset loading module.

Supports two modes:
  - raw:   directly loads 3-channel optical interference signal (N, 3, 36000)
  - phase: phase-demodulated Gammatone spectrum features (N, 128, 139)
"""

import os
import sys
import glob
import numpy as np
import scipy.io as sio
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add project root to path for importing gammatone_features
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gammatone_features import extract_gammatone_spectrum

# Portable release defaults; CLI scripts may override these via environment variables.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPTICAL_DIR = os.environ.get('OPTICAL_DIR', str(PROJECT_ROOT / 'data' / 'foh_enhanced'))
CACHE_DIR = os.environ.get('CACHE_DIR', str(PROJECT_ROOT / 'cache'))
CATEGORIES = ['A', 'B', 'C', 'D', 'E']
SR = 12000
RANDOM_STATE = 42
TEST_SIZE = 0.2


def scan_dataset(data_dir=OPTICAL_DIR):
    """Scan the dataset and return file path list and label array."""
    file_paths = []
    labels = []
    for cat_idx, cat in enumerate(CATEGORIES):
        cat_dir = os.path.join(data_dir, cat)
        mat_files = sorted(glob.glob(os.path.join(cat_dir, '*.mat')))
        file_paths.extend(mat_files)
        labels.extend([cat_idx] * len(mat_files))
        print(f"  Class {cat}: {len(mat_files)} files")
    print(f"  Total: {len(file_paths)} files")
    return file_paths, np.array(labels)


def load_raw_data(file_paths, labels, cache_prefix='raw'):
    """Load raw 3-channel data with caching (streamed mmap write to avoid memory overflow)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_X = os.path.join(CACHE_DIR, f'{cache_prefix}_X.npy')
    cache_y = os.path.join(CACHE_DIR, f'{cache_prefix}_y.npy')

    if os.path.exists(cache_X) and os.path.exists(cache_y):
        print(f"Loading raw data from cache...")
        X = np.load(cache_X, mmap_mode='r')
        y = np.load(cache_y)
        return X, y

    N = len(file_paths)
    # Read first file to determine shape
    sample = sio.loadmat(file_paths[0])['Vo3x3']
    n_samples, n_channels = sample.shape  # (36000, 3)

    print(f"Loading raw 3-channel data ({N} files)...")
    print(f"  Pre-allocating mmap file: ({N}, {n_channels}, {n_samples}), float32")

    # Create mmap file on disk; write sample by sample to avoid consuming RAM
    X_mmap = np.lib.format.open_memmap(
        cache_X, mode='w+', dtype=np.float32,
        shape=(N, n_channels, n_samples)
    )

    for i, fp in enumerate(tqdm(file_paths)):
        mat = sio.loadmat(fp)
        vo = mat['Vo3x3']  # (36000, 3)
        X_mmap[i] = vo.T.astype(np.float32)  # (3, 36000)

        # Flush to disk every 1000 files
        if (i + 1) % 1000 == 0:
            X_mmap.flush()

    X_mmap.flush()
    del X_mmap  # Close the write handle

    np.save(cache_y, labels.copy())
    print(f"Cache saved: {cache_X}")
    return np.load(cache_X, mmap_mode='r'), labels.copy()


def load_phase_gammatone_data(file_paths, labels, cache_prefix='phase_gammatone'):
    """Load phase-demodulated Gammatone spectrum features with caching (streamed mmap write to avoid memory overflow)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_X = os.path.join(CACHE_DIR, f'{cache_prefix}_X.npy')
    cache_y = os.path.join(CACHE_DIR, f'{cache_prefix}_y.npy')

    if os.path.exists(cache_X) and os.path.exists(cache_y):
        print(f"Loading phase_gammatone data from cache...")
        X = np.load(cache_X, mmap_mode='r')
        y = np.load(cache_y)
        return X, y

    N = len(file_paths)

    # Process one file first to determine spectrum shape
    sample_mat = sio.loadmat(file_paths[0])['Vo3x3']
    V1, V2, V3 = sample_mat[:, 0], sample_mat[:, 1], sample_mat[:, 2]
    phi_sample = np.arctan2(np.sqrt(3) * (V3 - V2), 2 * V1 - V2 - V3)
    spec_sample, _ = extract_gammatone_spectrum(phi_sample, sr=SR)
    spec_shape = spec_sample.shape  # (128, 139)

    print(f"Extracting phase demodulation + Gammatone features ({N} files)...")
    print(f"  Spectrum shape: {spec_shape}, pre-allocating mmap: ({N}, {spec_shape[0]}, {spec_shape[1]})")

    X_mmap = np.lib.format.open_memmap(
        cache_X, mode='w+', dtype=np.float32,
        shape=(N, spec_shape[0], spec_shape[1])
    )

    for i, fp in enumerate(tqdm(file_paths)):
        mat = sio.loadmat(fp)
        vo = mat['Vo3x3']
        V1, V2, V3 = vo[:, 0], vo[:, 1], vo[:, 2]

        phi = np.arctan2(np.sqrt(3) * (V3 - V2), 2 * V1 - V2 - V3)
        spec, _ = extract_gammatone_spectrum(phi, sr=SR)
        X_mmap[i] = spec.astype(np.float32)

        if (i + 1) % 1000 == 0:
            X_mmap.flush()

    X_mmap.flush()
    del X_mmap

    np.save(cache_y, labels.copy())
    print(f"Cache saved: {cache_X}")
    return np.load(cache_X, mmap_mode='r'), labels.copy()


def get_split_indices(y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Get stratified split indices; shared across experiments."""
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )
    return train_idx, test_idx


def get_split_indices_val(y, val_size=0.1, test_size=None, random_state=RANDOM_STATE):
    """Get 70/10/20 train/val/test stratified split indices.

    If test_size is None, TEST_SIZE (0.2) is used.
    val_size is carved proportionally from the training set.
    Returns (train_idx, val_idx, test_idx).
    """
    if test_size is None:
        test_size = TEST_SIZE
    indices = np.arange(len(y))
    # Step 1: split off the test set
    trainval_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )
    # Step 2: split val from trainval (val_frac relative to total = val_size / (1 - test_size))
    val_frac = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_frac, random_state=random_state,
        stratify=y[trainval_idx]
    )
    return train_idx, val_idx, test_idx


def get_split_by_mode(y, split_mode='80-20', random_state=RANDOM_STATE):
    """Unified split interface.
    split_mode:
      - '80-20': traditional 80/20 (no val)
      - '70-10-20': 70/10/20 train/val/test
    Returns dict with train_idx, val_idx (or None), test_idx.
    """
    if split_mode == '70-10-20':
        train_idx, val_idx, test_idx = get_split_indices_val(y, random_state=random_state)
        return {'train': train_idx, 'val': val_idx, 'test': test_idx}
    else:  # '80-20'
        train_idx, test_idx = get_split_indices(y, random_state=random_state)
        return {'train': train_idx, 'val': None, 'test': test_idx}


def compute_class_weights(y):
    """Compute class weights (inverse of sample count)."""
    import torch
    classes, counts = np.unique(y, return_counts=True)
    weights = 1.0 / counts.astype(np.float64)
    weights = weights / weights.sum() * len(classes)  # Normalize so mean = 1
    return torch.FloatTensor(weights)


def compute_sample_weights(y, train_idx):
    """Compute per-sample weights for WeightedRandomSampler."""
    train_labels = y[train_idx]
    classes, counts = np.unique(train_labels, return_counts=True)
    class_weight = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
    sample_weights = np.array([class_weight[label] for label in train_labels])
    return sample_weights


def load_raw_data_v2(file_paths, labels, cache_prefix='raw_v2_diff3'):
    """Load 3 differential channel data (V1-V2, V2-V3, V1-V3), eliminating DC bias to magnify dynamic range."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_X = os.path.join(CACHE_DIR, f'{cache_prefix}_X.npy')
    cache_y = os.path.join(CACHE_DIR, f'{cache_prefix}_y.npy')

    if os.path.exists(cache_X) and os.path.exists(cache_y):
        print(f"Loading raw_v2 (3 diff channels) from cache...")
        X = np.load(cache_X, mmap_mode='r')
        y = np.load(cache_y)
        return X, y

    # Load raw 3-channel data first
    X_raw, y_raw = load_raw_data(file_paths, labels)

    N = len(file_paths)
    n_samples = X_raw.shape[2]  # 36000

    print(f"Computing differential channels (3 channels) ({N} files)...")
    X_mmap = np.lib.format.open_memmap(
        cache_X, mode='w+', dtype=np.float32,
        shape=(N, 3, n_samples)
    )

    for i in tqdm(range(N)):
        v = np.array(X_raw[i], dtype=np.float32)  # (3, 36000)
        V1, V2, V3 = v[0], v[1], v[2]

        # Differential 3 channels (eliminates DC, equivalent to √3·B·sin(φ+offset))
        X_mmap[i, 0] = V1 - V2
        X_mmap[i, 1] = V2 - V3
        X_mmap[i, 2] = V1 - V3

        if (i + 1) % 1000 == 0:
            X_mmap.flush()

    X_mmap.flush()
    del X_mmap

    np.save(cache_y, labels.copy())
    print(f"Cache saved: {cache_X}")
    return np.load(cache_X, mmap_mode='r'), labels.copy()


def load_raw_data_v2_6ch(file_paths, labels, cache_prefix='raw_v2_6ch'):
    """Load 6-channel data: raw 3ch (V1, V2, V3) + differential 3ch (V1-V2, V2-V3, V1-V3)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_X = os.path.join(CACHE_DIR, f'{cache_prefix}_X.npy')
    cache_y = os.path.join(CACHE_DIR, f'{cache_prefix}_y.npy')

    if os.path.exists(cache_X) and os.path.exists(cache_y):
        print(f"Loading raw_v2 (6 channels) from cache...")
        X = np.load(cache_X, mmap_mode='r')
        y = np.load(cache_y)
        return X, y

    X_raw, y_raw = load_raw_data(file_paths, labels)

    N = len(file_paths)
    n_samples = X_raw.shape[2]  # 36000

    print(f"Concatenating raw + differential channels (6 channels) ({N} files)...")
    X_mmap = np.lib.format.open_memmap(
        cache_X, mode='w+', dtype=np.float32,
        shape=(N, 6, n_samples)
    )

    for i in tqdm(range(N)):
        v = np.array(X_raw[i], dtype=np.float32)  # (3, 36000)
        V1, V2, V3 = v[0], v[1], v[2]

        # Raw 3 channels
        X_mmap[i, 0] = V1
        X_mmap[i, 1] = V2
        X_mmap[i, 2] = V3
        # Differential 3 channels
        X_mmap[i, 3] = V1 - V2
        X_mmap[i, 4] = V2 - V3
        X_mmap[i, 5] = V1 - V3

        if (i + 1) % 1000 == 0:
            X_mmap.flush()

    X_mmap.flush()
    del X_mmap

    np.save(cache_y, labels.copy())
    print(f"Cache saved: {cache_X}")
    return np.load(cache_X, mmap_mode='r'), labels.copy()


class IndexDataset(Dataset):
    """Index-based PyTorch Dataset backed by a numpy array (supports mmap to avoid loading everything into RAM)."""
    def __init__(self, X, y, indices, unsqueeze_channel=False):
        """
        Args:
            X: numpy array (can be mmap)
            y: numpy label array
            indices: indices to use
            unsqueeze_channel: whether to add a channel dimension at dim=0 (for 2D CNN)
        """
        self.X = X
        self.y = y
        self.indices = indices
        self.unsqueeze_channel = unsqueeze_channel

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        import torch
        i = self.indices[idx]
        x = torch.from_numpy(np.array(self.X[i], dtype=np.float32))
        if self.unsqueeze_channel:
            x = x.unsqueeze(0)  # (1, H, W)
        label = int(self.y[i])
        return x, label


if __name__ == '__main__':
    print("=== Scan dataset ===")
    file_paths, labels = scan_dataset()

    print("\n=== Test raw loading ===")
    X_raw, y_raw = load_raw_data(file_paths, labels)
    print(f"  X_raw shape: {X_raw.shape}, y_raw shape: {y_raw.shape}")

    print("\n=== Test split ===")
    train_idx, test_idx = get_split_indices(y_raw)
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    print("\n=== Class weights ===")
    weights = compute_class_weights(y_raw)
    for i, cat in enumerate(CATEGORIES):
        print(f"  {cat}: weight={weights[i]:.4f}")
