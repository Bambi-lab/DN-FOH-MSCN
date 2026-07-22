"""
noise_generators.py
Non-AWGN noise generators for FOH robustness evaluation.

Noise types:
  N1: Pink noise (1/f spectrum, Voss-McCartney algorithm)
  N2: Low-frequency band-limited noise (white → 10-500 Hz bandpass)
  N3: ShipsEar background mixing (Class-E segments mixed at target SNR)
  N4: Combined N1+N3 (equal power)

Usage:
  from noise_generators import generate_pink_noise, add_noise_at_snr
"""

import numpy as np
from scipy.signal import butter, sosfilt


def generate_pink_noise(n_samples, n_channels=3, rng=None):
    """Generate pink noise (1/f) via Voss-McCartney algorithm.
    Returns array of shape (n_channels, n_samples).
    """
    if rng is None:
        rng = np.random
    # Use multiple white noise sources summed with different time constants
    n_octaves = 16
    pink = np.zeros((n_channels, n_samples), dtype=np.float64)
    for ch in range(n_channels):
        state = rng.randn(n_octaves)
        for i in range(n_samples):
            idx = (i + 1) & -(i + 1)  # lowest set bit position
            octave = 0
            while (idx >> octave) & 1 == 0:
                octave += 1
            if octave >= n_octaves:
                octave = n_octaves - 1
            state[octave] = rng.randn()
            pink[ch, i] = state.sum()
    # Normalize to unit RMS
    pink = pink.astype(np.float32)
    rms = np.sqrt(np.mean(pink ** 2, axis=-1, keepdims=True))
    pink = pink / np.maximum(rms, 1e-12)
    return pink


def generate_band_limited_noise(n_samples, n_channels=3, sr=12000,
                                lowcut=10, highcut=500, rng=None):
    """Generate band-limited Gaussian noise via Butterworth bandpass.
    Returns array of shape (n_channels, n_samples).
    """
    if rng is None:
        rng = np.random
    white = rng.randn(n_channels, n_samples).astype(np.float32)
    nyquist = sr / 2.0
    sos = butter(4, [lowcut / nyquist, highcut / nyquist],
                 btype='band', output='sos')
    filtered = sosfilt(sos, white, axis=-1).astype(np.float32)
    # Normalize to unit RMS
    rms = np.sqrt(np.mean(filtered ** 2, axis=-1, keepdims=True))
    filtered = filtered / np.maximum(rms, 1e-12)
    return filtered


def add_noise_at_snr(signal, noise, snr_db):
    """Add noise to signal at specified SNR (dB).
    SNR defined as RMS(signal) / RMS(noise).

    Args:
        signal: (B, C, T) or (C, T) numpy array
        noise:  (B, C, T) or (C, T) numpy array
        snr_db: float

    Returns:
        noisy_signal: same shape as signal
    """
    sig_power = np.mean(signal ** 2, axis=-1, keepdims=True)
    sig_power = np.maximum(sig_power, 1e-12)
    noise_power = np.mean(noise ** 2, axis=-1, keepdims=True)
    noise_power = np.maximum(noise_power, 1e-12)
    scale = np.sqrt(sig_power / noise_power) / (10 ** (snr_db / 20.0))
    return signal + noise * scale


def make_pink_noise_like(signal, rng=None):
    """Generate pink noise matching signal shape."""
    if rng is None:
        rng = np.random
    if signal.ndim == 3:
        B, C, T = signal.shape
        return generate_pink_noise(T, C, rng=rng)[None].repeat(B, axis=0)
    else:
        C, T = signal.shape
        return generate_pink_noise(T, C, rng=rng)


def make_band_limited_noise_like(signal, sr=12000, lowcut=10, highcut=500, rng=None):
    """Generate band-limited noise matching signal shape."""
    if rng is None:
        rng = np.random
    if signal.ndim == 3:
        B, C, T = signal.shape
        return generate_band_limited_noise(T, C, sr, lowcut, highcut, rng=rng)[None].repeat(B, axis=0)
    else:
        C, T = signal.shape
        return generate_band_limited_noise(T, C, sr, lowcut, highcut, rng=rng)


# Preload ShipsEar background segments (lazy init)
_bg_cache = None


def load_shipsear_background(data_dir=None, max_files=500):
    """Load ShipsEar Class-E segments for background mixing.
    Returns list of (C, T) arrays.
    """
    global _bg_cache
    if _bg_cache is not None:
        return _bg_cache
    import glob, scipy.io as sio
    if data_dir is None:
        from dataset_optical import OPTICAL_DIR
        data_dir = OPTICAL_DIR
    bg_dir = f"{data_dir}/E"
    files = sorted(glob.glob(f"{bg_dir}/*.mat"))[:max_files]
    segments = []
    for fp in files:
        mat = sio.loadmat(fp)['Vo3x3']  # (36000, 3)
        segments.append(mat.T.astype(np.float32))  # (3, 36000)
    _bg_cache = segments
    return _bg_cache


def mix_shipsear_background(signal, snr_db, bg_dir=None, rng=None):
    """Mix a ShipsEar background segment into signal at target SNR.

    Args:
        signal: (C, T) or (B, C, T) numpy array
        snr_db: float
        bg_dir: path to optical data dir (for Class-E)
        rng: numpy random state

    Returns:
        noisy_signal: same shape as signal
    """
    if rng is None:
        rng = np.random
    segments = load_shipsear_background(bg_dir)
    bg = segments[rng.randint(0, len(segments))]  # (3, 36000)
    if signal.ndim == 3:
        bg = bg[None]  # (1, 3, 36000)
    return add_noise_at_snr(signal, bg, snr_db)
