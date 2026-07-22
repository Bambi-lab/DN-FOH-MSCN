"""Canonical ShipsEar-to-FOH transforms preserved from the project sources.

The enhanced generator intentionally uses one legacy RandomState stream across
the sorted dataset. This matches the draw order of wav2mat_optical_enhanced.py.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import savemat
from scipy.io.wavfile import read as read_wav


CATEGORIES = ("A", "B", "C", "D", "E")


@dataclass(frozen=True)
class EnhancedSimulationConfig:
    a_dc: float = 1.0
    phase_white_std_rad: float = 0.005
    phase_drift_std_rad: float = 0.02
    phase_drift_control_points_per_s: float = 2.0
    visibility_min: float = 0.6
    visibility_max: float = 1.0
    detector_noise_std: float = 0.005
    seed: int = 42


def pcm_full_scale(audio: np.ndarray) -> np.ndarray:
    """Apply the exact PCM full-scale conversion used by the source generator."""
    if audio.dtype == np.int16:
        return audio.astype(np.float64) / 32768.0
    if audio.dtype == np.int32:
        return audio.astype(np.float64) / 2147483648.0
    if audio.dtype in (np.float32, np.float64):
        return audio.astype(np.float64)
    return audio.astype(np.float64) / 32768.0


def load_wav_phase(path: str | Path) -> tuple[int, np.ndarray]:
    sample_rate, audio = read_wav(Path(path))
    if audio.ndim != 1:
        raise ValueError(f"Expected mono WAV, received shape {audio.shape}")
    return int(sample_rate), pcm_full_scale(audio)


def simulate_ideal(phase: np.ndarray, a_dc: float = 1.0, visibility: float = 1.0) -> np.ndarray:
    phase = np.asarray(phase, dtype=np.float64)
    offsets = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    return a_dc + visibility * np.cos(phase[:, None] + offsets[None, :])


def generate_smooth_phase_noise(
    n_samples: int,
    sample_rate: int,
    rng: np.random.RandomState,
    white_std: float,
    drift_std: float,
    control_points_per_s: float,
) -> np.ndarray:
    white = rng.randn(n_samples) * white_std
    duration_s = n_samples / sample_rate
    n_control = max(3, int(duration_s * control_points_per_s))
    control_values = rng.randn(n_control) * drift_std
    control_x = np.linspace(0, n_samples - 1, n_control)
    drift = np.interp(np.arange(n_samples), control_x, control_values)
    return white + drift


def simulate_enhanced(
    phase: np.ndarray,
    sample_rate: int,
    rng: np.random.RandomState,
    config: EnhancedSimulationConfig = EnhancedSimulationConfig(),
) -> tuple[np.ndarray, dict]:
    phase = np.asarray(phase, dtype=np.float64)
    phase_noise = generate_smooth_phase_noise(
        len(phase), sample_rate, rng,
        config.phase_white_std_rad,
        config.phase_drift_std_rad,
        config.phase_drift_control_points_per_s,
    )
    noisy_phase = phase + phase_noise
    visibility = float(rng.uniform(config.visibility_min, config.visibility_max))
    optical = simulate_ideal(noisy_phase, config.a_dc, visibility)
    optical[:, 0] += rng.randn(len(phase)) * config.detector_noise_std
    optical[:, 1] += rng.randn(len(phase)) * config.detector_noise_std
    optical[:, 2] += rng.randn(len(phase)) * config.detector_noise_std
    metadata = {
        "visibility": visibility,
        "phase_signal_rms": float(np.sqrt(np.mean(phase ** 2))),
        "phase_noise_rms": float(np.sqrt(np.mean(phase_noise ** 2))),
        "config": asdict(config),
    }
    return optical, metadata


def differential_channels(optical: np.ndarray) -> np.ndarray:
    optical = np.asarray(optical)
    if optical.ndim != 2 or optical.shape[1] != 3:
        raise ValueError(f"Expected (samples, 3), received {optical.shape}")
    v1, v2, v3 = optical.T
    return np.column_stack((v1 - v2, v2 - v3, v1 - v3))


def demodulate_wrapped_phase(optical: np.ndarray) -> np.ndarray:
    """Use the exact preprocessing-code sign convention: sqrt(3)*(V3-V2)."""
    v1, v2, v3 = np.asarray(optical).T
    return np.arctan2(np.sqrt(3.0) * (v3 - v2), 2.0 * v1 - v2 - v3)


def save_optical_mat(path: str | Path, optical: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {"Vo3x3": np.asarray(optical)})
