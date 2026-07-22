from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foh_simulation import (  # noqa: E402
    EnhancedSimulationConfig,
    demodulate_wrapped_phase,
    differential_channels,
    simulate_enhanced,
    simulate_ideal,
)
from gammatone_features import extract_gammatone_spectrum  # noqa: E402
from scripts.noise_paper_experiment import DenoisedFOHMSCN, FOHMSCN, count_params  # noqa: E402


def main():
    sr = 12000
    t = np.arange(36000) / sr
    phase = 0.25 * np.sin(2 * np.pi * 137 * t) + 0.05 * np.sin(2 * np.pi * 431 * t)
    ideal = simulate_ideal(phase)
    np.testing.assert_allclose(ideal.sum(axis=1), 3.0, atol=2e-15, rtol=0)
    diff = differential_channels(ideal)
    residual = float(np.max(np.abs(diff[:, 2] - diff[:, 0] - diff[:, 1])))
    assert residual < 1e-14
    cfg = EnhancedSimulationConfig(seed=42)
    enhanced_a, _ = simulate_enhanced(phase, sr, np.random.RandomState(42), cfg)
    enhanced_b, _ = simulate_enhanced(phase, sr, np.random.RandomState(42), cfg)
    np.testing.assert_array_equal(enhanced_a, enhanced_b)
    demodulated = demodulate_wrapped_phase(enhanced_a)
    gammatone, centers = extract_gammatone_spectrum(demodulated, sr=sr)
    assert gammatone.shape == (128, 139)
    assert centers.shape == (128,)
    backbone = FOHMSCN().eval()
    full = DenoisedFOHMSCN().eval()
    with torch.no_grad():
        output = backbone(torch.from_numpy(diff.T[None].astype(np.float32)))
        full_output = full(torch.from_numpy(diff.T[None].astype(np.float32)))
    assert tuple(output.shape) == (1, 5)
    assert tuple(full_output.shape) == (1, 5)
    assert count_params(backbone) == 311350
    assert count_params(full) == 312383
    assert count_params(full.denoiser) == 1033
    print("SMOKE_TEST_PASS")
    print(f"d3_identity_max_abs_residual={residual:.3e}")
    print(f"gammatone_shape={gammatone.shape}")
    print(f"backbone_params={count_params(backbone)}")
    print(f"full_params={count_params(full)}")
    print(f"denoiser_params={count_params(full.denoiser)}")


if __name__ == "__main__":
    main()
