from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from foh_simulation import (
    CATEGORIES,
    EnhancedSimulationConfig,
    load_wav_phase,
    save_optical_mat,
    simulate_enhanced,
    simulate_ideal,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ideal or enhanced 3x3 FOH .mat files")
    parser.add_argument("--input-dir", type=Path, required=True, help="Five-class WAV root (A--E)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=("ideal", "enhanced"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-class", type=int, default=None, help="Smoke-test limit only")
    args = parser.parse_args()

    config = EnhancedSimulationConfig(seed=args.seed)
    rng = np.random.RandomState(args.seed)
    rows = []
    failures = []
    for class_id in CATEGORIES:
        wav_files = sorted((args.input_dir / class_id).glob("*.wav"))
        if args.limit_per_class is not None:
            wav_files = wav_files[: args.limit_per_class]
        for wav_path in wav_files:
            try:
                sample_rate, phase = load_wav_phase(wav_path)
                if args.condition == "ideal":
                    optical = simulate_ideal(phase)
                    sample_meta = {"visibility": 1.0}
                else:
                    optical, sample_meta = simulate_enhanced(phase, sample_rate, rng, config)
                output_name = f"vo3x3_{wav_path.stem}.mat"
                save_optical_mat(args.output_dir / class_id / output_name, optical)
                rows.append({
                    "class_id": class_id,
                    "source_id": f"{class_id}/{wav_path.name}",
                    "output_id": f"{class_id}/{output_name}",
                    "sample_rate": sample_rate,
                    "n_samples": len(phase),
                    "condition": args.condition,
                    "seed": args.seed,
                    "visibility": sample_meta["visibility"],
                })
            except Exception as exc:
                failures.append({"source_id": f"{class_id}/{wav_path.name}", "error": repr(exc)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "generation_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["class_id"])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "condition": args.condition,
        "seed": args.seed,
        "ordering": "classes A--E; lexicographically sorted WAV names; one shared RNG stream",
        "simulation_config": config.__dict__ if args.condition == "enhanced" else {"a_dc": 1.0, "visibility": 1.0},
        "generated_files": len(rows),
        "failures": failures,
    }
    (args.output_dir / "generation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"generated": len(rows), "failed": len(failures), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
