from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
from scripts import noise_paper_experiment as exp  # noqa: E402


def indices_from_manifest(manifest: Path, file_paths, split: str) -> np.ndarray:
    with manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = [f"{Path(path).parent.name}/{Path(path).name}" for path in file_paths]
    observed = [row["sample_id"] for row in rows]
    if expected != observed:
        raise ValueError("Split manifest sample order does not match the scanned dataset")
    return np.asarray([int(row["index"]) for row in rows if row["split"] == split], dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved project checkpoint on a fixed manifest")
    parser.add_argument("--model", required=True, choices=sorted(exp.MODEL_LABELS))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--optical-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=REPOSITORY_ROOT / "cache")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--snr-levels", nargs="+", default=["clean", "30", "20", "10", "5", "0"])
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "results" / "evaluation.json")
    args = parser.parse_args()

    exp.configure_dataset_paths(args.optical_dir, args.cache_dir)
    file_paths, labels = exp.scan_dataset(str(args.optical_dir))
    if args.model == "expb":
        X, y = exp.load_phase_gammatone_data(file_paths, labels)
        input_shape = (X.shape[1], X.shape[2])
    else:
        X, y = exp.load_raw_data_v2(file_paths, labels)
        input_shape = None
    indices = indices_from_manifest(args.split_manifest, file_paths, args.split)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = exp.make_model(args.model, input_shape=input_shape).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    results = {}
    for token in args.snr_levels:
        snr = exp.parse_snr_token(token)
        result = exp.evaluate_once(
            model, args.model, X, y, file_paths, labels, indices, snr,
            args.batch_size, 0, device, args.noise_seed,
        )
        results[exp.snr_key(snr)] = {
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "n_samples": len(result["y_true"]),
        }
    payload = {
        "model": args.model,
        "checkpoint_sha256": __import__("hashlib").sha256(args.checkpoint.read_bytes()).hexdigest(),
        "split_manifest": args.split_manifest.name,
        "noise_seed": args.noise_seed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
