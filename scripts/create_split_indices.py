from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


CATEGORIES = ("A", "B", "C", "D", "E")
AUGMENTATION_SUFFIXES = (
    "_original_denoised", "_pitch_denoised", "_random_noise_denoised",
    "_shift_denoised", "_snr_noise_denoised", "_stretch_denoised",
)


def recording_id(sample_name: str) -> str:
    stem = Path(sample_name).stem
    stem = re.sub(r"_seg\d+$", "", stem)
    for suffix in AUGMENTATION_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def scan(root: Path):
    rows = []
    for label, class_id in enumerate(CATEGORIES):
        files = sorted((root / class_id).glob("*.mat"))
        if not files:
            files = sorted((root / class_id).glob("*.wav"))
        for path in files:
            rows.append({
                "index": len(rows),
                "sample_id": f"{class_id}/{path.name}",
                "class_id": class_id,
                "label": label,
                "recording_id": recording_id(path.name),
            })
    if not rows:
        raise FileNotFoundError(f"No .mat or .wav samples under {root}")
    return rows


def segment_split(rows, seed: int):
    labels = np.asarray([row["label"] for row in rows])
    indices = np.arange(len(rows))
    trainval, test = train_test_split(indices, test_size=0.2, random_state=seed, stratify=labels)
    train, val = train_test_split(
        trainval, test_size=0.1 / 0.8, random_state=seed, stratify=labels[trainval]
    )
    assignment = {int(i): "train" for i in train}
    assignment.update({int(i): "val" for i in val})
    assignment.update({int(i): "test" for i in test})
    return assignment


def recording_split(rows, seed: int):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["label"]][row["recording_id"]].append(row["index"])
    rng = np.random.default_rng(seed)
    assignment = {}
    for label in sorted(grouped):
        rec_ids = list(grouped[label])
        n_test = max(1, int(len(rec_ids) * 0.2))
        n_test = min(n_test, len(rec_ids) - 1)
        rng.shuffle(rec_ids)
        test_ids = set(rec_ids[:n_test])
        for rec_id, indices in grouped[label].items():
            split = "test" if rec_id in test_ids else "train"
            assignment.update({int(i): split for i in indices})
    return assignment


def write_manifest(path: Path, rows, assignment, protocol: str, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["index", "sample_id", "class_id", "label", "recording_id", "split", "protocol", "split_seed"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "split": assignment[row["index"]], "protocol": protocol, "split_seed": seed})


def main():
    parser = argparse.ArgumentParser(description="Create deterministic segment and recording split manifests")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segment-seed", type=int, default=42)
    parser.add_argument("--recording-seeds", type=int, nargs="+", default=[42, 123, 456])
    args = parser.parse_args()
    rows = scan(args.data_dir)
    outputs = []
    segment_path = args.output_dir / f"segment_70_10_20_seed{args.segment_seed}.csv"
    write_manifest(segment_path, rows, segment_split(rows, args.segment_seed), "segment_70_10_20", args.segment_seed)
    outputs.append(segment_path)
    for seed in args.recording_seeds:
        path = args.output_dir / f"recording_80_20_seed{seed}.csv"
        write_manifest(path, rows, recording_split(rows, seed), "recording_80_20", seed)
        outputs.append(path)
    metadata = {
        "n_samples": len(rows),
        "class_counts": {c: sum(row["class_id"] == c for row in rows) for c in CATEGORIES},
        "source_listing_sha256": hashlib.sha256("\n".join(row["sample_id"] for row in rows).encode()).hexdigest(),
        "outputs": [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in outputs],
    }
    (args.output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
