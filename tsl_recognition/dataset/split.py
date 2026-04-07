"""
Data split generation and loading for signer-independent, random, or
predefined partitions.

Generates persistent JSON manifest files in the dataset's ``split_dir``
so that every training / validation / test run uses exactly the same
samples.

Three modes are supported:

``signer``
    Assigns entire signers to a single partition (train / val / test).
    Mapping is provided by the dataset's ``signer_split_map()``.

``random``
    Stratified random split (by class label) into train / val / test.

``predefined``
    Uses the splits shipped with the dataset (e.g. AUTSL). Requires
    processed ``.npy`` files to exist in the dataset's ``processed_dir``.

Usage::

    python -m tsl_recognition split
    python -m tsl_recognition split --split-mode random
    python -m tsl_recognition split --dataset autsl
    python -m tsl_recognition split --dataset autsl --split-mode random
"""

from __future__ import annotations

import datetime
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ..config import TrainConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _scan_all_files(
    cfg: TrainConfig,
    data_path: Path | None = None,
) -> tuple[list[dict], int]:
    """Scan the processed directory and return metadata for every valid file."""
    ds_info = cfg.dataset_info
    data_path = data_path or ds_info.processed_dir
    actions = cfg.actions
    label_map = {name: idx for idx, name in enumerate(actions)}

    entries: list[dict] = []
    expected_dim: int | None = None
    skipped = 0

    for action in tqdm(actions, desc="Scanning classes"):
        action_dir = data_path / action
        if not action_dir.exists():
            continue
        for npy_path in sorted(action_dir.glob("*.npy")):
            try:
                kp = np.load(npy_path, mmap_mode="r")
                if kp.ndim != 2:
                    skipped += 1
                    continue
                num_frames, num_features = kp.shape
                if num_frames < cfg.min_sequence_length:
                    skipped += 1
                    continue
                if expected_dim is None:
                    expected_dim = num_features
                elif num_features != expected_dim:
                    skipped += 1
                    continue

                signer = ds_info.extract_signer(npy_path.name) or "unknown"

                entries.append({
                    "path": str(npy_path),
                    "label": label_map[action],
                    "signer": signer,
                    "num_frames": int(num_frames),
                    "class_name": action,
                })
            except Exception as e:
                print(f"Error scanning {npy_path}: {e}")
                skipped += 1

    if not entries:
        raise RuntimeError(
            "No valid sequences found. Run keypoint extraction first."
        )
    assert expected_dim is not None

    print(f"\nScanned {len(entries)} valid files, skipped {skipped}")
    print(f"Feature dimension: {expected_dim}")
    return entries, expected_dim


# ---------------------------------------------------------------------------
# Signer-independent split
# ---------------------------------------------------------------------------
def _split_by_signer(
    entries: list[dict],
    signer_map: dict[str, list[str]],
) -> dict[str, list[dict]]:
    """Partition *entries* by signer identity.

    Parameters
    ----------
    entries : list[dict]
        Output of ``_scan_all_files``.
    signer_map : dict
        ``{"train": [...], "val": [...], "test": [...]}`` with signer IDs.
        Obtained from ``dataset_info.signer_split_map()``.
    """
    signer_to_partition: dict[str, str] = {}
    for partition, signers in signer_map.items():
        for s in signers:
            signer_to_partition[s] = partition

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for entry in entries:
        partition = signer_to_partition.get(entry["signer"])
        if partition is None:
            raise ValueError(
                f"Signer '{entry['signer']}' not in signer_split_map. "
                "Update the dataset's signer_split_map() to include this signer."
            )
        splits[partition].append(entry)

    return splits


# ---------------------------------------------------------------------------
# Stratified random split
# ---------------------------------------------------------------------------
def _split_random(
    entries: list[dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Stratified random split maintaining class proportions."""
    rng = np.random.RandomState(seed)

    by_class: dict[int, list[dict]] = defaultdict(list)
    for entry in entries:
        by_class[entry["label"]].append(entry)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for label in sorted(by_class):
        items = by_class[label]
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        if n_test + n_val >= n:
            n_test = max(1, n // 3)
            n_val = max(1, n // 3)

        splits["test"].extend(items[:n_test])
        splits["val"].extend(items[n_test : n_test + n_val])
        splits["train"].extend(items[n_test + n_val :])

    return splits


# ---------------------------------------------------------------------------
# Predefined splits
# ---------------------------------------------------------------------------
def _split_predefined(
    cfg: TrainConfig,
) -> tuple[dict[str, list[dict]], int]:
    """Build split entries from a dataset's predefined partitions."""
    ds_info = cfg.dataset_info
    raw_splits = ds_info.predefined_split_entries()
    if raw_splits is None:
        raise RuntimeError(
            f"Dataset {ds_info.name!r} does not provide predefined splits. "
            "Use --split-mode signer or --split-mode random instead."
        )

    actions = cfg.actions
    label_map = {name: idx for idx, name in enumerate(actions)}
    allowed = set(actions.tolist())

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    expected_dim: int | None = None
    skipped = 0
    missing_npy = 0

    for partition in ("train", "val", "test"):
        entries = raw_splits[partition]
        for entry in tqdm(entries, desc=f"Scanning {partition}"):
            class_name = entry["class_name"]
            if class_name not in allowed:
                skipped += 1
                continue

            npy_path = ds_info.output_npy_path(class_name, entry["sample_id"])
            if not npy_path.exists():
                missing_npy += 1
                continue

            try:
                kp = np.load(npy_path, mmap_mode="r")
                if kp.ndim != 2:
                    skipped += 1
                    continue
                num_frames, num_features = kp.shape
                if num_frames < cfg.min_sequence_length:
                    skipped += 1
                    continue
                if expected_dim is None:
                    expected_dim = num_features
                elif num_features != expected_dim:
                    skipped += 1
                    continue
            except Exception as e:
                print(f"Error loading {npy_path}: {e}")
                skipped += 1
                continue

            splits[partition].append({
                "path": str(npy_path),
                "label": label_map[class_name],
                "signer": entry.get("signer", "unknown"),
                "num_frames": int(num_frames),
                "class_name": class_name,
            })

    total = sum(len(v) for v in splits.values())
    if total == 0:
        raise RuntimeError(
            f"No valid .npy files found in {ds_info.processed_dir}. "
            "Run keypoint extraction first."
        )
    assert expected_dim is not None

    print(f"\nPredefined split: {total} valid files, "
          f"skipped {skipped}, missing .npy {missing_npy}")
    print(f"Feature dimension: {expected_dim}")
    return splits, expected_dim


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _write_manifests(
    splits: dict[str, list[dict]],
    feature_dim: int,
    mode: str,
    split_dir: Path,
) -> Path:
    """Write train/val/test JSON manifests and a metadata file."""
    split_dir.mkdir(parents=True, exist_ok=True)

    for partition in ("train", "val", "test"):
        manifest_path = split_dir / f"{partition}.json"
        with open(manifest_path, "w") as f:
            json.dump(splits[partition], f, indent=2)
        print(f"  {partition}: {len(splits[partition]):>6} samples -> {manifest_path.name}")

    all_classes_train = set(e["class_name"] for e in splits["train"])
    all_classes_val = set(e["class_name"] for e in splits["val"])
    all_classes_test = set(e["class_name"] for e in splits["test"])
    all_classes = all_classes_train | all_classes_val | all_classes_test

    signers_train = sorted(set(e.get("signer", "unknown") for e in splits["train"]))
    signers_val = sorted(set(e.get("signer", "unknown") for e in splits["val"]))
    signers_test = sorted(set(e.get("signer", "unknown") for e in splits["test"]))

    meta = {
        "mode": mode,
        "created": datetime.datetime.now().isoformat(),
        "feature_dim": feature_dim,
        "total_samples": sum(len(v) for v in splits.values()),
        "train_samples": len(splits["train"]),
        "val_samples": len(splits["val"]),
        "test_samples": len(splits["test"]),
        "total_classes": len(all_classes),
        "train_classes": len(all_classes_train),
        "val_classes": len(all_classes_val),
        "test_classes": len(all_classes_test),
        "val_missing_classes": sorted(all_classes - all_classes_val),
        "test_missing_classes": sorted(all_classes - all_classes_test),
        "signers": {
            "train": signers_train,
            "val": signers_val,
            "test": signers_test,
        },
    }

    meta_path = split_dir / "split_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return split_dir


def load_split(split_dir: Path) -> dict[str, Any]:
    """Load persisted split manifests.

    Returns
    -------
    dict
        ``train_files``, ``val_files``, ``test_files`` -- each a list of
        ``(path_str, label_int, num_frames)`` tuples compatible with
        ``LazySignDataset``. Also includes ``feature_dim`` and ``mode``.

    Raises
    ------
    FileNotFoundError
        If no split manifests exist yet (run ``generate_split`` first).
    """
    meta_path = split_dir / "split_meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"No split manifests found in {split_dir}. "
            "Run `python -m tsl_recognition split` first."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    result: dict = {
        "feature_dim": meta["feature_dim"],
        "mode": meta["mode"],
    }

    for partition in ("train", "val", "test"):
        manifest_path = split_dir / f"{partition}.json"
        with open(manifest_path) as f:
            entries = json.load(f)
        result[f"{partition}_files"] = [
            (e["path"], e["label"], e["num_frames"])
            for e in entries
        ]

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_split(
    cfg: TrainConfig,
    data_path: Path | None = None,
    split_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate and persist a train/val/test split.

    Parameters
    ----------
    cfg : TrainConfig
        Configuration (controls split_mode, val_split, test_split, etc.).
    data_path : Path, optional
        Override for the processed data directory.
    split_dir : Path, optional
        Override for the split output directory.

    Returns
    -------
    dict
        Same format as ``load_split`` output.
    """
    ds_info = cfg.dataset_info
    split_dir = split_dir or ds_info.split_dir
    mode = cfg.split_mode

    print(f"\n{'='*60}")
    print(f"Generating {mode.upper()} split for {ds_info.display_name}")
    print(f"{'='*60}\n")

    if mode == "predefined":
        splits, feature_dim = _split_predefined(cfg)
        all_cls: set[str] = set()
        for part_entries in splits.values():
            all_cls.update(e["class_name"] for e in part_entries)
    else:
        entries, feature_dim = _scan_all_files(cfg, data_path)
        all_cls = set(e["class_name"] for e in entries)

        if mode == "signer":
            signer_map = ds_info.signer_split_map()
            if signer_map is None:
                raise RuntimeError(
                    f"Dataset {ds_info.name!r} does not provide a signer split map. "
                    "Use --split-mode random or --split-mode predefined instead."
                )
            splits = _split_by_signer(entries, signer_map)
        elif mode == "random":
            splits = _split_random(
                entries,
                val_ratio=cfg.val_split,
                test_ratio=cfg.test_split,
            )
        else:
            raise ValueError(
                f"Unknown split_mode: {mode!r}. "
                "Use 'signer', 'random', or 'predefined'."
            )

    total = sum(len(v) for v in splits.values())
    print(f"\nSplit summary ({mode} mode):")
    out_dir = _write_manifests(splits, feature_dim, mode, split_dir)

    for part in ("train", "val", "test"):
        part_cls = set(e["class_name"] for e in splits[part])
        missing = len(all_cls) - len(part_cls)
        coverage = f"{len(part_cls)}/{len(all_cls)}"
        pct = len(splits[part]) / total * 100 if total else 0
        extra = f" ({missing} classes missing)" if missing else ""
        print(f"  {part:>5}: {len(splits[part]):>6} samples ({pct:5.1f}%), "
              f"classes: {coverage}{extra}")

    has_signers = any(
        e.get("signer") not in (None, "unknown")
        for part in splits.values()
        for e in part
    )
    if has_signers:
        for part in ("train", "val", "test"):
            signers = sorted(set(
                e["signer"] for e in splits[part]
                if e.get("signer") not in (None, "unknown")
            ))
            if signers:
                display = ", ".join(signers) if len(signers) <= 10 else f"{len(signers)} signers"
                print(f"  {part:>5} signers: {display}")

    print(f"\nManifests saved to: {out_dir}")

    return load_split(split_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from .registry import DATASET_CHOICES

    parser = argparse.ArgumentParser(description="Generate data split manifests")
    parser.add_argument(
        "--dataset", choices=DATASET_CHOICES, default="bosphorus",
    )
    parser.add_argument(
        "--split-mode", choices=["signer", "random", "predefined"], default=None,
    )
    parser.add_argument("--test", action="store_true", help="Use 10-class test config")
    args = parser.parse_args()

    config = TrainConfig.test(dataset=args.dataset) if args.test else TrainConfig.full(dataset=args.dataset)
    config.split_mode = args.split_mode or config.dataset_info.default_split_mode
    generate_split(config)
