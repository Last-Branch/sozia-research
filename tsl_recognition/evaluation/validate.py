"""
Validate that the inference preprocessing pipeline matches training.

Loads raw .npy keypoint files from the persisted **test** split and compares
predictions using:
  1. Training pipeline (LazySignDataset-style)
  2. Inference pipeline (preprocess_sequence)

If both produce the same predictions the inference pipeline is correct.

Usage::

    python -m tsl_recognition validate
    python -m tsl_recognition validate --samples 200
    python -m tsl_recognition validate --run-dir models/recognition/AUTSL_run_20260413_224156_gru
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import DEVICE, MODELS_DIR, TrainConfig
from ..dataset.loader import LazySignDataset
from .inference import _find_latest_run, _load_run, preprocess_sequence


def _load_test_files(
    split_dir: Path,
    actions: np.ndarray,
) -> list[tuple[str, int, int]]:
    """Load test-split file list from the persisted manifest.

    Parameters
    ----------
    split_dir : Path
        Directory containing the split JSON manifests.
    actions : np.ndarray
        Ordered class names (the label map is derived from the index).

    Returns
    -------
    list of (path, label_int, num_frames)
    """
    manifest = split_dir / "test.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"No test manifest at {manifest}. Run `python -m tsl_recognition split` first."
        )
    with open(manifest) as f:
        entries = json.load(f)

    label_map = {name: idx for idx, name in enumerate(actions)}
    allowed = set(label_map)
    files: list[tuple[str, int, int]] = []
    dropped = 0
    for e in entries:
        cls = e.get("class_name")
        if cls not in allowed:
            dropped += 1
            continue
        files.append((e["path"], label_map[cls], int(e["num_frames"])))
    if dropped:
        print(f"Filtered test split: dropped {dropped} samples outside run classes")
    return files


def run_validation(
    run_dir: Path | str | None = None,
    n_samples: int = 100,
    dataset: str = "bosphorus",
) -> dict:
    """Compare training vs inference preprocessing on *n_samples* test files.

    Uses the persisted test split to ensure the same held-out samples are
    evaluated every time.

    Parameters
    ----------
    run_dir : Path | str | None
        Path to a ``run_*`` directory produced by ``train.py``.  If *None*,
        the most recent run directory under ``MODELS_DIR`` is used.
    n_samples : int
        Number of test-split samples to compare.
    dataset : str
        Dataset name used to locate the correct split directory.

    Returns
    -------
    dict
        Accuracy and match statistics.
    """
    rd = Path(run_dir) if run_dir is not None else _find_latest_run(MODELS_DIR)
    run_data = _load_run(rd)
    model = run_data["model"]
    scaler = run_data["scaler"]
    actions = run_data["actions"]
    max_len = run_data["max_len"]
    seq_handling = run_data["sequence_handling"]
    normalize = run_data["normalize"]
    feature_dim = run_data["feature_dim"]

    split_dir = TrainConfig(dataset=dataset).dataset_info.split_dir
    test_files = _load_test_files(split_dir, actions)
    if not test_files:
        raise RuntimeError(
            "No test files found in split manifest matching run classes."
        )

    train_baseline_ds = LazySignDataset(
        file_info_list=test_files,
        max_seq_len=max_len,
        feature_dim=feature_dim,
        scaler=scaler,
        augment=False,
        sequence_handling=seq_handling,
    )

    random.seed(42)
    indices = list(range(len(test_files)))
    sampled = random.sample(indices, min(n_samples, len(indices)))
    print(f"\nValidating inference pipeline on {len(sampled)} test-split samples...")
    print(f"Run directory: {rd.name}\n")

    results = dict(
        training_correct=0, inference_correct=0, both_match=0, both_correct=0
    )
    mismatches: list[dict] = []

    for idx in tqdm(sampled, desc="Validating"):
        path, label, _ = test_files[idx]
        true_cls = actions[label]

        tkp, _lbl, t_len = train_baseline_ds[idx]
        with torch.no_grad():
            t_logits = model(
                tkp.unsqueeze(0).to(DEVICE),
                lengths=t_len.unsqueeze(0).to(DEVICE),
            )
            train_cls = actions[torch.argmax(t_logits, dim=1).item()]

        raw = np.load(path).astype(np.float32)
        frames = [raw[i] for i in range(len(raw))]
        ikp, actual_len = preprocess_sequence(
            frames,
            scaler,
            max_len,
            normalize=normalize,
            apply_interpolation=True,
            sequence_handling=seq_handling,
        )
        with torch.no_grad():
            i_logits = model(
                torch.tensor(ikp, dtype=torch.float32).unsqueeze(0).to(DEVICE),
                lengths=torch.tensor([actual_len], dtype=torch.long).to(DEVICE),
            )
            inf_cls = actions[torch.argmax(i_logits, dim=1).item()]

        if train_cls == true_cls:
            results["training_correct"] += 1
        if inf_cls == true_cls:
            results["inference_correct"] += 1
        if train_cls == inf_cls:
            results["both_match"] += 1
        if train_cls == true_cls and inf_cls == true_cls:
            results["both_correct"] += 1
        if train_cls != inf_cls:
            mismatches.append(
                dict(file=path, true=true_cls, train=train_cls, inf=inf_cls)
            )

    n = len(sampled)
    print(f"\n{'=' * 60}\nVALIDATION RESULTS\n{'=' * 60}")
    print(f"Samples tested: {n}\n")
    print(
        f"Training pipeline accuracy:  {results['training_correct'] / n * 100:.1f}% ({results['training_correct']}/{n})"
    )
    print(
        f"Inference pipeline accuracy: {results['inference_correct'] / n * 100:.1f}% ({results['inference_correct']}/{n})\n"
    )
    print(
        f"Predictions match:           {results['both_match'] / n * 100:.1f}% ({results['both_match']}/{n})"
    )
    print(
        f"Both correct:                {results['both_correct'] / n * 100:.1f}% ({results['both_correct']}/{n})"
    )

    if mismatches:
        print(f"\nMismatches ({len(mismatches)} samples):")
        for m in mismatches[:10]:
            print(f"  True: {m['true']}, Training: {m['train']}, Inference: {m['inf']}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
    else:
        print(
            "\nAll predictions match! Inference pipeline is correctly aligned with training."
        )

    print(f"\n{'-' * 60}")
    if results["both_match"] == n:
        print("PASS: Inference preprocessing produces identical results to training.")
    elif results["both_match"] / n > 0.95:
        print("MOSTLY PASS: Minor differences likely due to interpolation.")
    else:
        print("WARNING: Significant mismatch between training and inference pipelines.")
    return results
