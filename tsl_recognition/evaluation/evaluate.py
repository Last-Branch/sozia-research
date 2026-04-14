"""
Re-evaluate a saved model on the held-out test set.

Loads a trained model from any ``run_*`` directory and computes top-1 and
top-5 accuracy, per-class metrics, and a confusion matrix — without
retraining.  The updated ``scores.json`` is written back to the run
directory.

Usage::

    python -m tsl_recognition evaluate                                # latest run
    python -m tsl_recognition evaluate --run-dir models/recognition/AUTSL_run_* # specific run
    python -m tsl_recognition evaluate --all                          # all runs
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from torch.utils.data import DataLoader

from ..config import DEVICE, MODELS_DIR, TrainConfig
from ..dataset.loader import LazySignDataset
from .inference import _find_latest_run, _load_run
from .train import _save_scores, batch_top_k_accuracy


def _load_test_files(
    split_dir: Path,
    actions: np.ndarray,
) -> list[tuple[str, int, int]]:
    """Load test-split file list from the persisted manifest."""
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
    for e in entries:
        cls = e.get("class_name")
        if cls not in allowed:
            continue
        files.append((e["path"], label_map[cls], int(e["num_frames"])))
    return files


def evaluate_run(
    run_dir: Path | str | None = None,
    dataset: str = "bosphorus",
) -> dict:
    """Re-evaluate a saved model on the full test set.

    Parameters
    ----------
    run_dir : Path | str | None
        Path to a ``run_*`` directory produced by ``train.py``.  If *None*,
        the most recent run directory under ``MODELS_DIR`` is used.
    dataset : str
        Dataset name used to locate the split directory.

    Returns
    -------
    dict
        ``test_accuracy``, ``test_top5_accuracy``, ``best_val_accuracy``,
        ``best_val_top5_accuracy``, ``num_classes``, ``num_test_samples``.
    """
    rd = Path(run_dir) if run_dir is not None else _find_latest_run(MODELS_DIR)
    run_data = _load_run(rd)
    model = run_data["model"]
    scaler = run_data["scaler"]
    actions = run_data["actions"]
    max_len = run_data["max_len"]
    seq_handling = run_data["sequence_handling"]
    feature_dim = run_data["feature_dim"]
    num_classes = len(actions)

    scores_path = rd / "scores.json"
    best_val_acc = 0.0
    best_val_acc5 = 0.0
    if scores_path.exists():
        with open(scores_path) as f:
            old_scores = json.load(f)
        best_val_acc = old_scores.get("best_val_accuracy", 0.0)
        best_val_acc5 = old_scores.get("best_val_top5_accuracy", 0.0)

    ds_info = TrainConfig(dataset=dataset).dataset_info
    test_files = _load_test_files(ds_info.split_dir, actions)
    if not test_files:
        raise RuntimeError(
            "No test files found in split manifest matching run classes."
        )

    test_ds = LazySignDataset(
        file_info_list=test_files,
        max_seq_len=max_len,
        feature_dim=feature_dim,
        scaler=scaler,
        augment=False,
        sequence_handling=seq_handling,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"\nEvaluating on {len(test_files)} test samples...")
    print(f"{'=' * 60}")

    all_logits_list, all_true = [], []
    with torch.no_grad():
        for xb, yb, lengths in test_loader:
            logits = model(xb.to(DEVICE), lengths=lengths.to(DEVICE))
            all_logits_list.append(logits.cpu())
            all_true.extend(yb.numpy().tolist())

    all_logits = torch.cat(all_logits_list, dim=0)
    all_targets = torch.tensor(all_true, dtype=torch.long)
    all_preds = torch.argmax(all_logits, dim=1).numpy().tolist()

    all_labels = list(range(num_classes))
    test_acc = accuracy_score(all_true, all_preds)
    test_acc5 = batch_top_k_accuracy(all_logits, all_targets, k=5)
    cm = multilabel_confusion_matrix(all_true, all_preds, labels=all_labels)

    print(f"Test Accuracy (top-1): {test_acc:.4f}")
    print(f"Test Accuracy (top-5): {test_acc5:.4f}")
    print(f"Test samples:          {len(test_files)}")
    print(f"Num classes:           {num_classes}")

    _save_scores(rd, test_acc, test_acc5, best_val_acc, best_val_acc5, cm, actions)
    print(f"\nscores.json updated in: {rd}")

    return {
        "run_dir": rd,
        "test_accuracy": test_acc,
        "test_top5_accuracy": test_acc5,
        "best_val_accuracy": best_val_acc,
        "best_val_top5_accuracy": best_val_acc5,
        "num_classes": num_classes,
        "num_test_samples": len(test_files),
    }


def evaluate_all(dataset: str = "bosphorus") -> list[dict]:
    """Re-evaluate every ``run_*`` directory under ``MODELS_DIR``.

    Returns
    -------
    list[dict]
        One result dict per run (same format as ``evaluate_run``).
    """
    runs = sorted(MODELS_DIR.glob("run_*"))
    if not runs:
        raise FileNotFoundError(
            f"No run directories found in {MODELS_DIR}. Train a model first."
        )

    results = []
    for rd in runs:
        print(f"\n{'=' * 60}")
        print(f"RE-EVALUATING: {rd.name}")
        print(f"{'=' * 60}")
        try:
            result = evaluate_run(rd, dataset=dataset)
            results.append(result)
        except Exception as e:
            print(f"  SKIPPED ({e})")
            continue

    if results:
        print(f"\n\n{'=' * 80}")
        print("RE-EVALUATION SUMMARY")
        print(f"{'=' * 80}")
        print(f"{'Run':<45} {'Top-1':>8} {'Top-5':>8} {'Classes':>8}")
        print(f"{'-' * 45} {'-' * 8} {'-' * 8} {'-' * 8}")
        for r in results:
            name = r["run_dir"].name
            print(
                f"{name:<45} "
                f"{r['test_accuracy']:>7.4f} "
                f"{r['test_top5_accuracy']:>7.4f} "
                f"{r['num_classes']:>8}"
            )
    return results
