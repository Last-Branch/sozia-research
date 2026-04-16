"""
Training loop for the lip-reading word classifier.

Architecture: ActionGRU (small preset) with input_size=249.
Labels: sign class names (Turkish words) — one word label per sequence.
Supervision comes from the existing TSL recognition datasets: signers mouth
the words they sign, providing natural weak supervision for lip motion.

Artifacts per run::

    models/lip_reading/<dataset>_run_<timestamp>_gru/
        config.json          # hyper-parameters + metadata
        vocab.json           # index → class-name mapping (for server inference)
        best_model.pt        # best checkpoint by val accuracy
        final_model.pt       # weights at the last training epoch
        checkpoints/         # periodic snapshots every 50 epochs
        scores.json          # final test & val metrics + per-class breakdown
        training_log.csv     # per-epoch metrics
        plots/
            loss_curve.png
            accuracy_curve.png
"""

from __future__ import annotations

import csv
import datetime
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tsl_recognition.config import SEED, set_seed
from tsl_recognition.evaluation.train import (
    batch_accuracy,
    batch_top_k_accuracy,
    evaluate,
)
from tsl_recognition.models import build_model

from .config import DEVICE, LIP_FEATURE_DIM, LIP_MODELS_DIR, LipTrainConfig
from .dataset import build_lip_loaders

CHECKPOINT_EVERY = 50


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------


def _build_scheduler(
    cfg: LipTrainConfig,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
) -> lrs.LRScheduler | None:
    """Build the LR scheduler matching ``cfg.lr_scheduler``."""
    name = (cfg.lr_scheduler or "none").lower()

    if name in {"none", "off", "false", "no"}:
        return None

    if name in {"plateau", "reduceonplateau", "reducelronplateau"}:
        return lrs.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

    if name in {"onecycle", "onecyclelr"}:
        steps_per_epoch = len(train_loader)
        warmup_epochs = max(int(cfg.warmup_epochs), 0)
        pct_start = float(min(max(warmup_epochs / max(int(cfg.epochs), 1), 1e-3), 0.5))
        if cfg.onecycle_max_lr is None:
            cfg.onecycle_max_lr = float(min(5e-3, cfg.learning_rate * 5.0))
        return lrs.OneCycleLR(
            optimizer,
            max_lr=float(cfg.onecycle_max_lr),
            epochs=cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            div_factor=cfg.onecycle_div_factor,
            final_div_factor=cfg.onecycle_final_div_factor,
            anneal_strategy="cos",
        )

    if name in {"cosine", "cosine_warm_restarts", "cosinewarmrestarts"}:
        cosine = lrs.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(cfg.cosine_t0),
            T_mult=int(cfg.cosine_t_mult),
            eta_min=float(cfg.cosine_eta_min),
        )
        warmup_epochs = max(int(cfg.warmup_epochs), 0)
        if warmup_epochs > 0:
            warmup = lrs.LinearLR(
                optimizer,
                start_factor=float(cfg.warmup_start_factor),
                total_iters=warmup_epochs,
            )
            return lrs.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        return cosine

    raise ValueError(
        f"Unknown lr_scheduler={cfg.lr_scheduler!r}. "
        "Expected one of: onecycle, cosine_warm_restarts, plateau, none."
    )


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


def _create_run_dir(cfg: LipTrainConfig, timestamp: str) -> Path:
    """Create and return the per-run output directory."""
    run_dir = LIP_MODELS_DIR / f"{cfg.dataset_info.display_name}_run_{timestamp}_gru"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    return run_dir


def _save_vocab(run_dir: Path, classes: list[str]) -> None:
    """Write index → class-name mapping to ``vocab.json``.

    The server's ``LipReadingEngine`` loads this file to convert classifier
    output indices back to Turkish word strings.
    """
    vocab = {i: name for i, name in enumerate(classes)}
    (run_dir / "vocab.json").write_text(json.dumps(vocab, indent=2, ensure_ascii=False))


def _save_config(run_dir: Path, cfg: LipTrainConfig, total_params: int) -> None:
    """Serialize hyper-parameters and run metadata to ``config.json``."""
    meta = asdict(cfg)
    meta.update(
        {
            "model_arch": "gru",
            "model_size": "small",
            "feature_dim": LIP_FEATURE_DIM,
            "total_params": total_params,
            "device": str(DEVICE),
        }
    )
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))


def _save_training_log(run_dir: Path, log_rows: list[dict]) -> None:
    """Write per-epoch metrics to ``training_log.csv``."""
    if not log_rows:
        return
    path = run_dir / "training_log.csv"
    fieldnames = list(log_rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)


def _save_plots(run_dir: Path, log_rows: list[dict]) -> None:
    """Generate loss and accuracy curve PNGs from the training log."""
    if not log_rows:
        return

    epochs = [r["epoch"] for r in log_rows]
    train_loss = [r["train_loss"] for r in log_rows]
    train_acc = [r["train_acc"] for r in log_rows]
    val_loss = [r["val_loss"] for r in log_rows if r["val_loss"] is not None]
    val_acc = [r["val_acc"] for r in log_rows if r["val_acc"] is not None]
    val_epochs = [r["epoch"] for r in log_rows if r["val_loss"] is not None]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, label="Train loss")
    if val_loss:
        ax.plot(val_epochs, val_loss, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve — Lip Reading")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "loss_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_acc, label="Train top-1")
    if val_acc:
        ax.plot(val_epochs, val_acc, label="Val top-1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curve — Lip Reading")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "accuracy_curve.png", dpi=150)
    plt.close(fig)


def _save_scores(
    run_dir: Path,
    test_acc: float,
    test_acc5: float,
    best_val_acc: float,
    best_val_acc5: float,
    cm: np.ndarray,
    actions: list[str],
) -> None:
    """Write final evaluation metrics to ``scores.json``."""
    per_class: dict[str, dict] = {}
    macro_precision, macro_recall, macro_f1 = 0.0, 0.0, 0.0
    n_with_support = 0

    for i, label in enumerate(actions):
        tn, fp, fn, tp = cm[i].ravel()
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        support = int(tp + fn)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        if support > 0:
            macro_precision += precision
            macro_recall += recall
            macro_f1 += f1
            n_with_support += 1

    n = max(n_with_support, 1)
    scores = {
        "test_accuracy": test_acc,
        "test_top5_accuracy": test_acc5,
        "best_val_accuracy": best_val_acc,
        "best_val_top5_accuracy": best_val_acc5,
        "num_classes": len(actions),
        "macro_precision": round(macro_precision / n, 4),
        "macro_recall": round(macro_recall / n, 4),
        "macro_f1": round(macro_f1 / n, 4),
        "per_class": per_class,
    }
    (run_dir / "scores.json").write_text(
        json.dumps(scores, indent=2, ensure_ascii=False)
    )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train(cfg: LipTrainConfig | None = None) -> dict:
    """Train the lip-reading word classifier end-to-end.

    The training loop validates against the **validation** set (for LR
    scheduling, early stopping, best-model checkpointing).  The **test** set
    is used only once at the very end for the final reported accuracy.

    All artifacts are written to a dedicated run directory under
    ``models/lip_reading/``.

    Parameters
    ----------
    cfg : LipTrainConfig, optional
        Training configuration. If *None*, uses ``LipTrainConfig.full()``.

    Returns
    -------
    dict
        ``model``, ``run_dir``, ``model_path``, ``test_accuracy``,
        ``test_top5_accuracy``, ``best_val_acc``, ``actions``, ``vocab``.
    """
    if cfg is None:
        cfg = LipTrainConfig.full()

    set_seed(SEED)

    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _create_run_dir(cfg, run_stamp)
    print(f"\nRun directory: {run_dir}")

    data = build_lip_loaders(cfg)
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    class_weights = data["class_weights"]
    actions = cfg.classes_to_process

    print(
        f"\nArchitecture: GRU (small) | "
        f"input_size={LIP_FEATURE_DIM} | "
        f"classes={cfg.num_classes}"
    )
    print(f"Dropout: {cfg.dropout}")

    model = build_model(
        arch="gru",
        input_size=LIP_FEATURE_DIM,
        num_classes=cfg.num_classes,
        model_size="small",
        dropout=cfg.dropout,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    _save_config(run_dir, cfg, total_params)
    _save_vocab(run_dir, actions)

    ls = cfg.label_smoothing if cfg.label_smoothing > 0 else 0.0
    if cfg.use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ls)
        print("Using weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=ls)
    if ls > 0:
        print(f"Label smoothing: {ls}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4
    )
    scheduler = _build_scheduler(cfg, optimizer, train_loader)

    if scheduler is None:
        print("LR scheduler: none")
    else:
        print(f"LR scheduler: {cfg.lr_scheduler}")
    if cfg.grad_clip_norm > 0:
        print(f"Gradient clipping: max_norm={cfg.grad_clip_norm}")
    if cfg.early_stopping_patience > 0:
        print(
            f"Early stopping: patience={cfg.early_stopping_patience} "
            f"(min_epochs={cfg.min_epochs})"
        )

    best_val_acc = 0.0
    best_val_acc5 = 0.0
    best_val_epoch = 0
    log_rows: list[dict] = []
    best_model_path = run_dir / "best_model.pt"

    print(f"\nTraining for {cfg.epochs} epochs (lr={cfg.learning_rate})")
    print("-" * 60)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        running_acc5 = 0.0

        for xb, yb, lengths in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb, lengths=lengths)
            loss = criterion(logits, yb)
            loss.backward()
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            if isinstance(scheduler, lrs.OneCycleLR):
                scheduler.step()
            running_loss += loss.item()
            detached = logits.detach()
            running_acc += batch_accuracy(detached, yb)
            running_acc5 += batch_top_k_accuracy(detached, yb, k=5)

        train_loss = running_loss / len(train_loader)
        train_acc = running_acc / len(train_loader)
        train_acc5 = running_acc5 / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        val_loss: float | None = None
        val_acc: float | None = None
        val_acc5: float | None = None

        if epoch == 1 or epoch % cfg.val_every == 0 or epoch == cfg.epochs:
            val_loss, val_acc, val_acc5 = evaluate(model, val_loader, criterion)
            if isinstance(scheduler, lrs.ReduceLROnPlateau):
                scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_acc5 = val_acc5
                best_val_epoch = epoch
                best_marker = " * (saved)"
                torch.save(model.state_dict(), best_model_path)
            else:
                best_marker = ""

            print(
                f"Epoch {epoch:3d}/{cfg.epochs} | "
                f"loss {train_loss:.4f} | "
                f"train {train_acc:.4f} (top5 {train_acc5:.4f}) | "
                f"val {val_acc:.4f} (top5 {val_acc5:.4f}) | "
                f"lr {current_lr:.1e}{best_marker}"
            )

        if scheduler is not None and not isinstance(
            scheduler, (lrs.OneCycleLR, lrs.ReduceLROnPlateau)
        ):
            scheduler.step()

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "train_acc5": round(train_acc5, 6),
                "val_loss": round(val_loss, 6) if val_loss is not None else None,
                "val_acc": round(val_acc, 6) if val_acc is not None else None,
                "val_acc5": round(val_acc5, 6) if val_acc5 is not None else None,
                "lr": current_lr,
            }
        )

        if epoch % CHECKPOINT_EVERY == 0:
            ckpt_path = run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)

        if (
            cfg.early_stopping_patience > 0
            and best_val_epoch > 0
            and epoch >= cfg.min_epochs
            and (epoch - best_val_epoch) >= cfg.early_stopping_patience
        ):
            print(
                f"\nEarly stopping at epoch {epoch}: "
                f"no improvement for {cfg.early_stopping_patience} epochs "
                f"(best val_acc={best_val_acc:.4f} at epoch {best_val_epoch})"
            )
            break

    print("-" * 60)
    print(
        f"Best validation accuracy: "
        f"{best_val_acc:.4f} top-1, {best_val_acc5:.4f} top-5 "
        f"(epoch {best_val_epoch})"
    )

    final_model_path = run_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")

    if best_model_path.exists():
        model.load_state_dict(
            torch.load(best_model_path, map_location=DEVICE, weights_only=True)
        )
        print("Restored best model weights")

    # ------------------------------------------------------------------
    # Final evaluation on held-out test set
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print(f"{'=' * 60}")

    eval_model = build_model(
        arch="gru",
        input_size=LIP_FEATURE_DIM,
        num_classes=cfg.num_classes,
        model_size="small",
        dropout=cfg.dropout,
    ).to(DEVICE)
    eval_model.load_state_dict(
        torch.load(best_model_path, map_location=DEVICE, weights_only=True)
    )
    eval_model.eval()

    all_logits_list: list[torch.Tensor] = []
    all_true: list[int] = []
    with torch.no_grad():
        for xb, yb, lengths in test_loader:
            logits = eval_model(xb.to(DEVICE), lengths=lengths.to(DEVICE))
            all_logits_list.append(logits.cpu())
            all_true.extend(yb.numpy().tolist())

    all_logits = torch.cat(all_logits_list, dim=0)
    all_targets = torch.tensor(all_true, dtype=torch.long)
    all_preds = torch.argmax(all_logits, dim=1).numpy().tolist()

    test_acc = accuracy_score(all_true, all_preds)
    test_acc5 = batch_top_k_accuracy(all_logits, all_targets, k=5)
    cm = multilabel_confusion_matrix(
        all_true, all_preds, labels=list(range(cfg.num_classes))
    )

    print(f"Test Accuracy (top-1): {test_acc:.4f}")
    print(f"Test Accuracy (top-5): {test_acc5:.4f}")

    _save_scores(run_dir, test_acc, test_acc5, best_val_acc, best_val_acc5, cm, actions)
    _save_training_log(run_dir, log_rows)
    _save_plots(run_dir, log_rows)
    print(f"\nAll artifacts saved to: {run_dir}")

    return {
        "model": eval_model,
        "run_dir": run_dir,
        "model_path": best_model_path,
        "test_accuracy": test_acc,
        "test_top5_accuracy": test_acc5,
        "best_val_acc": best_val_acc,
        "actions": actions,
        "vocab": {i: name for i, name in enumerate(actions)},
    }
