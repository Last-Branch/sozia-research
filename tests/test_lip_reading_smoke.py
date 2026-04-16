"""Smoke tests for the lip_reading pipeline.

These tests verify that modules import cleanly, core objects can be
constructed with no data on disk, and the 249-dim GRU variant produces
output of the expected shape.  No GPU, no real dataset files required.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# 1. Import smoke
# ---------------------------------------------------------------------------


def test_lip_reading_imports() -> None:
    """Top-level ``lip_reading`` package import must succeed."""
    import lip_reading  # noqa: F401


def test_lip_reading_submodules_import() -> None:
    """Core sub-modules must be importable without side-effects."""
    from lip_reading import config  # noqa: F401
    from lip_reading.dataset import LipReadingDataset, build_lip_loaders  # noqa: F401
    from lip_reading.train import train  # noqa: F401


# ---------------------------------------------------------------------------
# 2. LipTrainConfig defaults
# ---------------------------------------------------------------------------


def test_lip_train_config_defaults() -> None:
    """LipTrainConfig() must expose the documented default values."""
    from lip_reading.config import LipTrainConfig

    cfg = LipTrainConfig()

    assert cfg.dataset == "bosphorus"
    assert cfg.classes_to_process == []
    assert cfg.max_sequence_length == 150
    assert cfg.min_sequence_length == 10
    assert cfg.batch_size == 64
    assert cfg.epochs == 300
    assert cfg.lr_scheduler == "onecycle"
    assert cfg.dropout == pytest.approx(0.4)
    assert cfg.augment_train is True


def test_lip_train_config_num_classes() -> None:
    """num_classes property must equal len(classes_to_process)."""
    from lip_reading.config import LipTrainConfig

    cfg = LipTrainConfig(classes_to_process=["Aci", "Acik", "Bal"])
    assert cfg.num_classes == 3


def test_lip_train_config_test_factory() -> None:
    """LipTrainConfig.test() must use the reduced epoch count."""
    from lip_reading.config import LipTrainConfig

    cfg = LipTrainConfig.test()
    assert cfg.epochs == 50
    assert cfg.max_sequence_length == 100


# ---------------------------------------------------------------------------
# 3. Constants
# ---------------------------------------------------------------------------


def test_lip_feature_dim_is_249() -> None:
    """LIP_FEATURE_DIM must equal 249 (83 face landmarks × 3 coords)."""
    from lip_reading.config import LIP_FEATURE_DIM

    assert LIP_FEATURE_DIM == 249


def test_face_slice_selects_249_dims() -> None:
    """FACE_SLICE applied to a 507-dim vector must yield 249 elements."""
    from lip_reading.config import FACE_SLICE

    dummy = np.zeros(507)
    assert dummy[FACE_SLICE].shape == (249,)


def test_face_slice_matches_tsl_config() -> None:
    """lip_reading.FACE_SLICE must match tsl_recognition.config.FACE_SLICE."""
    from lip_reading.config import FACE_SLICE as lip_slice
    from tsl_recognition.config import FACE_SLICE as tsl_slice

    assert lip_slice == tsl_slice


# ---------------------------------------------------------------------------
# 4. LipReadingDataset with synthetic data
# ---------------------------------------------------------------------------


def _make_synthetic_npy(path: Path, n_frames: int = 20) -> None:
    """Write a 507-dim synthetic keypoint file at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.random.rand(n_frames, 507).astype(np.float32))


def test_lip_reading_dataset_length(tmp_path: Path) -> None:
    """LipReadingDataset.__len__ must match the file list length."""
    from lip_reading.dataset import LipReadingDataset

    p = tmp_path / "seq.npy"
    _make_synthetic_npy(p, n_frames=30)
    ds = LipReadingDataset([(str(p), 0, 30)], max_seq_len=50)
    assert len(ds) == 1


def test_lip_reading_dataset_output_shapes(tmp_path: Path) -> None:
    """LipReadingDataset must return tensors of the expected shapes."""
    from lip_reading.dataset import LipReadingDataset

    p = tmp_path / "seq.npy"
    _make_synthetic_npy(p, n_frames=30)
    ds = LipReadingDataset([(str(p), 2, 30)], max_seq_len=50)
    x, y, length = ds[0]

    assert x.shape == (50, 249), f"Expected (50, 249), got {x.shape}"
    assert y.item() == 2
    assert length.item() == 30


def test_lip_reading_dataset_truncates_long_sequence(tmp_path: Path) -> None:
    """Sequences longer than max_seq_len must be truncated to max_seq_len."""
    from lip_reading.dataset import LipReadingDataset

    p = tmp_path / "long.npy"
    _make_synthetic_npy(p, n_frames=200)
    ds = LipReadingDataset([(str(p), 0, 200)], max_seq_len=50)
    x, _, length = ds[0]

    assert x.shape == (50, 249)
    assert length.item() == 50


def test_lip_reading_dataset_pads_short_sequence(tmp_path: Path) -> None:
    """Sequences shorter than max_seq_len must be zero-padded."""
    from lip_reading.dataset import LipReadingDataset

    p = tmp_path / "short.npy"
    _make_synthetic_npy(p, n_frames=10)
    ds = LipReadingDataset([(str(p), 0, 10)], max_seq_len=50)
    x, _, length = ds[0]

    assert x.shape == (50, 249)
    assert length.item() == 10
    # Padding rows must be zero
    assert torch.all(x[10:] == 0.0)


def test_lip_reading_dataset_face_slice_applied(tmp_path: Path) -> None:
    """Features returned must be the face-landmark slice of the 507-dim vector."""
    from lip_reading.config import FACE_SLICE
    from lip_reading.dataset import LipReadingDataset

    p = tmp_path / "seq.npy"
    raw = np.random.rand(20, 507).astype(np.float32)
    np.save(p, raw)

    ds = LipReadingDataset([(str(p), 0, 20)], max_seq_len=20)
    x, _, _ = ds[0]

    expected = raw[:, FACE_SLICE]
    np.testing.assert_allclose(x.numpy(), expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# 5. GRU forward pass at 249-dim input
# ---------------------------------------------------------------------------


def test_gru_forward_pass_249_dim() -> None:
    """ActionGRU with input_size=249 must produce logits of shape (batch, classes)."""
    from tsl_recognition.models import build_model

    model = build_model(arch="gru", input_size=249, num_classes=10, model_size="small")
    model.eval()

    dummy = torch.zeros(2, 15, 249)
    with torch.no_grad():
        logits = model(dummy)

    assert logits.shape == (2, 10), f"Expected (2, 10), got {logits.shape}"


def test_gru_forward_pass_249_dim_is_finite() -> None:
    """GRU output on zero input must contain no NaN or Inf."""
    from tsl_recognition.models import build_model

    model = build_model(arch="gru", input_size=249, num_classes=5, model_size="small")
    model.eval()

    dummy = torch.zeros(2, 15, 249)
    with torch.no_grad():
        logits = model(dummy)

    assert torch.isfinite(logits).all(), "logits contain NaN or Inf"


# ---------------------------------------------------------------------------
# 6. vocab.json round-trip
# ---------------------------------------------------------------------------


def test_vocab_json_round_trip(tmp_path: Path) -> None:
    """vocab.json must map integer strings → class names round-trippably."""
    from lip_reading.train import _save_vocab

    classes = ["Aci", "Acik", "Bal"]
    _save_vocab(tmp_path, classes)

    vocab_path = tmp_path / "vocab.json"
    assert vocab_path.exists()

    with open(vocab_path) as f:
        loaded = json.load(f)

    # JSON keys are always strings; convert back to int
    recovered = {int(k): v for k, v in loaded.items()}
    assert recovered == {0: "Aci", 1: "Acik", 2: "Bal"}


# ---------------------------------------------------------------------------
# 7. CLI --help exits 0
# ---------------------------------------------------------------------------


def test_lip_reading_cli_help_exits_zero() -> None:
    """``python -m lip_reading --help`` must exit with code 0."""
    result = subprocess.run(
        [sys.executable, "-m", "lip_reading", "--help"],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"--help exited {result.returncode}\n"
        f"stdout: {result.stdout.decode()}\n"
        f"stderr: {result.stderr.decode()}"
    )
