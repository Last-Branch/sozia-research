"""Smoke tests for the tsl_recognition pipeline.

These tests verify that modules import cleanly, core objects can be
constructed with no data on disk, and the GRU model produces output of
the expected shape.  No GPU, no real dataset files required.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import csv

import pytest
import torch


# ---------------------------------------------------------------------------
# 1. Import smoke
# ---------------------------------------------------------------------------


def test_tsl_recognition_imports() -> None:
    """tsl_recognition package-level import must succeed."""
    import tsl_recognition  # noqa: F401


def test_tsl_recognition_submodules_import() -> None:
    """Core sub-modules must be importable without side-effects."""
    from tsl_recognition import config  # noqa: F401
    from tsl_recognition.models import MODEL_REGISTRY, build_model  # noqa: F401
    from tsl_recognition.dataset import registry  # noqa: F401


# ---------------------------------------------------------------------------
# 2. TrainConfig defaults
# ---------------------------------------------------------------------------


def test_train_config_default_construction() -> None:
    """TrainConfig() with no arguments must use documented defaults."""
    from tsl_recognition.config import TrainConfig

    cfg = TrainConfig()

    assert cfg.dataset == "bosphorus"
    assert cfg.classes_to_process == []
    assert cfg.max_sequence_length == 150
    assert cfg.min_sequence_length == 10
    assert cfg.batch_size == 64
    assert cfg.epochs == 300
    assert cfg.learning_rate == pytest.approx(5e-4)
    assert cfg.model_arch == "gru"
    assert cfg.lr_scheduler == "onecycle"


def test_train_config_num_classes_reflects_classes_list() -> None:
    """num_classes property must equal len(classes_to_process)."""
    from tsl_recognition.config import TrainConfig

    cfg = TrainConfig(classes_to_process=["A", "B", "C"])
    assert cfg.num_classes == 3


def test_train_config_model_size_small_threshold() -> None:
    """model_size is 'small' for <=50 classes, 'large' for >50."""
    from tsl_recognition.config import TrainConfig

    small_cfg = TrainConfig(classes_to_process=list(map(str, range(50))))
    assert small_cfg.model_size == "small"

    large_cfg = TrainConfig(classes_to_process=list(map(str, range(51))))
    assert large_cfg.model_size == "large"


def test_train_config_model_size_override() -> None:
    """model_size_override must take precedence over the auto-detection rule."""
    from tsl_recognition.config import TrainConfig

    cfg = TrainConfig(
        classes_to_process=list(map(str, range(10))),
        model_size_override="xlarge",
    )
    assert cfg.model_size == "xlarge"


def test_train_config_feature_dim_constant() -> None:
    """FEATURE_DIM must equal 507 (pose + face + hands)."""
    from tsl_recognition.config import FEATURE_DIM

    assert FEATURE_DIM == 507


# ---------------------------------------------------------------------------
# 3. Model registry + GRU forward pass
# ---------------------------------------------------------------------------


def test_model_registry_contains_gru() -> None:
    """MODEL_REGISTRY must expose a 'gru' key."""
    from tsl_recognition.models import MODEL_REGISTRY

    assert "gru" in MODEL_REGISTRY


def test_build_model_returns_nn_module() -> None:
    """build_model('gru', ...) must return a torch.nn.Module."""
    import torch.nn as nn
    from tsl_recognition.models import build_model

    model = build_model(arch="gru", input_size=507, num_classes=10, model_size="small")
    assert isinstance(model, nn.Module)


def test_build_model_unknown_arch_raises() -> None:
    """build_model with an unknown arch must raise ValueError."""
    from tsl_recognition.models import build_model

    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model(arch="nonexistent", input_size=507, num_classes=10)


def test_gru_forward_pass_output_shape() -> None:
    """GRU forward pass on a tiny dummy batch must produce the right output shape."""
    from tsl_recognition.models import build_model

    model = build_model(arch="gru", input_size=507, num_classes=10, model_size="small")
    model.eval()

    # (batch=2, seq_len=5, features=507)
    dummy = torch.zeros(2, 5, 507)

    with torch.no_grad():
        logits = model(dummy)

    assert logits.shape == (2, 10), f"Expected (2, 10), got {logits.shape}"


def test_gru_forward_pass_is_finite() -> None:
    """GRU output on zero input must contain no NaN or Inf values."""
    from tsl_recognition.models import build_model

    model = build_model(arch="gru", input_size=507, num_classes=5, model_size="small")
    model.eval()

    dummy = torch.zeros(2, 5, 507)

    with torch.no_grad():
        logits = model(dummy)

    assert torch.isfinite(logits).all(), "logits contain NaN or Inf"


# ---------------------------------------------------------------------------
# 4. Dataset registry
# ---------------------------------------------------------------------------


def test_dataset_registry_contains_known_datasets() -> None:
    """DATASET_REGISTRY must include 'bosphorus' and 'autsl'."""
    from tsl_recognition.dataset.registry import DATASET_REGISTRY

    assert "bosphorus" in DATASET_REGISTRY
    assert "autsl" in DATASET_REGISTRY


def test_get_dataset_info_bosphorus_returns_instance(tmp_path: Path) -> None:
    """get_dataset_info('bosphorus', ...) must return a DatasetInfo without data on disk."""
    from tsl_recognition.dataset.base import DatasetInfo
    from tsl_recognition.dataset.registry import get_dataset_info

    info = get_dataset_info("bosphorus", tmp_path)

    assert isinstance(info, DatasetInfo)
    assert info.name == "bosphorus"


def _write_bosphorus_classes_csv(base_dir: Path) -> Path:
    """Write a minimal BosphorusSign22k_classes.csv for testing."""
    csv_path = base_dir / "data" / "BosphorusSign22k" / "BosphorusSign22k_classes.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["SubsetID", "ClassID", "ClassName_tr", "ClassName_eng"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "SubsetID": "Health",
                "ClassID": "0001",
                "ClassName_tr": "Aci",
                "ClassName_eng": "Pain",
            }
        )
        writer.writerow(
            {
                "SubsetID": "Finance",
                "ClassID": "0002",
                "ClassName_tr": "Acik",
                "ClassName_eng": "Open",
            }
        )
        writer.writerow(
            {
                "SubsetID": "General",
                "ClassID": "0003",
                "ClassName_tr": "Bal",
                "ClassName_eng": "Honey",
            }
        )
    return csv_path


def test_bosphorus_class_names_returns_turkish_names(tmp_path: Path) -> None:
    """class_names() must return Turkish names from the classes CSV, sorted by ClassID."""
    from tsl_recognition.dataset.registry import get_dataset_info

    _write_bosphorus_classes_csv(tmp_path)
    info = get_dataset_info("bosphorus", tmp_path)

    names = info.class_names()
    assert names == ["Aci", "Acik", "Bal"]


def test_bosphorus_iter_raw_videos_yields_turkish_names(tmp_path: Path) -> None:
    """iter_raw_videos() must yield Turkish class names, not numeric ClassIDs."""
    _write_bosphorus_classes_csv(tmp_path)

    raw_dir = tmp_path / "data" / "BosphorusSign22k" / "raw"
    class_dir = raw_dir / "0001"
    class_dir.mkdir(parents=True)
    (class_dir / "User_2_001.mp4").touch()

    from tsl_recognition.dataset.registry import get_dataset_info

    info = get_dataset_info("bosphorus", tmp_path)

    results = list(info.iter_raw_videos())
    assert len(results) == 1
    sample_id, class_name, video_path = results[0]
    assert class_name == "Aci"
    assert sample_id == "User_2_001"


def test_bosphorus_iter_raw_videos_filters_by_class(tmp_path: Path) -> None:
    """iter_raw_videos(classes=['Acik']) must skip classes not in the filter."""
    _write_bosphorus_classes_csv(tmp_path)

    raw_dir = tmp_path / "data" / "BosphorusSign22k" / "raw"
    for class_id, name in [("0001", "Aci"), ("0002", "Acik")]:
        d = raw_dir / class_id
        d.mkdir(parents=True)
        (d / "User_2_001.mp4").touch()

    from tsl_recognition.dataset.registry import get_dataset_info

    info = get_dataset_info("bosphorus", tmp_path)

    results = list(info.iter_raw_videos(classes=["Acik"]))
    assert len(results) == 1
    assert results[0][1] == "Acik"


def test_get_dataset_info_autsl_returns_instance(tmp_path: Path) -> None:
    """get_dataset_info('autsl', ...) must return a DatasetInfo without data on disk."""
    from tsl_recognition.dataset.base import DatasetInfo
    from tsl_recognition.dataset.registry import get_dataset_info

    info = get_dataset_info("autsl", tmp_path)

    assert isinstance(info, DatasetInfo)
    assert info.name == "autsl"


def test_get_dataset_info_unknown_raises(tmp_path: Path) -> None:
    """get_dataset_info with an unknown name must raise ValueError."""
    from tsl_recognition.dataset.registry import get_dataset_info

    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset_info("nonexistent", tmp_path)


# ---------------------------------------------------------------------------
# 5. CLI --help exits 0
# ---------------------------------------------------------------------------


def test_cli_help_exits_zero() -> None:
    """``python -m tsl_recognition --help`` must exit with code 0."""
    result = subprocess.run(
        [sys.executable, "-m", "tsl_recognition", "--help"],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"--help exited {result.returncode}\nstdout: {result.stdout.decode()}\n"
        f"stderr: {result.stderr.decode()}"
    )
