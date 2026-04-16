"""
Configuration constants and training hyper-parameters for the lip-reading pipeline.

The lip-reading model is a word classifier trained on the face-landmark subset
of the existing TSL recognition datasets.  The 83-point face subset spans
feature dimensions [132:381] (249-dim) in the 507-dim keypoint vector produced
by the TSL extraction pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tsl_recognition.dataset.base import DatasetInfo

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_base_dir() -> Path:
    """Determine the project root directory.

    Resolution order:
      1. ``PROJECT_ROOT`` environment variable (if set).
      2. Parent of the ``lip_reading`` package directory (auto-detected).
      3. Current working directory as a fallback.
    """
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        p = Path(env_root)
        if p.exists():
            return p
    pkg_dir = Path(__file__).resolve().parent  # .../sozia-research/lip_reading
    candidate = pkg_dir.parent  # .../sozia-research
    if (candidate / "lip_reading").is_dir():
        return candidate
    return Path.cwd()


BASE_DIR = _resolve_base_dir()

# ---------------------------------------------------------------------------
# Feature dimensions
# ---------------------------------------------------------------------------
# Slice of the 507-dim keypoint vector that contains the 83-point face subset.
# Layout in the 507-dim vector:
#   [0:132]   pose (33 × 4)
#   [132:381] face (83 × 3)  ← lip-reading features
#   [381:444] left hand (21 × 3)
#   [444:507] right hand (21 × 3)
FACE_SLICE = slice(132, 381)
LIP_FEATURE_DIM = 249  # 83 landmarks × 3 (x, y, z)

# ---------------------------------------------------------------------------
# Mouth-only landmark subset (outer + inner lip contours, 40 landmarks × 3)
# ---------------------------------------------------------------------------
# MediaPipe FaceMesh IDs for outer and inner lip contours.
_MOUTH_LANDMARK_IDS: frozenset[int] = frozenset(
    {
        # Outer lip contour (20)
        0, 17, 37, 39, 40, 61, 84, 91, 146, 181,
        185, 267, 269, 270, 291, 314, 321, 375, 405, 409,
        # Inner lip contour (20)
        13, 14, 78, 80, 81, 82, 87, 88, 95, 178,
        191, 308, 310, 311, 312, 317, 318, 324, 402, 415,
    }
)

# Positions of mouth landmarks within the sorted FACE_LANDMARK_INDICES tuple,
# then expanded to the flat feature indices inside the 249-dim face vector.
# Computed at import time — avoids magic numbers in downstream code.
from tsl_recognition.config import FACE_LANDMARK_INDICES as _FACE_LM_IDX  # noqa: E402

_MOUTH_POSITIONS: tuple[int, ...] = tuple(
    i for i, lm_id in enumerate(_FACE_LM_IDX) if lm_id in _MOUTH_LANDMARK_IDS
)
MOUTH_FEATURE_INDICES: tuple[int, ...] = tuple(
    fi for p in _MOUTH_POSITIONS for fi in (p * 3, p * 3 + 1, p * 3 + 2)
)
MOUTH_FEATURE_DIM: int = len(MOUTH_FEATURE_INDICES)  # 120

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LIP_MODELS_DIR = BASE_DIR / "models" / "lip_reading"
LIP_SCALERS_DIR = BASE_DIR / "scalers" / "lip_reading"


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class LipTrainConfig:
    """Training hyper-parameters for the lip-reading word classifier.

    Attribute names that overlap with ``TrainConfig`` are intentionally
    kept identical so that shared helpers work via duck-typing without
    modification.
    """

    # Dataset
    dataset: str = "bosphorus"
    classes_to_process: list[str] = field(default_factory=list)

    # Sequence handling
    max_sequence_length: int = 150
    min_sequence_length: int = 10
    sequence_handling: str = "truncate"

    # Training
    batch_size: int = 64
    epochs: int = 300
    learning_rate: float = 5e-4
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.1
    dropout: float = 0.4

    # Early stopping
    early_stopping_patience: int = 35
    min_epochs: int = 200
    val_every: int = 5

    # LR scheduling (kept identical to TrainConfig for duck-typing)
    lr_scheduler: str = "onecycle"
    warmup_epochs: int = 10
    warmup_start_factor: float = 0.1
    onecycle_max_lr: float | None = None
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4
    cosine_t0: int = 10
    cosine_t_mult: int = 2
    cosine_eta_min: float = 1e-6

    # Data
    use_class_weights: bool = True
    normalize_features: bool = True
    num_workers: int = 4
    # Augmentation disabled until tuned for face-only features.
    augment_train: bool = False
    split_mode: str = "signer"
    # Feature subset: when True, only the 40 mouth/lip landmarks (120-dim)
    # are used instead of the full 83-landmark face set (249-dim).
    use_mouth_only: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def dataset_info(self) -> "DatasetInfo":
        """Return the ``DatasetInfo`` instance for the selected dataset."""
        from tsl_recognition.dataset.registry import get_dataset_info

        return get_dataset_info(self.dataset, BASE_DIR)

    @staticmethod
    def _all_classes(dataset: str = "bosphorus") -> list[str]:
        from tsl_recognition.dataset.registry import get_dataset_info

        try:
            return get_dataset_info(dataset, BASE_DIR).class_names()
        except Exception:
            return []

    @classmethod
    def full(cls, dataset: str = "bosphorus") -> "LipTrainConfig":
        """Full training configuration for all classes in the dataset."""
        return cls(
            dataset=dataset,
            classes_to_process=cls._all_classes(dataset),
        )

    @classmethod
    def test(cls, n_classes: int = 10, dataset: str = "bosphorus") -> "LipTrainConfig":
        """Quick smoke-test configuration (10 classes, 50 epochs)."""
        all_classes = cls._all_classes(dataset)
        return cls(
            dataset=dataset,
            classes_to_process=all_classes[:n_classes],
            max_sequence_length=100,
            batch_size=32,
            epochs=50,
            early_stopping_patience=10,
            min_epochs=30,
        )

    @property
    def feature_dim(self) -> int:
        """Input feature dimension: 120 (mouth-only) or 249 (full face)."""
        return MOUTH_FEATURE_DIM if self.use_mouth_only else LIP_FEATURE_DIM

    @property
    def feature_indices(self) -> tuple[int, ...] | None:
        """Feature index subset to apply after face normalization, or None."""
        return MOUTH_FEATURE_INDICES if self.use_mouth_only else None

    @property
    def num_classes(self) -> int:
        """Total number of word classes to train on."""
        return len(self.classes_to_process)
