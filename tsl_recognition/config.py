"""
Configuration constants, path resolution, and hyperparameters.

All tuneable knobs for the TSL recognition pipeline live here so every other
module just does ``from tsl_recognition.config import cfg`` (or imports
individual values).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .dataset.base import DatasetInfo

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility across all libraries.

    Call this at the beginning of each training run to ensure deterministic
    behaviour regardless of prior RNG state.

    Parameters
    ----------
    seed : int, optional
        Random seed value (default: ``SEED``).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_base_dir() -> Path:
    """Determine the project root directory.

    Resolution order:
      1. ``PROJECT_ROOT`` environment variable (if set).
      2. Parent of the ``tsl_recognition`` package directory (auto-detected).
      3. Current working directory as a fallback.

    Returns
    -------
    Path
        The resolved project root directory.
    """
    # 1. Explicit environment variable
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        p = Path(env_root)
        if p.exists():
            return p

    # 2. Auto-detect from package location: tsl_recognition/ lives one level below root
    pkg_dir = Path(__file__).resolve().parent  # .../sozia-research/tsl_recognition
    candidate = pkg_dir.parent  # .../sozia-research
    if (candidate / "tsl_recognition").is_dir():
        return candidate

    # 3. Fallback
    return Path.cwd()


BASE_DIR = _resolve_base_dir()

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
# Set DATA_ROOT to point to the directory containing your dataset folders
# (e.g., BosphorusSign22k/, AUTSL/). Defaults to BASE_DIR / "data".
DATA_ROOT = Path(os.environ.get("DATA_ROOT", BASE_DIR / "data"))

# MediaPipe model directory (downloaded automatically on first use)
MP_MODEL_DIR = BASE_DIR / "mp-models"

# Trained-model output directory (recognition models)
MODELS_DIR = BASE_DIR / "models" / "recognition"

# Fitted scalers per dataset
SCALERS_DIR = BASE_DIR / "scalers"

# ---------------------------------------------------------------------------
# MediaPipe model URLs
# ---------------------------------------------------------------------------
MODEL_URLS = {
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}

# ---------------------------------------------------------------------------
# Landmark dimensions
# ---------------------------------------------------------------------------
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21

# Full MediaPipe face mesh count (478 points). Kept for reference and for the
# batch-conversion script that remaps old 1692-dim .npy files.
FACE_LANDMARKS_FULL = 478

# ---------------------------------------------------------------------------
# Reduced face-landmark subset -- linguistically relevant regions only.
#
# Indices refer to MediaPipe FaceMesh point IDs (0-477).
# Grouped by function:
#   Lips outer  (20): mouth shape / mouthings
#   Lips inner  (20): lip rounding, aperture
#   R eyebrow   ( 5): grammatical markers (raised / furrowed)
#   L eyebrow   ( 5): same
#   R eye        ( 8): eye widening / squinting
#   L eye        ( 8): same
#   Iris        (10): gaze direction  (468-477)
#   Nose         ( 3): spatial reference anchor
#   Chin         ( 4): jaw opening
# Total: 83 landmarks x 3 (x, y, z) = 249 face features
# ---------------------------------------------------------------------------
FACE_LANDMARK_INDICES: tuple[int, ...] = tuple(
    sorted(
        (
            # Lips -- outer contour
            61,
            146,
            91,
            181,
            84,
            17,
            314,
            405,
            321,
            375,
            291,
            409,
            270,
            269,
            267,
            0,
            37,
            39,
            40,
            185,
            # Lips -- inner contour
            78,
            95,
            88,
            178,
            87,
            14,
            317,
            402,
            318,
            324,
            308,
            415,
            310,
            311,
            312,
            13,
            82,
            81,
            80,
            191,
            # Right eyebrow
            46,
            53,
            52,
            65,
            55,
            # Left eyebrow
            276,
            283,
            282,
            295,
            285,
            # Right eye contour
            33,
            133,
            157,
            158,
            159,
            160,
            144,
            145,
            # Left eye contour
            263,
            362,
            384,
            385,
            386,
            387,
            373,
            374,
            # Iris (right 468-472, left 473-477)
            468,
            469,
            470,
            471,
            472,
            473,
            474,
            475,
            476,
            477,
            # Nose bridge + tip
            1,
            4,
            5,
            # Chin
            152,
            175,
            199,
            200,
        )
    )
)

FACE_LANDMARKS = len(FACE_LANDMARK_INDICES)  # 83

# Feature vector dimension:
# - Pose:       33 points x 4 features (x, y, z, visibility) = 132
# - Face:       83 points x 3 features (x, y, z)             = 249
# - Left hand:  21 points x 3 features (x, y, z)             =  63
# - Right hand: 21 points x 3 features (x, y, z)             =  63
# Total: 132 + 249 + 63 + 63 = 507 features per frame
FEATURE_DIM = POSE_LANDMARKS * 4 + FACE_LANDMARKS * 3 + HAND_LANDMARKS * 3 * 2  # 507

# Byte offsets for each landmark group within the 507-dim vector (used by
# dataset/interpolation.py and dataset/augmentation.py).
POSE_SLICE = slice(0, 132)
FACE_SLICE = slice(132, 381)
LEFT_HAND_SLICE = slice(381, 444)
RIGHT_HAND_SLICE = slice(444, 507)


# ---------------------------------------------------------------------------
# Training / data configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Mutable training hyper-parameters. Create via ``TrainConfig.full()``
    or ``TrainConfig.test()`` for a quick smoke-test run."""

    # Dataset selection (registry key: "bosphorus", "autsl", ...)
    dataset: str = "bosphorus"

    # Class selection
    classes_to_process: list[str] = field(default_factory=list)

    # Sequence handling
    max_sequence_length: int = 150
    min_sequence_length: int = 10
    # Strategy for sequences longer than max_sequence_length:
    #   "truncate"       -- keep first max_sequence_length frames (default;
    #                      preserves native frame rate / temporal density)
    #   "uniform_sample" -- evenly sample max_sequence_length frames across
    #                      the full sequence so no temporal region is lost
    #                      (better for attention models; can hurt RNNs that
    #                      depend on smooth frame-to-frame dynamics)
    sequence_handling: str = "truncate"

    # Training
    batch_size: int = 64
    epochs: int = 300
    learning_rate: float = 5e-4
    grad_clip_norm: float = 1.0  # max gradient norm (0 to disable)
    label_smoothing: float = 0.1  # label smoothing for CrossEntropyLoss (0 to disable)

    # Early stopping (0 to disable)
    early_stopping_patience: int = 35  # stop if val acc doesn't improve for N epochs
    min_epochs: int = 250  # don't allow early stopping before this epoch

    # Validation frequency
    val_every: int = 5  # validate every N epochs

    # LR scheduling
    # - "onecycle": OneCycleLR stepped every batch (recommended default)
    # - "cosine_warm_restarts": Linear warmup (epoch-stepped) + CosineAnnealingWarmRestarts
    # - "plateau": ReduceLROnPlateau (legacy)
    # - "none": fixed learning rate
    lr_scheduler: str = "onecycle"

    # Warmup (used for cosine_warm_restarts; OneCycle uses pct_start instead)
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1

    # OneCycleLR params
    # If None, max_lr will be derived from learning_rate at runtime.
    onecycle_max_lr: float | None = None
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4

    # CosineAnnealingWarmRestarts params
    cosine_t0: int = 10
    cosine_t_mult: int = 2
    cosine_eta_min: float = 1e-6

    # Data handling
    use_class_weights: bool = True
    use_weighted_sampling: bool = False
    normalize_features: bool = True
    num_workers: int = 4
    augment_train: bool = True  # apply data augmentation to training set

    # Split configuration
    split_mode: str = "signer"  # "signer" or "random"
    val_split: float = 0.15  # used only in random mode
    test_split: float = 0.15  # used only in random mode

    # Model architecture
    # One of the keys in models.MODEL_REGISTRY (e.g. "gru")
    model_arch: str = "gru"

    # Model size override (None = auto-detect from num_classes)
    # Set to "small", "large", or "xlarge" to force a specific size preset.
    model_size_override: str | None = None

    # Dropout probability (None = auto-select based on model_size).
    # Defaults: small/large -> 0.4, xlarge -> 0.2 (applied in evaluation/train.py).
    dropout: float | None = None

    # Extraction
    apply_interpolation: bool = True

    # ------------------------------------------------------------------
    # Dataset info helper
    # ------------------------------------------------------------------
    @property
    def dataset_info(self) -> "DatasetInfo":
        """Return the :class:`DatasetInfo` instance for the selected dataset."""
        from .dataset.registry import get_dataset_info

        return get_dataset_info(self.dataset, BASE_DIR)

    @staticmethod
    def _all_sign_classes(dataset: str = "bosphorus") -> list[str]:
        """Get all sign class names for the given dataset.

        Returns an empty list if the dataset directory / label file doesn't exist.
        """
        from .dataset.registry import get_dataset_info

        try:
            info = get_dataset_info(dataset, BASE_DIR)
            return info.class_names()
        except Exception:
            return []

    @classmethod
    def full(cls, dataset: str = "bosphorus") -> "TrainConfig":
        """Full training configuration for the given dataset."""
        return cls(
            dataset=dataset,
            classes_to_process=cls._all_sign_classes(dataset),
            max_sequence_length=150,
            batch_size=64,
            epochs=300,
            learning_rate=5e-4,
            lr_scheduler="onecycle",
            warmup_epochs=10,
        )

    @classmethod
    def test(cls, n_classes: int = 10, dataset: str = "bosphorus") -> "TrainConfig":
        """Quick smoke-test configuration."""
        all_classes = cls._all_sign_classes(dataset)
        return cls(
            dataset=dataset,
            classes_to_process=all_classes[:n_classes],
            max_sequence_length=100,
            batch_size=32,
            epochs=100,
            learning_rate=1e-3,
            lr_scheduler="onecycle",
            warmup_epochs=5,
        )

    @property
    def num_classes(self) -> int:
        """Return the total number of sign classes to train on."""
        return len(self.classes_to_process)

    @property
    def actions(self) -> np.ndarray:
        """Return the list of sign class names as a numpy array."""
        return np.array(self.classes_to_process)

    @property
    def model_size(self) -> str:
        """Determine model size based on number of classes.

        If ``model_size_override`` is set, that value is returned directly.
        Otherwise: ``'small'`` for <=50 classes, ``'large'`` for >50.
        """
        if self.model_size_override is not None:
            return self.model_size_override
        return "small" if self.num_classes <= 50 else "large"
