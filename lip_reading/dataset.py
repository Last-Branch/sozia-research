"""
Dataset loader for the lip-reading pipeline.

Reuses train/val/test split manifests from the TSL recognition pipeline.
Face-landmark features (249-dim) are sliced inline from the existing 507-dim
keypoint files, avoiding disk duplication.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .augmentation import LipAugmentConfig, augment_spatial, augment_temporal
from .config import DEVICE, FACE_SLICE, LIP_FEATURE_DIM, LIP_SCALERS_DIR, LipTrainConfig


def _worker_init_fn(worker_id: int) -> None:
    """Seed NumPy in each DataLoader worker for reproducible augmentation."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        np.random.seed(worker_info.seed % (2**32))


class LipReadingDataset(Dataset):
    """Memory-efficient dataset that slices face landmarks from 507-dim keypoint files.

    The underlying ``.npy`` files are shared with the TSL recognition pipeline.
    Face features at ``FACE_SLICE`` (dims 132..381, 249-dim) are extracted per
    sample at load time, so no separate extraction step is required.
    """

    def __init__(
        self,
        file_info_list: list[tuple[str, int, int]],
        max_seq_len: int,
        scaler: StandardScaler | None = None,
        sequence_handling: str = "truncate",
        feature_indices: tuple[int, ...] | None = None,
        augment: bool = False,
        augment_cfg: LipAugmentConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------
        file_info_list : list of (path, label, num_frames)
            Entries from the TSL split manifests.
        max_seq_len : int
            Maximum sequence length; longer sequences are truncated or sampled.
        scaler : StandardScaler | None
            Pre-fitted scaler for 249-dim face features.
        sequence_handling : str
            ``"truncate"`` (keep first N frames) or
            ``"uniform_sample"`` (evenly sample N frames from the full sequence).
        feature_indices : tuple[int, ...] | None
            Optional index subset applied to the 249-dim face vector after
            normalization (e.g. mouth-only landmarks). ``None`` keeps all 249.
        augment : bool
            When ``True``, apply temporal and spatial augmentation (training only).
        augment_cfg : LipAugmentConfig | None
            Augmentation hyper-parameters; uses defaults if ``None``.
        """
        self.file_info = file_info_list
        self.max_seq_len = max_seq_len
        self.scaler = scaler
        self.sequence_handling = sequence_handling
        self.feature_indices = feature_indices
        self.augment = augment
        self.augment_cfg = augment_cfg or LipAugmentConfig()
        self._feature_dim = (
            len(feature_indices) if feature_indices is not None else LIP_FEATURE_DIM
        )

    def __len__(self) -> int:
        return len(self.file_info)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load a 507-dim keypoint file, slice face features, pad/truncate.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - keypoints: (max_seq_len, 249)
            - label: scalar class index
            - actual_length: number of real (non-padded) frames
        """
        path, label, _ = self.file_info[idx]

        # Load 507-dim keypoints and extract the 249-dim face slice
        keypoints = np.load(path).astype(np.float32)[:, FACE_SLICE]  # (T, 249)

        # Phase 1: temporal augmentation in raw space (may change frame count)
        if self.augment:
            rng = np.random.default_rng(np.random.randint(0, 2**31))
            keypoints = augment_temporal(keypoints, rng, self.augment_cfg)

        num_frames = keypoints.shape[0]

        if num_frames > self.max_seq_len:
            if self.sequence_handling == "uniform_sample":
                indices = (
                    np.linspace(0, num_frames - 1, self.max_seq_len).round().astype(int)
                )
                keypoints = keypoints[indices]
            else:
                keypoints = keypoints[: self.max_seq_len]
            num_frames = self.max_seq_len

        actual_length = num_frames

        if self.scaler is not None:
            keypoints = (keypoints - self.scaler.mean_) / self.scaler.scale_

        if self.feature_indices is not None:
            keypoints = keypoints[:, self.feature_indices]

        # Phase 2: spatial augmentation in normalised space
        if self.augment:
            keypoints = augment_spatial(keypoints, rng, self.augment_cfg)

        if actual_length < self.max_seq_len:
            padding = np.zeros(
                (self.max_seq_len - actual_length, self._feature_dim), dtype=np.float32
            )
            keypoints = np.vstack([keypoints, padding])

        return (
            torch.tensor(keypoints, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(actual_length, dtype=torch.long),
        )


def get_or_compute_scaler(
    train_files: list[tuple[str, int, int]],
    cfg: LipTrainConfig,
) -> StandardScaler | None:
    """Load or compute a ``StandardScaler`` fitted on face-only training features.

    Scalers are stored in ``scalers/lip_reading/<dataset>/scaler_<split>.pkl``.
    Statistics are computed from training files only to prevent data leakage.
    """
    if not cfg.normalize_features:
        print("Feature normalization: OFF")
        return None

    scaler_dir = LIP_SCALERS_DIR / cfg.dataset_info.display_name
    scaler_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = scaler_dir / f"scaler_{cfg.split_mode}.pkl"

    if scaler_path.exists():
        print(f"\nLoading pre-computed lip scaler from {scaler_path}...")
        with open(scaler_path, "rb") as f:
            scaler: StandardScaler = pickle.load(f)
        print(f"Scaler loaded: mean shape={scaler.mean_.shape}")
        return scaler

    print(
        f"\nComputing lip-reading normalization from {len(train_files)} training files..."
    )
    n_total = 0
    mean = np.zeros(LIP_FEATURE_DIM, dtype=np.float64)
    M2 = np.zeros(LIP_FEATURE_DIM, dtype=np.float64)

    for path, _, _ in tqdm(train_files, desc="Computing lip stats (train only)"):
        keypoints = np.load(path, mmap_mode="r")[:, FACE_SLICE]  # (T, 249)
        num_frames = min(len(keypoints), cfg.max_sequence_length)
        for i in range(num_frames):
            n_total += 1
            delta = keypoints[i].astype(np.float64) - mean
            mean += delta / n_total
            delta2 = keypoints[i].astype(np.float64) - mean
            M2 += delta * delta2

    variance = M2 / n_total
    std = np.sqrt(variance)
    std[std < 1e-8] = 1.0

    scaler = StandardScaler()
    scaler.mean_ = mean.astype(np.float32)
    scaler.scale_ = std.astype(np.float32)
    scaler.var_ = variance.astype(np.float32)
    scaler.n_features_in_ = LIP_FEATURE_DIM
    scaler.n_samples_seen_ = n_total

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Lip scaler saved to {scaler_path}")
    return scaler


def build_lip_loaders(cfg: LipTrainConfig) -> dict[str, Any]:
    """Load the TSL split manifests and build lip-reading ``DataLoader`` instances.

    The split manifests are generated by the TSL recognition pipeline
    (``python -m tsl_recognition split``) and point to 507-dim ``.npy`` files.
    Face features are sliced inside ``LipReadingDataset.__getitem__``.

    Parameters
    ----------
    cfg : LipTrainConfig
        Training configuration including dataset, class list, and split mode.

    Returns
    -------
    dict
        Keys: ``train_loader``, ``val_loader``, ``test_loader``,
        ``train_ds``, ``val_ds``, ``test_ds``,
        ``scaler``, ``class_weights``, ``feature_dim``, ``label_map``,
        ``train_files``, ``val_files``, ``test_files``, ``split_mode``.

    Raises
    ------
    FileNotFoundError
        If no split manifests exist for the selected dataset.
    """
    split_dir = cfg.dataset_info.split_dir
    meta_path = split_dir / "split_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No split manifests found in {split_dir}. "
            "Run `python -m tsl_recognition split` first."
        )

    with open(meta_path) as f:
        meta = json.load(f)
    split_mode = str(meta["mode"])

    actions = cfg.classes_to_process
    allowed_classes = set(actions)
    label_map = {label: num for num, label in enumerate(actions)}

    def _load_partition(partition: str) -> list[tuple[str, int, int]]:
        manifest_path = split_dir / f"{partition}.json"
        with open(manifest_path) as f:
            entries = json.load(f)
        kept: list[tuple[str, int, int]] = []
        dropped = 0
        for e in entries:
            cls = e.get("class_name")
            if cls not in allowed_classes:
                dropped += 1
                continue
            kept.append((e["path"], label_map[cls], int(e["num_frames"])))
        if dropped:
            print(
                f"  Filtered {partition}: dropped {dropped} samples "
                "outside classes_to_process"
            )
        return kept

    train_files = _load_partition("train")
    val_files = _load_partition("val")
    test_files = _load_partition("test")

    print(
        f"\nLoaded {split_mode} split: "
        f"{len(train_files)} train / {len(val_files)} val / {len(test_files)} test"
    )

    scaler = get_or_compute_scaler(train_files, cfg)

    train_labels = [f[1] for f in train_files]
    class_counts = Counter(train_labels)
    total_train = len(train_files)
    class_weights = torch.tensor(
        [
            total_train / (len(class_counts) * class_counts.get(i, 1))
            for i in range(cfg.num_classes)
        ],
        dtype=torch.float32,
    ).to(DEVICE)
    class_weights = class_weights / class_weights.mean()
    print(
        f"  Class weights range: [{class_weights.min():.2f}, {class_weights.max():.2f}]"
    )

    seq_handling = cfg.sequence_handling
    feat_idx = cfg.feature_indices
    train_ds = LipReadingDataset(
        train_files,
        cfg.max_sequence_length,
        scaler,
        seq_handling,
        feat_idx,
        augment=cfg.augment_train,
    )
    val_ds = LipReadingDataset(
        val_files,
        cfg.max_sequence_length,
        scaler,
        seq_handling,
        feat_idx,
        augment=False,
    )
    test_ds = LipReadingDataset(
        test_files,
        cfg.max_sequence_length,
        scaler,
        seq_handling,
        feat_idx,
        augment=False,
    )

    nw = cfg.num_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
        worker_init_fn=_worker_init_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
        worker_init_fn=_worker_init_fn,
    )

    aug_status = "ON (temporal + spatial)" if cfg.augment_train else "OFF"
    print(f"Training augmentation: {aug_status}")
    print(f"Sequence handling: {seq_handling}")
    fdim = cfg.feature_dim
    print(f"\nDataset ({split_mode} split):")
    print(f"  train: ({len(train_files)}, {cfg.max_sequence_length}, {fdim})")
    print(f"  val:   ({len(val_files)}, {cfg.max_sequence_length}, {fdim})")
    print(f"  test:  ({len(test_files)}, {cfg.max_sequence_length}, {fdim})")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "scaler": scaler,
        "class_weights": class_weights,
        "feature_dim": cfg.feature_dim,
        "label_map": label_map,
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
        "split_mode": split_mode,
    }
