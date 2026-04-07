"""
Data augmentation pipeline for sign language recognition.

Provides temporal, spatial, and landmark-dropout augmentations that operate
directly on flat 507-dim keypoint vectors (no restructuring needed).

Three augmentation families, applied in two phases inside
``LazySignDataset.__getitem__``:

**Phase 1 -- before normalisation** (call :func:`augment_temporal`):

  Temporal transforms that change the *number* of frames (speed variation,
  frame drop). Must run before truncation / padding.

**Phase 2 -- after normalisation** (call :func:`augment_spatial`, then
:func:`apply_landmark_dropout`):

  Spatial transforms (jitter, scale, translate) and landmark dropout.
  **All value-level perturbations operate in normalised (zero-mean,
  unit-variance) space** so that every feature receives a perturbation
  proportional to its natural variation. Raw-space perturbations are
  dangerous because 28 % of features have scale < 0.01 (face z-coords,
  hand root z-coords, pose visibility) -- a jitter of 0.01 in raw space
  becomes 10-180 000 standard deviations after normalisation, which
  completely destroys the signal.

Usage
-----
In ``LazySignDataset.__getitem__``::

    rng = np.random.default_rng()
    kp = augment_temporal(kp, rng, cfg)       # phase 1 (raw)
    kp = truncate(kp)
    kp = normalise(kp)
    kp = augment_spatial(kp, rng, cfg)        # phase 2 (normalised)
    kp = apply_landmark_dropout(kp, rng, cfg) # phase 2 (normalised)
    kp = pad(kp)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import FACE_LANDMARK_INDICES, FACE_LANDMARKS, HAND_LANDMARKS, POSE_LANDMARKS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AugmentConfig:
    """Hyperparameters for the augmentation pipeline.

    All probabilities are independent -- each augmentation is applied (or
    skipped) with its own coin flip so that any combination is possible.

    **Spatial parameters are in normalised space** (i.e. units of
    standard deviations), so a jitter_std of 0.05 means 5 % of each
    feature's natural variation -- safe regardless of raw-scale
    differences between features.
    """

    # -- Mirror (raw space) --
    mirror_prob: float = 0.5

    # -- Temporal (raw space) --
    temporal_resample_prob: float = 0.3
    speed_range: tuple[float, float] = (0.8, 1.2)

    frame_drop_prob: float = 0.2
    frame_drop_rate_range: tuple[float, float] = (0.05, 0.15)

    temporal_warp_prob: float = 0.3
    temporal_warp_sigma: float = 0.5
    """Controls how non-uniform the temporal warp is.

    Higher values produce more extreme speed variation between segments.
    At ``0`` the warp degenerates to identity (all weights equal).
    """

    # -- Spatial (normalised space) --
    jitter_prob: float = 0.5
    jitter_std: float = 0.05  # std-devs in normalised space
    jitter_group_scales: dict[str, float] | None = None
    """Per-group multipliers applied to ``jitter_std``.

    Hands are noisier in practice (MediaPipe hand tracking is less
    stable), while face landmarks are very precise. When set to
    ``None`` (default), uses ``{"pose": 1.0, "face": 0.5,
    "left_hand": 1.5, "right_hand": 1.5}``.
    """

    scale_prob: float = 0.3
    scale_range: tuple[float, float] = (0.95, 1.05)

    translate_prob: float = 0.3
    translate_range: float = 0.1  # per-axis shift in std-devs

    independent_scale_prob: float = 0.2
    independent_scale_range: tuple[float, float] = (0.9, 1.1)
    """Per-group scale factors simulate inter-signer morphological variation.

    Each body-part group (pose, face, left hand, right hand) is
    independently scaled by a factor drawn uniformly from this range,
    producing longer/shorter arms, bigger/smaller hands, etc.
    """

    # -- Landmark dropout (normalised space) --
    landmark_dropout_prob: float = 0.1  # per-group probability


# ---------------------------------------------------------------------------
# Index masks (computed once at module load)
# ---------------------------------------------------------------------------


def _build_landmark_masks() -> dict[str, np.ndarray]:
    """Build per-axis and per-group index masks into the 507-dim vector."""
    pose_dim = POSE_LANDMARKS * 4  # 132
    face_dim = FACE_LANDMARKS * 3  # 249
    lhand_dim = HAND_LANDMARKS * 3  # 63
    rhand_dim = HAND_LANDMARKS * 3  # 63

    pose_start = 0
    face_start = pose_dim          # 132
    lhand_start = face_start + face_dim   # 381
    rhand_start = lhand_start + lhand_dim  # 444

    groups = {
        "pose":       np.arange(pose_start,   pose_start  + pose_dim),
        "face":       np.arange(face_start,   face_start  + face_dim),
        "left_hand":  np.arange(lhand_start,  lhand_start + lhand_dim),
        "right_hand": np.arange(rhand_start,  rhand_start + rhand_dim),
    }

    x_indices: list[int] = []
    y_indices: list[int] = []
    z_indices: list[int] = []

    for i in range(POSE_LANDMARKS):
        base = pose_start + i * 4
        x_indices.append(base)
        y_indices.append(base + 1)
        z_indices.append(base + 2)

    for i in range(FACE_LANDMARKS):
        base = face_start + i * 3
        x_indices.append(base)
        y_indices.append(base + 1)
        z_indices.append(base + 2)

    for i in range(HAND_LANDMARKS):
        base = lhand_start + i * 3
        x_indices.append(base)
        y_indices.append(base + 1)
        z_indices.append(base + 2)

    for i in range(HAND_LANDMARKS):
        base = rhand_start + i * 3
        x_indices.append(base)
        y_indices.append(base + 1)
        z_indices.append(base + 2)

    x_arr = np.array(x_indices)
    y_arr = np.array(y_indices)
    z_arr = np.array(z_indices)
    spatial = np.sort(np.concatenate([x_arr, y_arr, z_arr]))

    spatial_set = set(spatial.tolist())
    group_spatial = {
        name: np.array(sorted(i for i in idx if i in spatial_set))
        for name, idx in groups.items()
    }

    return {
        "groups": groups,
        "group_spatial": group_spatial,
        "spatial": spatial,
        "x": x_arr,
        "y": y_arr,
        "z": z_arr,
    }


_MASKS = _build_landmark_masks()


def _build_mirror_swap_map() -> np.ndarray:
    """Build a permutation array for left-right landmark mirroring.

    Returns a 1-D int array of length ``FEATURE_DIM`` (507). Indexing a
    frame vector ``v[perm]`` swaps:

    * Left-hand landmarks (21 pts) with right-hand landmarks (21 pts).
    * 16 paired pose landmarks (left shoulder <-> right shoulder, etc.).
    * 36 paired face landmarks (eyes, eyebrows, iris, lips).

    Unpaired / midline landmarks map to themselves.
    The permutation is its own inverse (applying it twice gives identity).
    """
    dim = POSE_LANDMARKS * 4 + FACE_LANDMARKS * 3 + HAND_LANDMARKS * 3 * 2

    perm = np.arange(dim, dtype=np.intp)

    pose_start = 0
    face_start = POSE_LANDMARKS * 4                       # 132
    lhand_start = face_start + FACE_LANDMARKS * 3          # 381
    rhand_start = lhand_start + HAND_LANDMARKS * 3         # 444

    # 1. Swap left hand <-> right hand (63 features each)
    for i in range(HAND_LANDMARKS * 3):
        perm[lhand_start + i] = rhand_start + i
        perm[rhand_start + i] = lhand_start + i

    # 2. Swap paired pose landmarks (4 features each)
    _POSE_PAIRS: list[tuple[int, int]] = [
        (1, 4), (2, 5), (3, 6),
        (7, 8),
        (9, 10),
        (11, 12), (13, 14), (15, 16),
        (17, 18), (19, 20), (21, 22),
        (23, 24), (25, 26), (27, 28),
        (29, 30), (31, 32),
    ]
    for a, b in _POSE_PAIRS:
        for off in range(4):
            perm[pose_start + a * 4 + off] = pose_start + b * 4 + off
            perm[pose_start + b * 4 + off] = pose_start + a * 4 + off

    # 3. Swap paired face landmarks (3 features each)
    _face_idx = {mp: loc for loc, mp in enumerate(FACE_LANDMARK_INDICES)}

    _FACE_PAIRS_MP: list[tuple[int, int]] = [
        # Lips - outer contour (9 pairs; midline 0, 17 stay)
        (61, 291), (146, 375), (91, 321), (181, 405), (84, 314),
        (37, 267), (39, 269),  (40, 270), (185, 409),
        # Lips - inner contour (9 pairs; midline 13, 14 stay)
        (78, 308), (95, 324),  (88, 318), (178, 402), (87, 317),
        (82, 312), (81, 311),  (80, 310), (191, 415),
        # Eyebrows (5 pairs)
        (46, 276), (53, 283), (52, 282), (65, 295), (55, 285),
        # Eye contours (8 pairs)
        (33, 263), (133, 362), (157, 384), (158, 385),
        (159, 386), (160, 387), (144, 373), (145, 374),
        # Iris (5 pairs)
        (468, 473), (469, 474), (470, 475), (471, 476), (472, 477),
    ]
    for mp_a, mp_b in _FACE_PAIRS_MP:
        loc_a = _face_idx[mp_a]
        loc_b = _face_idx[mp_b]
        for off in range(3):
            perm[face_start + loc_a * 3 + off] = face_start + loc_b * 3 + off
            perm[face_start + loc_b * 3 + off] = face_start + loc_a * 3 + off

    return perm


_MIRROR_PERM = _build_mirror_swap_map()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def augment_temporal(
    keypoints: np.ndarray,
    rng: np.random.Generator | None = None,
    cfg: AugmentConfig | None = None,
) -> np.ndarray:
    """Apply temporal augmentations (speed variation + frame drop).

    Call **before** truncation and normalisation.

    Parameters
    ----------
    keypoints : np.ndarray, shape ``(num_frames, feature_dim)``
        Raw keypoint sequence.
    rng : np.random.Generator, optional
    cfg : AugmentConfig, optional

    Returns
    -------
    np.ndarray
        Sequence with (possibly) different frame count.
    """
    if rng is None:
        rng = np.random.default_rng()
    if cfg is None:
        cfg = AugmentConfig()

    keypoints = _mirror_lr(keypoints, rng, cfg)
    keypoints = _temporal_resample(keypoints, rng, cfg)
    keypoints = _temporal_warp(keypoints, rng, cfg)
    keypoints = _frame_drop(keypoints, rng, cfg)
    return keypoints


def augment_spatial(
    keypoints: np.ndarray,
    rng: np.random.Generator | None = None,
    cfg: AugmentConfig | None = None,
) -> np.ndarray:
    """Apply spatial augmentations (jitter, scale, translate).

    Call **after** normalisation.

    Parameters
    ----------
    keypoints : np.ndarray, shape ``(num_frames, feature_dim)``
        **Normalised** keypoint sequence.
    rng : np.random.Generator, optional
    cfg : AugmentConfig, optional

    Returns
    -------
    np.ndarray
        Perturbed keypoint sequence (same shape).
    """
    if rng is None:
        rng = np.random.default_rng()
    if cfg is None:
        cfg = AugmentConfig()

    keypoints = _spatial_jitter(keypoints, rng, cfg)
    keypoints = _spatial_scale(keypoints, rng, cfg)
    keypoints = _independent_scale(keypoints, rng, cfg)
    keypoints = _spatial_translate(keypoints, rng, cfg)
    return keypoints


def apply_landmark_dropout(
    keypoints: np.ndarray,
    rng: np.random.Generator | None = None,
    cfg: AugmentConfig | None = None,
) -> np.ndarray:
    """Zero out entire landmark groups to simulate occlusion.

    Call **after** normalisation so that zeroed features sit at the
    normalised mean (0.0) rather than at extreme negative values.

    Parameters
    ----------
    keypoints : np.ndarray, shape ``(num_frames, feature_dim)``
        **Normalised** keypoint sequence.
    rng : np.random.Generator, optional
    cfg : AugmentConfig, optional

    Returns
    -------
    np.ndarray
        Keypoint sequence with some landmark groups possibly zeroed out.
    """
    if rng is None:
        rng = np.random.default_rng()
    if cfg is None:
        cfg = AugmentConfig()
    return _landmark_dropout(keypoints, rng, cfg)


# ---------------------------------------------------------------------------
# Mirror (Phase 1 - raw space)
# ---------------------------------------------------------------------------


def _mirror_lr(kp: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Mirror landmarks left-right to simulate a left-handed signer."""
    if rng.random() >= cfg.mirror_prob:
        return kp

    kp = kp[:, _MIRROR_PERM]

    midline_x = kp[:, 0:1]
    x_idx = _MASKS["x"]
    kp[:, x_idx] = 2.0 * midline_x - kp[:, x_idx]

    return kp


# ---------------------------------------------------------------------------
# Temporal augmentations
# ---------------------------------------------------------------------------


def _temporal_resample(kp: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Randomly change playback speed via uniform temporal resampling."""
    if rng.random() >= cfg.temporal_resample_prob:
        return kp

    num_frames = kp.shape[0]
    if num_frames < 4:
        return kp

    speed = rng.uniform(*cfg.speed_range)
    new_len = max(3, int(round(num_frames / speed)))
    indices = np.linspace(0, num_frames - 1, new_len).round().astype(int)
    return kp[indices]


def _temporal_warp(kp: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Non-uniform temporal resampling to simulate natural signing rhythm."""
    if rng.random() >= cfg.temporal_warp_prob:
        return kp

    num_frames = kp.shape[0]
    if num_frames < 4:
        return kp

    weights = rng.lognormal(0.0, cfg.temporal_warp_sigma, size=num_frames)
    cumulative = np.cumsum(weights)
    cumulative = (
        (cumulative - cumulative[0])
        / (cumulative[-1] - cumulative[0])
        * (num_frames - 1)
    )
    indices = np.clip(cumulative.round().astype(int), 0, num_frames - 1)
    return kp[indices]


def _frame_drop(kp: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Randomly drop a fraction of frames to simulate temporal noise."""
    if rng.random() >= cfg.frame_drop_prob:
        return kp

    num_frames = kp.shape[0]
    if num_frames < 6:
        return kp

    drop_rate = rng.uniform(*cfg.frame_drop_rate_range)
    n_drop = max(1, int(num_frames * drop_rate))

    if num_frames - n_drop < 3:
        return kp

    drop_idx = rng.choice(num_frames, size=n_drop, replace=False)
    keep_mask = np.ones(num_frames, dtype=bool)
    keep_mask[drop_idx] = False
    return kp[keep_mask]


# ---------------------------------------------------------------------------
# Spatial augmentations (operate in NORMALISED space)
# ---------------------------------------------------------------------------


_DEFAULT_JITTER_GROUP_SCALES: dict[str, float] = {
    "pose": 1.0,
    "face": 0.5,
    "left_hand": 1.5,
    "right_hand": 1.5,
}


def _spatial_jitter(kp: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Add Gaussian noise to all spatial coordinates (x, y, z)."""
    if rng.random() >= cfg.jitter_prob:
        return kp

    scales = cfg.jitter_group_scales or _DEFAULT_JITTER_GROUP_SCALES
    group_spatial = _MASKS["group_spatial"]

    kp = kp.copy()
    for group_name, idx in group_spatial.items():
        if len(idx) == 0:
            continue
        group_std = cfg.jitter_std * scales.get(group_name, 1.0)
        noise = rng.normal(0.0, group_std, size=(kp.shape[0], len(idx))).astype(np.float32)
        kp[:, idx] += noise
    return kp


def _spatial_scale(kp: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Scale all spatial coordinates by a single random factor."""
    if rng.random() >= cfg.scale_prob:
        return kp

    factor = rng.uniform(*cfg.scale_range)
    spatial = _MASKS["spatial"]
    kp = kp.copy()
    kp[:, spatial] *= factor
    return kp


def _independent_scale(kp: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Scale each body-part group by an independent random factor."""
    if rng.random() >= cfg.independent_scale_prob:
        return kp

    group_spatial = _MASKS["group_spatial"]
    kp = kp.copy()
    for _name, idx in group_spatial.items():
        if len(idx) == 0:
            continue
        factor = rng.uniform(*cfg.independent_scale_range)
        kp[:, idx] *= factor
    return kp


def _spatial_translate(kp: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Shift all spatial coordinates by a random per-axis offset."""
    if rng.random() >= cfg.translate_prob:
        return kp

    shift = rng.uniform(-cfg.translate_range, cfg.translate_range, size=3).astype(np.float32)

    kp = kp.copy()
    kp[:, _MASKS["x"]] += shift[0]
    kp[:, _MASKS["y"]] += shift[1]
    kp[:, _MASKS["z"]] += shift[2]
    return kp


# ---------------------------------------------------------------------------
# Landmark dropout
# ---------------------------------------------------------------------------


def _landmark_dropout(kp: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Zero out entire landmark groups to simulate occlusion."""
    groups = _MASKS["groups"]
    group_names = list(groups.keys())

    drop_flags = {
        name: rng.random() < cfg.landmark_dropout_prob for name in group_names
    }

    # Never drop all groups
    if all(drop_flags.values()):
        keep = rng.choice(group_names)
        drop_flags[keep] = False

    if not any(drop_flags.values()):
        return kp

    kp = kp.copy()
    for name, should_drop in drop_flags.items():
        if should_drop:
            kp[:, groups[name]] = 0.0

    return kp
