"""
Augmentation pipeline for the lip-reading dataset.

Operates on the face-landmark slice (249-dim or 120-dim mouth-only) rather
than the full 507-dim TSL vector, so the group masks and mirror permutation
from ``tsl_recognition.dataset.augmentation`` cannot be used directly.

Two phases, matching the TSL convention:

**Phase 1 -- temporal (raw space, before normalisation)**

  Speed variation (uniform resample), non-uniform temporal warp, and frame
  drop. These transforms only index along the time axis and are shape-agnostic.

**Phase 2 -- spatial (normalised space, after normalisation)**

  Coordinate jitter and global scale. No group-level scaling or mirror because
  (a) the face-only vector has no hand/pose groups, and (b) the mouth is
  symmetric so left-right mirror doesn't change the class label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LipAugmentConfig:
    """Hyper-parameters for the lip-reading augmentation pipeline."""

    # Temporal (raw space)
    temporal_resample_prob: float = 0.4
    speed_range: tuple[float, float] = (0.8, 1.2)

    temporal_warp_prob: float = 0.3
    temporal_warp_sigma: float = 0.4

    frame_drop_prob: float = 0.2
    frame_drop_rate_range: tuple[float, float] = (0.05, 0.15)

    # Spatial (normalised space)
    jitter_prob: float = 0.6
    jitter_std: float = 0.04  # std-devs in normalised space; face landmarks are precise

    scale_prob: float = 0.3
    scale_range: tuple[float, float] = (0.95, 1.05)


def augment_temporal(
    keypoints: np.ndarray,
    rng: np.random.Generator,
    cfg: LipAugmentConfig,
) -> np.ndarray:
    """Apply temporal augmentations in raw (un-normalised) space.

    Parameters
    ----------
    keypoints : np.ndarray, shape (T, F)
        Raw face-landmark sequence.
    rng : np.random.Generator
    cfg : LipAugmentConfig

    Returns
    -------
    np.ndarray
        Sequence with (possibly) a different frame count.
    """
    keypoints = _temporal_resample(keypoints, rng, cfg)
    keypoints = _temporal_warp(keypoints, rng, cfg)
    keypoints = _frame_drop(keypoints, rng, cfg)
    return keypoints


def augment_spatial(
    keypoints: np.ndarray,
    rng: np.random.Generator,
    cfg: LipAugmentConfig,
) -> np.ndarray:
    """Apply spatial augmentations in normalised space.

    Parameters
    ----------
    keypoints : np.ndarray, shape (T, F)
        Normalised face-landmark sequence.
    rng : np.random.Generator
    cfg : LipAugmentConfig

    Returns
    -------
    np.ndarray
        Perturbed sequence (same shape).
    """
    keypoints = _spatial_jitter(keypoints, rng, cfg)
    keypoints = _spatial_scale(keypoints, rng, cfg)
    return keypoints


# ---------------------------------------------------------------------------
# Temporal transforms
# ---------------------------------------------------------------------------


def _temporal_resample(
    kp: np.ndarray, rng: np.random.Generator, cfg: LipAugmentConfig
) -> np.ndarray:
    if rng.random() >= cfg.temporal_resample_prob:
        return kp
    num_frames = kp.shape[0]
    if num_frames < 4:
        return kp
    speed = rng.uniform(*cfg.speed_range)
    new_len = max(3, int(round(num_frames / speed)))
    indices = np.linspace(0, num_frames - 1, new_len).round().astype(int)
    return kp[indices]


def _temporal_warp(
    kp: np.ndarray, rng: np.random.Generator, cfg: LipAugmentConfig
) -> np.ndarray:
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


def _frame_drop(
    kp: np.ndarray, rng: np.random.Generator, cfg: LipAugmentConfig
) -> np.ndarray:
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
# Spatial transforms
# ---------------------------------------------------------------------------


def _spatial_jitter(
    kp: np.ndarray, rng: np.random.Generator, cfg: LipAugmentConfig
) -> np.ndarray:
    if rng.random() >= cfg.jitter_prob:
        return kp
    noise = rng.normal(0.0, cfg.jitter_std, size=kp.shape).astype(np.float32)
    return kp + noise


def _spatial_scale(
    kp: np.ndarray, rng: np.random.Generator, cfg: LipAugmentConfig
) -> np.ndarray:
    if rng.random() >= cfg.scale_prob:
        return kp
    factor = rng.uniform(*cfg.scale_range)
    return kp * factor
