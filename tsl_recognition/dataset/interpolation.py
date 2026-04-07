"""
Interpolate missing pose/hand/face landmark segments in sequences of flat keypoint vectors.

MediaPipe sometimes fails to detect hands or pose for a few frames (occlusion,
motion blur). Missing parts are stored as zeros. This module fills those gaps
using linear interpolation from the nearest valid frames so downstream models
see continuous trajectories.

Layout matches extract_keypoints output (defined in config.py):
  pose:       POSE_LANDMARKS * 4   (x, y, z, visibility)  -> POSE_SLICE
  face:       FACE_LANDMARKS * 3   (x, y, z)               -> FACE_SLICE
  left hand:  HAND_LANDMARKS * 3                            -> LEFT_HAND_SLICE
  right hand: HAND_LANDMARKS * 3                            -> RIGHT_HAND_SLICE
"""

from __future__ import annotations

import numpy as np

from ..config import (
    FACE_SLICE,
    FEATURE_DIM,
    LEFT_HAND_SLICE,
    POSE_SLICE,
    RIGHT_HAND_SLICE,
)

EXPECTED_DIM = FEATURE_DIM

# Derive offsets and lengths from the canonical slice constants in config.
_IDX_POSE = POSE_SLICE.start
_LEN_POSE = POSE_SLICE.stop - POSE_SLICE.start
_IDX_FACE = FACE_SLICE.start
_LEN_FACE = FACE_SLICE.stop - FACE_SLICE.start
_IDX_LEFT = LEFT_HAND_SLICE.start
_LEN_LEFT = LEFT_HAND_SLICE.stop - LEFT_HAND_SLICE.start
_IDX_RIGHT = RIGHT_HAND_SLICE.start
_LEN_RIGHT = RIGHT_HAND_SLICE.stop - RIGHT_HAND_SLICE.start


def _slice_part(seq: np.ndarray, start: int, length: int) -> np.ndarray:
    """(T, D) -> (T, length) for the segment [start:start+length]."""
    return seq[:, start : start + length].copy()


def _is_missing(part: np.ndarray, tol: float = 1e-6) -> np.ndarray | np.bool_:
    """(T, ...) -> (T,) bool True where that part is missing (all near-zero)."""
    if part.ndim == 2:
        return np.all(np.abs(part) <= tol, axis=1)
    return np.all(np.abs(part) <= tol)


def _interpolate_part(valid_mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Fill missing positions by linear interpolation.

    values: (T, D), valid_mask: (T,) bool. Interpolates each dimension across time.
    """
    T, D = values.shape
    out = values.copy()
    if np.all(valid_mask):
        return out

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return out
    if len(valid_idx) == 1:
        out[~valid_mask] = values[valid_idx[0]]
        return out

    missing_idx = np.where(~valid_mask)[0]
    for d in range(D):
        out[missing_idx, d] = np.interp(
            missing_idx,
            valid_idx,
            values[valid_idx, d],
        )
    return out


def interpolate_missing_keypoints(
    seq: np.ndarray,
    *,
    missing_tol: float = 1e-6,
) -> np.ndarray:
    """Fill missing pose/face/hand segments in a keypoint sequence by linear
    interpolation from neighboring valid frames.

    Parameters
    ----------
    seq : np.ndarray
        Shape (T, EXPECTED_DIM).
    missing_tol : float
        Treat a part as missing when all its values are <= this in absolute value.

    Returns
    -------
    np.ndarray
        (T, EXPECTED_DIM) with missing segments filled. Unchanged where already valid.
    """
    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim != 2 or seq.shape[1] != EXPECTED_DIM:
        raise ValueError(
            f"Expected shape (T, {EXPECTED_DIM}); got {seq.shape}"
        )

    out = seq.copy()

    parts = [
        ("pose",       _IDX_POSE,  _LEN_POSE),
        ("face",       _IDX_FACE,  _LEN_FACE),
        ("left_hand",  _IDX_LEFT,  _LEN_LEFT),
        ("right_hand", _IDX_RIGHT, _LEN_RIGHT),
    ]
    for _name, start, length in parts:
        part = _slice_part(out, start, length)
        missing = _is_missing(part, tol=missing_tol)
        if not np.any(missing):
            continue
        filled = _interpolate_part(~missing, part)
        out[:, start : start + length] = filled

    return out


def count_missing_per_frame(seq: np.ndarray, missing_tol: float = 1e-6) -> dict[str, np.ndarray]:
    """For debugging: report which parts are missing in each frame.

    Returns
    -------
    dict
        Keys "pose", "face", "left_hand", "right_hand". Values are (T,) bool arrays.
    """
    seq = np.asarray(seq)
    if seq.ndim != 2 or seq.shape[1] != EXPECTED_DIM:
        raise ValueError(f"Expected shape (T, {EXPECTED_DIM}); got {seq.shape}")

    result = {}
    for name, start, length in [
        ("pose",       _IDX_POSE,  _LEN_POSE),
        ("face",       _IDX_FACE,  _LEN_FACE),
        ("left_hand",  _IDX_LEFT,  _LEN_LEFT),
        ("right_hand", _IDX_RIGHT, _LEN_RIGHT),
    ]:
        part = _slice_part(seq, start, length)
        result[name] = _is_missing(part, tol=missing_tol)
    return result
