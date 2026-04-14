"""
Real-time and video-file inference for sign language recognition.

Supports four modes via ``InferenceMode``:
    TRIGGER     - press SPACE to start/stop recording
    MOTION      - auto-detect sign boundaries via hand motion
    CONTINUOUS  - sliding window prediction every N frames
    VIDEO_FILE  - collect all frames from video, predict at end

Usage::

    python -m tsl_recognition infer --mode motion
    python -m tsl_recognition infer --mode video --video path/to/video.mp4
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ..config import (
    DEVICE,
    FACE_LANDMARKS,
    HAND_LANDMARKS,
    MODELS_DIR,
    MP_MODEL_DIR,
    POSE_LANDMARKS,
    SCALERS_DIR,
    TrainConfig,
)
from ..dataset.interpolation import EXPECTED_DIM, interpolate_missing_keypoints
from ..extraction.landmarks import (
    create_landmarkers,
    detect_landmarks,
    extract_keypoints,
)
from ..models import build_model
from .visualization import draw_results, draw_status_bar


def _has_display() -> bool:
    """Return True if a GUI display is available for cv2.imshow."""
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class InferenceMode(Enum):
    TRIGGER = auto()
    MOTION = auto()
    CONTINUOUS = auto()
    VIDEO_FILE = auto()


@dataclass
class SignRecorder:
    """Manages sign recording state with motion detection.

    Attributes
    ----------
    frames : list
        Collected keypoint frames for the current sign.
    is_recording : bool
        Whether we're currently recording a sign.
    motion_history : deque
        Recent hand motion velocities (for smoothing).
    low_motion_count : int
        Number of consecutive frames with low motion.
    last_hand_positions : np.ndarray | None
        Hand positions from previous frame (for velocity calculation).
    """

    frames: list = field(default_factory=list)
    is_recording: bool = False
    motion_history: deque = field(default_factory=lambda: deque(maxlen=15))
    low_motion_count: int = 0
    last_hand_positions: np.ndarray | None = None

    def reset(self) -> None:
        """Reset the recorder state for a new sign."""
        self.frames = []
        self.is_recording = False
        self.low_motion_count = 0
        self.last_hand_positions = None

    def compute_hand_motion(self, keypoints: np.ndarray) -> float:
        """Calculate hand motion velocity from keypoints.

        Parameters
        ----------
        keypoints : np.ndarray
            Flattened keypoint vector (507 dimensions).

        Returns
        -------
        float
            Hand motion velocity (Euclidean distance moved since last frame).
        """
        lstart = POSE_LANDMARKS * 4 + FACE_LANDMARKS * 3
        rstart = lstart + HAND_LANDMARKS * 3

        hand_len = HAND_LANDMARKS * 3
        lh = keypoints[lstart : lstart + hand_len].reshape(-1, 3)[:, :2]
        rh = keypoints[rstart : rstart + hand_len].reshape(-1, 3)[:, :2]

        positions = []
        if np.any(np.abs(lh) > 1e-6):
            positions.append(np.mean(lh, axis=0))
        if np.any(np.abs(rh) > 1e-6):
            positions.append(np.mean(rh, axis=0))

        if not positions:
            return 0.0

        cur = np.array(positions)

        if self.last_hand_positions is None or len(cur) != len(
            self.last_hand_positions
        ):
            self.last_hand_positions = cur
            return 0.0

        vel = float(np.mean(np.linalg.norm(cur - self.last_hand_positions, axis=1)))
        self.last_hand_positions = cur
        self.motion_history.append(vel)
        return vel

    def get_smoothed_motion(self) -> float:
        """Get temporally smoothed motion value.

        Returns
        -------
        float
            Average motion over recent frames.
        """
        return float(np.mean(list(self.motion_history))) if self.motion_history else 0.0


def preprocess_sequence(
    frames: list[np.ndarray],
    scaler: StandardScaler | None,
    max_len: int,
    normalize: bool = True,
    apply_interpolation: bool = True,
    sequence_handling: str = "truncate",
) -> tuple[np.ndarray, int]:
    """Preprocess collected keypoints identically to training pipeline.

    Parameters
    ----------
    frames : list[np.ndarray]
        List of keypoint arrays (one per frame).
    scaler : StandardScaler | None
        Fitted scaler for normalization.
    max_len : int
        Maximum sequence length (truncate/pad to this).
    normalize : bool, optional
        Whether to apply normalization (default: True).
    apply_interpolation : bool, optional
        Whether to interpolate missing landmarks (default: True).
    sequence_handling : str, optional
        Strategy for sequences longer than *max_len*:
        ``"truncate"`` keeps the first *max_len* frames (default);
        ``"uniform_sample"`` evenly samples *max_len* frames across the
        full sequence (matches ``LazySignDataset`` training behaviour).

    Returns
    -------
    tuple[np.ndarray, int]
        - Preprocessed keypoint array of shape (max_len, feature_dim).
        - Actual (non-padded) sequence length.
    """
    kp = np.array(frames, dtype=np.float32)
    n = len(kp)

    if apply_interpolation and kp.shape[1] == EXPECTED_DIM:
        try:
            kp = interpolate_missing_keypoints(kp).astype(np.float32)
        except Exception:
            pass

    if normalize and scaler is not None:
        kp = scaler.transform(kp)

    if n > max_len:
        if sequence_handling == "uniform_sample":
            indices = np.linspace(0, n - 1, max_len).round().astype(int)
            kp = kp[indices]
        else:
            kp = kp[:max_len]
        n = max_len

    actual_length = n

    if n < max_len:
        kp = np.vstack([kp, np.zeros((max_len - n, kp.shape[1]), dtype=np.float32)])

    return kp, actual_length


def predict_sign(
    model: nn.Module,
    kp_array: np.ndarray,
    actual_length: int,
    device: torch.device,
    actions: np.ndarray,
    top_k: int = 5,
) -> tuple[list[tuple[str, float]], np.ndarray]:
    """Predict sign class from preprocessed keypoint sequence.

    Parameters
    ----------
    model : nn.Module
        Trained model (any architecture from the model registry).
    kp_array : np.ndarray
        Preprocessed keypoint sequence of shape (max_len, feature_dim).
    actual_length : int
        Number of real (non-padded) frames in *kp_array*.
    device : torch.device
        Device to run inference on (CPU/GPU).
    actions : np.ndarray
        Array of sign class names.
    top_k : int, optional
        Number of top predictions to return (default: 5).

    Returns
    -------
    tuple[list, np.ndarray]
        - List of (class_name, probability) tuples for top-k predictions
        - Full probability distribution over all classes
    """
    x = torch.tensor(kp_array, dtype=torch.float32).unsqueeze(0).to(device)
    lengths = torch.tensor([actual_length], dtype=torch.long).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x, lengths=lengths), dim=1).cpu().numpy()[0]
    idx = np.argsort(probs)[::-1][:top_k]
    return [(actions[i], float(probs[i])) for i in idx], probs


def _find_latest_run(models_dir: Path) -> Path:
    """Return the most recent ``run_*`` directory under *models_dir*.

    Raises
    ------
    FileNotFoundError
        If no run directories exist.
    """
    runs = sorted(models_dir.glob("run_*"))
    if not runs:
        raise FileNotFoundError(
            f"No run directories found in {models_dir}. Train a model first."
        )
    return runs[-1]


def _load_run(run_dir: Path) -> dict:
    """Load a trained model and its associated artefacts from a run directory.

    All metadata is read from the ``config.json`` saved by ``train.py``, so
    the correct architecture, scaler, and class list are always used.

    The scaler is loaded from ``dataset_info.processed_dir`` (derived from the
    dataset name stored in ``config.json``), not from a hardcoded path.

    Parameters
    ----------
    run_dir : Path
        Path to a ``run_*`` directory produced by ``train.py``.

    Returns
    -------
    dict
        Keys: ``model``, ``scaler``, ``actions``, ``max_len``,
        ``sequence_handling``, ``normalize``, ``feature_dim``.
    """
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {run_dir}")
    with open(config_path) as f:
        meta = json.load(f)

    model_arch: str = meta["model_arch"]
    model_size: str = meta["model_size"]
    feature_dim: int = int(meta["feature_dim"])
    dropout: float = float(meta.get("dropout") or 0.4)
    split_mode: str = meta.get("split_mode", "signer")
    classes: list[str] = meta["classes_to_process"]
    num_classes: int = len(classes)
    max_len: int = int(meta["max_sequence_length"])
    seq_handling: str = meta.get("sequence_handling", "truncate")
    normalize: bool = bool(meta.get("normalize_features", True))
    dataset_name: str = meta.get("dataset", "bosphorus")

    print(f"Run directory : {run_dir}")
    print(f"Architecture  : {model_arch} ({model_size})")
    print(f"Classes       : {num_classes}")
    print(f"Feature dim   : {feature_dim}")
    print(f"Max seq len   : {max_len}")
    print(f"Seq handling  : {seq_handling}")

    scaler = None
    if normalize:
        scaler_path = (
            SCALERS_DIR
            / TrainConfig(dataset=dataset_name).dataset_info.display_name
            / f"scaler_{split_mode}.pkl"
        )
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            print(f"Scaler loaded : {scaler_path.name}")
        else:
            print(
                f"WARNING: scaler not found at {scaler_path} — running without normalisation"
            )

    model = build_model(
        arch=model_arch,
        input_size=feature_dim,
        num_classes=num_classes,
        model_size=model_size,
        dropout=dropout,
    ).to(DEVICE)

    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best_model.pt in {run_dir}")
    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Checkpoint    : {ckpt_path.name}")

    actions = np.array(classes)

    return {
        "model": model,
        "scaler": scaler,
        "actions": actions,
        "max_len": max_len,
        "sequence_handling": seq_handling,
        "normalize": normalize,
        "feature_dim": feature_dim,
    }


# ===================================================================
# Main inference loop
# ===================================================================
def run_inference(
    mode: InferenceMode = InferenceMode.MOTION,
    cfg: TrainConfig | None = None,
    video_path: Path | str | None = None,
    run_dir: Path | str | None = None,
    show_landmarks: bool = True,
    top_k: int = 5,
    motion_start: float = 0.015,
    motion_end: float = 0.005,
    motion_end_frames: int = 10,
    min_sign_frames: int = 15,
    continuous_every: int = 30,
    normalize: bool = True,
    headless: bool = False,
) -> None:
    """Run the inference loop (webcam or video file).

    Parameters
    ----------
    run_dir : Path | str | None
        Path to a ``run_*`` directory produced by ``train.py``.  If *None*,
        the most recent run directory under ``MODELS_DIR`` is used.
    headless : bool
        If *True*, skip all cv2.imshow/waitKey calls.  Auto-detected when
        not set explicitly and no DISPLAY / WAYLAND_DISPLAY env var is found.
    """
    if cfg is None:
        cfg = TrainConfig.full()
    show_gui = not headless and _has_display()
    if not show_gui:
        print("No display detected — running in headless mode (no GUI window)")

    rd = Path(run_dir) if run_dir is not None else _find_latest_run(MODELS_DIR)
    run_data = _load_run(rd)
    model = run_data["model"]
    scaler = run_data["scaler"]
    actions = run_data["actions"]
    max_len = run_data["max_len"]
    seq_handling = run_data["sequence_handling"]
    normalize = run_data["normalize"]

    print(f"Feature normalization: {'ON' if normalize else 'OFF'}")
    print(f"Inference mode: {mode.name}")

    if mode == InferenceMode.VIDEO_FILE:
        if not video_path:
            raise ValueError("--video path is required for video mode")
        vp = Path(video_path)
        if not vp.exists():
            raise FileNotFoundError(f"Video not found: {vp}")
        cap = cv2.VideoCapture(str(vp))
        print(f"Video: {vp}\nExpected class: {vp.parent.name}")
    else:
        vp = None
        cap = cv2.VideoCapture(0, getattr(cv2, "CAP_V4L2", cv2.CAP_ANY))
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print("Webcam initialized")

    rec = SignRecorder()
    cur_preds = None
    history: list[str] = []
    fcount = 0
    ts = 0
    print(f"\n{'=' * 60}\nStarting inference loop...\n{'=' * 60}\n")

    with create_landmarkers(MP_MODEL_DIR) as (face_lm, hand_lm, hand_crop_lm, pose_lm):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if mode == InferenceMode.VIDEO_FILE:
                    break
                continue
            fcount += 1
            if mode != InferenceMode.VIDEO_FILE:
                frame = cv2.flip(frame, 1)
            ts += 33
            fr, hr, pr = detect_landmarks(
                frame, face_lm, hand_lm, hand_crop_lm, pose_lm, ts
            )
            if show_landmarks:
                frame = draw_results(frame, fr, hr, pr)
            kp = extract_keypoints(fr, hr, pr)
            kp = np.asarray(kp, dtype=np.float32).reshape(-1)
            if kp.size < EXPECTED_DIM:
                kp = np.pad(kp, (0, EXPECTED_DIM - kp.size))
            elif kp.size > EXPECTED_DIM:
                kp = kp[:EXPECTED_DIM]
            dbg: dict = {}

            if mode == InferenceMode.TRIGGER:
                if rec.is_recording:
                    rec.frames.append(kp)
                    if len(rec.frames) >= max_len:
                        arr, alen = preprocess_sequence(
                            rec.frames, scaler, max_len, normalize, True, seq_handling
                        )
                        cur_preds, _ = predict_sign(
                            model, arr, alen, DEVICE, actions, top_k
                        )
                        history.append(cur_preds[0][0])
                        print(
                            f"Prediction: {cur_preds[0][0]} ({cur_preds[0][1] * 100:.1f}%)"
                        )
                        rec.reset()

            elif mode == InferenceMode.MOTION:
                mot = rec.compute_hand_motion(kp)
                sm = rec.get_smoothed_motion()
                dbg["motion"] = mot
                dbg["smoothed_motion"] = sm

                if not rec.is_recording:
                    if sm > motion_start:
                        rec.is_recording = True
                        rec.frames = [kp]
                        rec.low_motion_count = 0
                        print(f"Sign started (motion={sm:.4f})")
                else:
                    rec.frames.append(kp)

                    if sm < motion_end:
                        rec.low_motion_count += 1
                    else:
                        rec.low_motion_count = 0

                    do_pred = False
                    if (
                        rec.low_motion_count >= motion_end_frames
                        and len(rec.frames) >= min_sign_frames
                    ):
                        print(f"Sign ended (low motion for {motion_end_frames} frames)")
                        do_pred = True
                    elif len(rec.frames) >= max_len:
                        print("Sign ended (max frames reached)")
                        do_pred = True

                    if do_pred:
                        arr, alen = preprocess_sequence(
                            rec.frames, scaler, max_len, normalize, True, seq_handling
                        )
                        cur_preds, _ = predict_sign(
                            model, arr, alen, DEVICE, actions, top_k
                        )
                        history.append(cur_preds[0][0])
                        print(
                            f"Prediction ({len(rec.frames)} frames): {cur_preds[0][0]} ({cur_preds[0][1] * 100:.1f}%)"
                        )
                        for j, (c, p) in enumerate(cur_preds[:3]):
                            print(f"  {j + 1}. {c}: {p * 100:.1f}%")
                        rec.reset()

            elif mode == InferenceMode.CONTINUOUS:
                rec.frames.append(kp)
                if len(rec.frames) >= max_len:
                    rec.frames = rec.frames[-max_len:]
                if (
                    fcount % continuous_every == 0
                    and len(rec.frames) >= min_sign_frames
                ):
                    arr, alen = preprocess_sequence(
                        rec.frames, scaler, max_len, normalize, True, seq_handling
                    )
                    cur_preds, _ = predict_sign(
                        model, arr, alen, DEVICE, actions, top_k
                    )

            elif mode == InferenceMode.VIDEO_FILE:
                rec.frames.append(kp)

            if show_gui:
                frame = draw_status_bar(frame, rec, mode, cur_preds, dbg)
                cv2.imshow("Sign Language Recognition", frame)
            key = cv2.waitKey(10) & 0xFF if show_gui else 0xFF
            if key == ord("q"):
                break
            elif key == ord(" ") and mode == InferenceMode.TRIGGER:
                if rec.is_recording:
                    if len(rec.frames) >= min_sign_frames:
                        arr, alen = preprocess_sequence(
                            rec.frames, scaler, max_len, normalize, True, seq_handling
                        )
                        cur_preds, _ = predict_sign(
                            model, arr, alen, DEVICE, actions, top_k
                        )
                        history.append(cur_preds[0][0])
                        print(
                            f"Prediction ({len(rec.frames)} frames): {cur_preds[0][0]} ({cur_preds[0][1] * 100:.1f}%)"
                        )
                    else:
                        print(
                            f"Too few frames ({len(rec.frames)}), need {min_sign_frames}"
                        )
                    rec.reset()
                else:
                    rec.is_recording = True
                    rec.frames = []
                    print("Recording started...")

    cap.release()
    if show_gui:
        cv2.destroyAllWindows()

    if mode == InferenceMode.VIDEO_FILE and rec.frames and vp is not None:
        print(f"\n{'=' * 60}\nVIDEO FILE RESULTS\n{'=' * 60}")
        print(f"Video: {vp.name}\nExpected class: {vp.parent.name}")
        print(f"Frames extracted: {len(rec.frames)}")
        arr, alen = preprocess_sequence(
            rec.frames, scaler, max_len, normalize, True, seq_handling
        )
        preds, _ = predict_sign(model, arr, alen, DEVICE, actions, top_k)
        exp = vp.parent.name
        print(f"\nTop-{top_k} Predictions:")
        for i, (c, p) in enumerate(preds):
            mk = " <-- MATCH!" if c == exp else ""
            print(f"  {i + 1}. {c}: {p * 100:.1f}%{mk}")
        print(f"\nFinal prediction: {preds[0][0]}")
        print(f"Ground truth: {exp}")
        print(f"Correct: {'YES' if preds[0][0] == exp else 'NO'}")

    if mode != InferenceMode.VIDEO_FILE and history:
        print(f"\n{'=' * 60}\nSESSION SUMMARY\n{'=' * 60}")
        print(f"Total predictions: {len(history)}")
        print(f"Predictions: {' -> '.join(history[-10:])}")
