"""Drawing helpers for MediaPipe landmarks and inference HUD.

This module provides functions to visualize:
1. Detected landmarks (face, hands, pose) on video frames
2. Inference status and predictions (HUD overlay)
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from .inference import InferenceMode, SignRecorder

import cv2
import mediapipe as mp
try:
    _hands = mp.tasks.vision.HandLandmarksConnections
    _face = mp.tasks.vision.FaceLandmarksConnections
    _draw = mp.tasks.vision.drawing_utils
    _sty = mp.tasks.vision.drawing_styles
    HAS_DRAW = True
except Exception:
    HAS_DRAW = False

HAS_POSE = False
if HAS_DRAW:
    try:
        _pose = mp.tasks.vision.PoseLandmarksConnections
        HAS_POSE = True
    except Exception:
        pass


def _pts(frame: np.ndarray, lms: list[Any], color: tuple[int, int, int], r: int = 2) -> None:
    """Draw simple circles for landmarks (fallback when MediaPipe drawing not available).

    Parameters
    ----------
    frame : np.ndarray
        BGR frame to draw on.
    lms : list
        List of landmark objects with x, y attributes (normalized 0-1).
    color : tuple
        BGR color tuple.
    r : int, optional
        Circle radius in pixels (default: 2).
    """
    h, w = frame.shape[:2]
    for lm in lms:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), r, color, -1)


def draw_hand_landmarks(f: np.ndarray, hr: Any) -> np.ndarray:
    """Draw hand landmarks with connections and handedness labels.

    Parameters
    ----------
    f : np.ndarray
        Input BGR frame.
    hr : HandLandmarkerResult
        Hand detection result from MediaPipe.

    Returns
    -------
    np.ndarray
        Frame with hand landmarks drawn.
    """
    if not HAS_DRAW or not hr.hand_landmarks:
        return f
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).copy()
    for i, hlm in enumerate(hr.hand_landmarks):
        _draw.draw_landmarks(
            rgb, hlm, _hands.HAND_CONNECTIONS,
            _sty.get_default_hand_landmarks_style(),
            _sty.get_default_hand_connections_style(),
        )
        if hr.handedness and i < len(hr.handedness):
            lb = hr.handedness[i][0].category_name
            h, w, _ = rgb.shape
            tx = int(min(lm.x for lm in hlm) * w)
            ty = int(min(lm.y for lm in hlm) * h) - 10
            cv2.putText(rgb, lb, (tx, max(0, ty)), cv2.FONT_HERSHEY_DUPLEX, 0.6, (88, 205, 54), 1, cv2.LINE_AA)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_face_landmarks(f: np.ndarray, fr: Any) -> np.ndarray:
    """Draw face mesh landmarks with tessellation and contours.

    Parameters
    ----------
    f : np.ndarray
        Input BGR frame.
    fr : FaceLandmarkerResult
        Face detection result from MediaPipe.

    Returns
    -------
    np.ndarray
        Frame with face landmarks drawn.
    """
    if not HAS_DRAW or not fr.face_landmarks:
        return f
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).copy()
    for flm in fr.face_landmarks:
        for conn, sfn in [
            (_face.FACE_LANDMARKS_TESSELATION, _sty.get_default_face_mesh_tesselation_style),
            (_face.FACE_LANDMARKS_CONTOURS, _sty.get_default_face_mesh_contours_style),
            (_face.FACE_LANDMARKS_LEFT_IRIS, _sty.get_default_face_mesh_iris_connections_style),
            (_face.FACE_LANDMARKS_RIGHT_IRIS, _sty.get_default_face_mesh_iris_connections_style),
        ]:
            _draw.draw_landmarks(
                image=rgb, landmark_list=flm, connections=conn,
                landmark_drawing_spec=None, connection_drawing_spec=sfn(),
            )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_pose_landmarks(f: np.ndarray, pr: Any) -> np.ndarray:
    """Draw pose (body) landmarks with skeletal connections.

    Parameters
    ----------
    f : np.ndarray
        Input BGR frame.
    pr : PoseLandmarkerResult
        Pose detection result from MediaPipe.

    Returns
    -------
    np.ndarray
        Frame with pose landmarks drawn.
    """
    if not HAS_POSE or not pr.pose_landmarks:
        return f
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).copy()
    cs = _draw.DrawingSpec(color=(0, 255, 0), thickness=2)
    ls = _sty.get_default_pose_landmarks_style()
    for plm in pr.pose_landmarks:
        _draw.draw_landmarks(rgb, plm, _pose.POSE_LANDMARKS, ls, cs)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_results(frame: np.ndarray, face_r: Any, hand_r: Any, pose_r: Any) -> np.ndarray:
    """Draw all detected landmarks on a BGR frame.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR frame.
    face_r : FaceLandmarkerResult
        Face detection result.
    hand_r : HandLandmarkerResult
        Hand detection result.
    pose_r : PoseLandmarkerResult
        Pose detection result.

    Returns
    -------
    np.ndarray
        Frame with all landmarks drawn.
    """
    frame = draw_hand_landmarks(frame, hand_r)
    frame = draw_face_landmarks(frame, face_r)
    frame = draw_pose_landmarks(frame, pose_r)
    if not HAS_DRAW and face_r.face_landmarks:
        _pts(frame, face_r.face_landmarks[0], (0, 180, 255), 1)
    if not HAS_POSE and pose_r.pose_landmarks:
        _pts(frame, pose_r.pose_landmarks[0], (0, 255, 0), 3)
    return frame


def draw_status_bar(
    frame: np.ndarray,
    recorder: SignRecorder,
    mode: InferenceMode,
    predictions: list[tuple[str, float]] | None = None,
    debug_info: dict[str, Any] | None = None,
) -> np.ndarray:
    """Draw the HUD status bar at the top of the frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame to draw on (modified in-place).
    recorder : SignRecorder
        Current recording state.
    mode : InferenceMode
        Current inference mode.
    predictions : list | None, optional
        List of (class_name, probability) tuples.
    debug_info : dict | None, optional
        Debug information (e.g., motion values).

    Returns
    -------
    np.ndarray
        Frame with HUD overlay drawn.
    """
    from .inference import InferenceMode

    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 140), (20, 20, 20), -1)
    mc = {
        InferenceMode.TRIGGER: (0, 255, 255),
        InferenceMode.MOTION: (0, 255, 0),
        InferenceMode.CONTINUOUS: (255, 128, 0),
        InferenceMode.VIDEO_FILE: (255, 0, 255),
    }
    cv2.putText(frame, f"Mode: {mode.name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mc.get(mode, (255, 255, 255)), 2, cv2.LINE_AA)

    if recorder.is_recording:
        cv2.putText(frame, f"RECORDING: {len(recorder.frames)} frames", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        if int(time.time() * 4) % 2:
            cv2.circle(frame, (w - 30, 20), 10, (0, 0, 255), -1)
    else:
        cv2.putText(frame, "Ready (waiting for sign)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1, cv2.LINE_AA)

    if mode == InferenceMode.MOTION and debug_info:
        sm = debug_info.get("smoothed_motion", 0)
        bw = int(min(debug_info.get("motion", 0) * 2000, 200))
        cv2.rectangle(frame, (10, 60), (10 + bw, 75), (0, 200, 200), -1)
        cv2.rectangle(frame, (10, 60), (210, 75), (100, 100, 100), 1)
        cv2.putText(frame, f"Motion: {sm:.4f}", (220, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    if predictions:
        y0 = 85
        cv2.putText(frame, "Predictions:", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        for i, (cls, prob) in enumerate(predictions[:3]):
            y = y0 + 18 + i * 16
            cv2.rectangle(frame, (100, y - 10), (100 + int(prob * 150), y), (0, 180, 0), -1)
            cv2.putText(frame, f"{cls}: {prob * 100:.1f}%", (260, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if mode == InferenceMode.TRIGGER:
        cv2.putText(frame, "SPACE start/stop, 'q' quit", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    elif mode == InferenceMode.MOTION:
        cv2.putText(frame, "Move hands to auto-detect, 'q' quit", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    return frame
