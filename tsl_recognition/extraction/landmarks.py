"""
MediaPipe landmark detection utilities.

Provides:
    - Automatic model download (``ensure_model``)
    - Context-managed landmarker creation (``create_landmarkers``)
    - Per-frame detection (``detect_landmarks``)
    - Flat keypoint vector extraction (``extract_keypoints``)
"""

from __future__ import annotations

import urllib.request
from contextlib import ExitStack, contextmanager
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from ..config import (
    FACE_LANDMARK_INDICES,
    FACE_LANDMARKS,
    HAND_LANDMARKS,
    MODEL_URLS,
    MP_MODEL_DIR,
    POSE_LANDMARKS,
)

# ---------------------------------------------------------------------------
# MediaPipe task aliases
# ---------------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
def ensure_model(model_path: Path, url: str) -> Path:
    """Download a MediaPipe model file if it does not already exist.

    This function checks if the model file exists locally. If not, it downloads
    the model from the provided URL and saves it to the specified path.

    Parameters
    ----------
    model_path : Path
        Local path where the model should be saved.
    url : str
        URL to download the model from if it doesn't exist locally.

    Returns
    -------
    Path
        The path to the model file (either existing or newly downloaded).
    """
    model_path = Path(model_path)
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading model to {model_path}")
        urllib.request.urlretrieve(url, model_path)
    return model_path


# ---------------------------------------------------------------------------
# Landmarker context manager
# ---------------------------------------------------------------------------
@contextmanager
def create_landmarkers(model_dir: Path | str | None = None):
    """Yield ``(face_lm, hand_lm, hand_crop_lm, pose_lm)`` inside a managed context.

    Pose and hand landmarkers use VIDEO mode for temporal consistency.
    The face landmarker and hand-crop landmarker use IMAGE mode because they
    operate on pose-guided crops whose dimensions vary per frame.

    Parameters
    ----------
    model_dir : Path | str | None, optional
        Directory containing MediaPipe model files. If None, uses MP_MODEL_DIR.

    Yields
    ------
    tuple[FaceLandmarker, HandLandmarker, HandLandmarker, PoseLandmarker]
        The four initialized landmarker objects: face (IMAGE), hand (VIDEO),
        hand-crop (IMAGE), and pose (VIDEO).
    """
    model_dir = Path(model_dir or MP_MODEL_DIR)
    face_path = ensure_model(model_dir / "face_landmarker.task", MODEL_URLS["face"])
    hand_path = ensure_model(model_dir / "hand_landmarker.task", MODEL_URLS["hand"])
    pose_path = ensure_model(model_dir / "pose_landmarker_full.task", MODEL_URLS["pose"])

    # IMAGE mode: required for pose-guided crops whose dimensions vary per frame.
    # Lower confidence thresholds compensate for the face being small in full-body shots.
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(face_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )
    # IMAGE mode for pose-guided crop fallback; only one hand per crop expected.
    hand_crop_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
    )

    with ExitStack() as stack:
        face_lm = stack.enter_context(FaceLandmarker.create_from_options(face_options))
        hand_lm = stack.enter_context(HandLandmarker.create_from_options(hand_options))
        hand_crop_lm = stack.enter_context(
            HandLandmarker.create_from_options(hand_crop_options)
        )
        pose_lm = stack.enter_context(PoseLandmarker.create_from_options(pose_options))
        yield face_lm, hand_lm, hand_crop_lm, pose_lm


# ---------------------------------------------------------------------------
# Pose-guided face cropping
# ---------------------------------------------------------------------------
# Pose landmark indices used to locate the face region.
_FACE_POSE_IDS = [0, 2, 5, 7, 8]   # nose, left eye, right eye, left ear, right ear
_EAR_LEFT, _EAR_RIGHT = 7, 8       # indices into pose landmarks

# Minimum crop half-size (pixels) so we never feed a tiny patch.
_MIN_CROP_HALF = 150

# Multiplier applied to the inter-ear distance to get the crop half-size.
_CROP_SCALE = 3.0


def _face_crop_region(
    pose_landmarks,
    frame_h: int,
    frame_w: int,
) -> tuple[int, int, int, int] | None:
    """Compute a square crop region around the face using pose landmarks.

    Returns ``(y1, y2, x1, x2)`` pixel coordinates, or *None* if pose
    landmarks are unavailable.
    """
    if not pose_landmarks:
        return None

    lms = pose_landmarks
    cx = np.mean([lms[i].x for i in _FACE_POSE_IDS]) * frame_w
    cy = np.mean([lms[i].y for i in _FACE_POSE_IDS]) * frame_h

    ear_dist = abs(lms[_EAR_LEFT].x - lms[_EAR_RIGHT].x) * frame_w
    half = max(ear_dist * _CROP_SCALE, _MIN_CROP_HALF)

    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(frame_w, int(cx + half))
    y2 = min(frame_h, int(cy + half))

    if (x2 - x1) < 50 or (y2 - y1) < 50:
        return None
    return y1, y2, x1, x2


def _remap_face_landmarks(face_result, crop_region, frame_h: int, frame_w: int):
    """Remap face landmark coordinates from the crop back to the full frame.

    Face landmarks returned by the FaceLandmarker are normalised to the crop
    image.  This function converts them to be normalised w.r.t. the original
    full-size frame so they are on the same coordinate system as the pose and
    hand landmarks.
    """
    if not face_result.face_landmarks:
        return face_result

    y1, y2, x1, x2 = crop_region
    crop_w = x2 - x1
    crop_h = y2 - y1

    for face_lms in face_result.face_landmarks:
        for lm in face_lms:
            lm.x = (lm.x * crop_w + x1) / frame_w
            lm.y = (lm.y * crop_h + y1) / frame_h
            # z is relative depth; scale by crop_w / frame_w to keep proportional.
            lm.z = lm.z * crop_w / frame_w

    return face_result


# ---------------------------------------------------------------------------
# Pose-guided hand cropping
# ---------------------------------------------------------------------------
_WRIST_LEFT, _WRIST_RIGHT = 15, 16
_ELBOW_LEFT, _ELBOW_RIGHT = 13, 14
_PINKY_LEFT, _PINKY_RIGHT = 17, 18
_INDEX_LEFT, _INDEX_RIGHT = 19, 20

_MIN_HAND_CROP_HALF = 120
_HAND_CROP_SCALE = 2.5
_HAND_UPSCALE_MIN = 256


def _hand_crop_region(
    pose_landmarks,
    wrist_idx: int,
    elbow_idx: int,
    pinky_idx: int,
    index_idx: int,
    frame_h: int,
    frame_w: int,
) -> tuple[int, int, int, int] | None:
    """Compute a square crop region around a hand using pose landmarks.

    The crop is centred slightly below the wrist (toward the fingers) so that
    the full hand is captured.

    Parameters
    ----------
    pose_landmarks
        List of pose landmark objects from MediaPipe.
    wrist_idx : int
        Pose landmark index for the wrist (15 or 16).
    elbow_idx : int
        Pose landmark index for the elbow (13 or 14).
    pinky_idx : int
        Pose landmark index for the hand pinky (17 or 18).
    index_idx : int
        Pose landmark index for the hand index finger (19 or 20).
    frame_h : int
        Frame height in pixels.
    frame_w : int
        Frame width in pixels.

    Returns
    -------
    tuple[int, int, int, int] | None
        ``(y1, y2, x1, x2)`` pixel coordinates, or *None* if the wrist
        landmark has low visibility or pose landmarks are unavailable.
    """
    if not pose_landmarks:
        return None

    lms = pose_landmarks
    wrist = lms[wrist_idx]

    if getattr(wrist, "visibility", 1.0) < 0.5:
        return None

    wx = wrist.x * frame_w
    wy = wrist.y * frame_h

    elbow = lms[elbow_idx]
    elbow_dist = np.hypot(
        (wrist.x - elbow.x) * frame_w,
        (wrist.y - elbow.y) * frame_h,
    )
    half = max(elbow_dist * _HAND_CROP_SCALE, _MIN_HAND_CROP_HALF)

    # Offset centre toward fingers so the crop covers the full hand.
    pinky = lms[pinky_idx]
    index = lms[index_idx]
    finger_cx = ((pinky.x + index.x) / 2) * frame_w
    finger_cy = ((pinky.y + index.y) / 2) * frame_h
    dx = finger_cx - wx
    dy = finger_cy - wy
    norm = np.hypot(dx, dy) + 1e-6
    cx = wx + 0.3 * half * dx / norm
    cy = wy + 0.3 * half * dy / norm

    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(frame_w, int(cx + half))
    y2 = min(frame_h, int(cy + half))

    if (x2 - x1) < 50 or (y2 - y1) < 50:
        return None
    return y1, y2, x1, x2


def _remap_hand_landmarks(hand_result, crop_region, frame_h: int, frame_w: int):
    """Remap hand landmark coordinates from a crop back to the full frame.

    Analogous to :func:`_remap_face_landmarks`.  Hand landmarks returned by
    the HandLandmarker are normalised to the crop image; this function
    converts them to be normalised w.r.t. the original full-size frame.
    """
    if not hand_result.hand_landmarks:
        return hand_result

    y1, y2, x1, x2 = crop_region
    crop_w = x2 - x1
    crop_h = y2 - y1

    for hand_lms in hand_result.hand_landmarks:
        for lm in hand_lms:
            lm.x = (lm.x * crop_w + x1) / frame_w
            lm.y = (lm.y * crop_h + y1) / frame_h
            lm.z = lm.z * crop_w / frame_w

    return hand_result


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def _detected_hand_labels(hand_result) -> set[str]:
    """Return the set of detected hand labels (``{"left", "right"}``)."""
    labels: set[str] = set()
    if hand_result.hand_landmarks and hand_result.handedness:
        for handedness_list in hand_result.handedness:
            h = handedness_list[0]
            label = getattr(h, "category_name", None) or getattr(
                h, "display_name", ""
            )
            labels.add(label.lower())
    return labels


def _try_hand_crop_detect(
    rgb: np.ndarray,
    pose_lms,
    hand_crop_lm,
    wrist_idx: int,
    elbow_idx: int,
    pinky_idx: int,
    index_idx: int,
    frame_h: int,
    frame_w: int,
):
    """Attempt pose-guided hand detection on a cropped (and optionally upscaled) region.

    Returns
    -------
    hand_result or None
        The detection result (with coordinates remapped to the full frame),
        or *None* if no crop could be computed or no hand was found.
    """
    crop_region = _hand_crop_region(
        pose_lms, wrist_idx, elbow_idx, pinky_idx, index_idx, frame_h, frame_w,
    )
    if crop_region is None:
        return None

    y1, y2, x1, x2 = crop_region
    hand_crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])

    crop_h, crop_w = hand_crop.shape[:2]
    if min(crop_h, crop_w) < _HAND_UPSCALE_MIN:
        scale = _HAND_UPSCALE_MIN / min(crop_h, crop_w)
        hand_crop = cv2.resize(
            hand_crop,
            (int(crop_w * scale), int(crop_h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=hand_crop)
    crop_result = hand_crop_lm.detect(mp_crop)

    if not crop_result.hand_landmarks:
        return None

    crop_result = _remap_hand_landmarks(crop_result, crop_region, frame_h, frame_w)
    return crop_result


def _merge_hand_results(original, fallback_results: list):
    """Merge fallback hand detections into the original hand result.

    Appends any newly detected hands (from crop-based fallback) to the
    original result's ``hand_landmarks`` and ``handedness`` lists.
    """
    for fb in fallback_results:
        if fb is None or not fb.hand_landmarks:
            continue
        original.hand_landmarks.extend(fb.hand_landmarks)
        original.handedness.extend(fb.handedness)
    return original


def detect_landmarks(frame, face_lm, hand_lm, hand_crop_lm, pose_lm, timestamp_ms: int):
    """Run all landmarkers on a single BGR frame.

    **Pose-guided face detection** – BosphorusSign22k videos are full-body
    shots in which the face occupies a small portion of the 1920x1080 frame.
    The FaceLandmarker's internal short-range face detector consistently fails
    on such frames.  To work around this, we:

    1. Run the PoseLandmarker first (it handles full-body frames well).
    2. Use the detected nose / eye / ear landmarks to crop a generous region
       around the face.
    3. Run the FaceLandmarker (IMAGE mode) on the crop.
    4. Remap the resulting face landmarks back to full-frame coordinates.

    **Pose-guided hand fallback** – similarly, when the full-frame
    HandLandmarker fails to detect one or both hands, we crop around the
    wrist position obtained from the pose landmarks, optionally upscale the
    crop, and re-run a second HandLandmarker (IMAGE mode) on the crop.

    Parameters
    ----------
    frame : np.ndarray
        Input frame in BGR format (OpenCV default).
    face_lm : FaceLandmarker
        Initialized face landmarker (IMAGE mode).
    hand_lm : HandLandmarker
        Initialized hand landmarker (VIDEO mode).
    hand_crop_lm : HandLandmarker
        Initialized hand landmarker (IMAGE mode) for pose-guided crop fallback.
    pose_lm : PoseLandmarker
        Initialized pose landmarker (VIDEO mode).
    timestamp_ms : int
        Timestamp in milliseconds for VIDEO mode temporal tracking.

    Returns
    -------
    tuple
        (face_result, hand_result, pose_result) containing detection results
        from all landmarkers.
    """
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # 1. Pose & hands on the full frame (VIDEO mode – needs timestamp).
    pose_result = pose_lm.detect_for_video(mp_image, timestamp_ms)
    hand_result = hand_lm.detect_for_video(mp_image, timestamp_ms)

    # 2. Face detection on a pose-guided crop (IMAGE mode – no timestamp).
    pose_lms = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
    crop_region = _face_crop_region(pose_lms, h, w)

    if crop_region is not None:
        y1, y2, x1, x2 = crop_region
        face_crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])
        mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop)
        face_result = face_lm.detect(mp_crop)
        face_result = _remap_face_landmarks(face_result, crop_region, h, w)
    else:
        # Fallback: try the full frame (covers close-up videos or missing pose data).
        face_result = face_lm.detect(mp_image)

    # 3. Pose-guided hand fallback – if either hand was not detected on the
    #    full frame, try cropping around the wrist and re-running detection.
    if pose_lms is not None:
        detected = _detected_hand_labels(hand_result)
        fallbacks: list = []

        if "left" not in detected:
            fallbacks.append(
                _try_hand_crop_detect(
                    rgb, pose_lms, hand_crop_lm,
                    _WRIST_LEFT, _ELBOW_LEFT, _PINKY_LEFT, _INDEX_LEFT,
                    h, w,
                )
            )
        if "right" not in detected:
            fallbacks.append(
                _try_hand_crop_detect(
                    rgb, pose_lms, hand_crop_lm,
                    _WRIST_RIGHT, _ELBOW_RIGHT, _PINKY_RIGHT, _INDEX_RIGHT,
                    h, w,
                )
            )

        if fallbacks:
            hand_result = _merge_hand_results(hand_result, fallbacks)

    return face_result, hand_result, pose_result


def extract_keypoints(face_result, hand_result, pose_result) -> np.ndarray:
    """Flatten landmark results into a single 1-D numpy vector.

    Only the face landmarks listed in ``FACE_LANDMARK_INDICES`` (83 points) are
    kept; the remaining face-mesh points are discarded to reduce noise.

    Layout: ``[pose(33*4), face(83*3), left_hand(21*3), right_hand(21*3)]``
    Total dimensions: 132 + 249 + 63 + 63 = 507

    Parameters
    ----------
    face_result : FaceLandmarkerResult
        Face detection result from MediaPipe.
    hand_result : HandLandmarkerResult
        Hand detection result from MediaPipe.
    pose_result : PoseLandmarkerResult
        Pose detection result from MediaPipe.

    Returns
    -------
    np.ndarray
        1D numpy array of shape (507,) containing all landmark coordinates.
    """
    pose_landmarks = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else []
    face_landmarks = face_result.face_landmarks[0] if face_result.face_landmarks else []

    left_landmarks: list = []
    right_landmarks: list = []
    if hand_result.hand_landmarks and hand_result.handedness:
        for landmarks, handedness_list in zip(
            hand_result.hand_landmarks, hand_result.handedness
        ):
            handedness = handedness_list[0]
            label = getattr(handedness, "category_name", None) or getattr(
                handedness, "display_name", ""
            )
            if label.lower() == "left":
                left_landmarks = landmarks
            elif label.lower() == "right":
                right_landmarks = landmarks

    pose = (
        np.array(
            [[lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.0)] for lm in pose_landmarks]
        ).flatten()
        if pose_landmarks
        else np.zeros(POSE_LANDMARKS * 4)
    )
    if face_landmarks:
        face_all = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks])
        face = face_all[list(FACE_LANDMARK_INDICES)].flatten()
    else:
        face = np.zeros(FACE_LANDMARKS * 3)
    lh = (
        np.array([[lm.x, lm.y, lm.z] for lm in left_landmarks]).flatten()
        if left_landmarks
        else np.zeros(HAND_LANDMARKS * 3)
    )
    rh = (
        np.array([[lm.x, lm.y, lm.z] for lm in right_landmarks]).flatten()
        if right_landmarks
        else np.zeros(HAND_LANDMARKS * 3)
    )
    return np.concatenate([pose, face, lh, rh])
