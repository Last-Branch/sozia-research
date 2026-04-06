"""
Extraction subpackage — MediaPipe keypoint extraction from sign-language videos.

Key entry points:
    - :func:`run_extraction` — process a full dataset
    - :func:`process_video` — extract keypoints from a single video
    - :func:`create_landmarkers` — context manager for MediaPipe landmarkers
    - :func:`detect_landmarks` — run all landmarkers on a single frame
    - :func:`extract_keypoints` — flatten landmark results to a 1-D vector
"""

from .landmarks import create_landmarkers, detect_landmarks, ensure_model, extract_keypoints
from .pipeline import process_video, run_extraction

__all__ = [
    "create_landmarkers",
    "detect_landmarks",
    "ensure_model",
    "extract_keypoints",
    "process_video",
    "run_extraction",
]
