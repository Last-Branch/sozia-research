"""
Extract keypoints from sign-language videos.

Walks the dataset's raw video directory (layout defined by the active
:class:`DatasetInfo`), runs MediaPipe face/hand/pose landmarkers in VIDEO
mode, optionally interpolates missing landmarks, and saves per-video
``.npy`` keypoint arrays under the dataset's processed directory.
Supports resumption via a JSON progress log.

Typical usage::

    python -m tsl_recognition.extraction                       # BosphorusSign22k (default)
    python -m tsl_recognition.extraction --dataset autsl       # AUTSL
    python -m tsl_recognition.extraction --test                # quick 10-class smoke test
"""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from ..config import MP_MODEL_DIR, TrainConfig
from ..dataset.interpolation import EXPECTED_DIM, interpolate_missing_keypoints
from .landmarks import create_landmarkers, detect_landmarks, extract_keypoints


def process_video(
    video_path: Path,
    face_lm,
    hand_lm,
    hand_crop_lm,
    pose_lm,
    start_timestamp: int = 0,
) -> tuple[list[np.ndarray] | None, int]:
    """Extract per-frame keypoints from a single video using VIDEO mode.

    This function processes each frame of a video, extracts landmarks using
    MediaPipe, and converts them to keypoint vectors.

    Parameters
    ----------
    video_path : Path
        Path to the video file to process.
    face_lm : FaceLandmarker
        Initialized face landmarker.
    hand_lm : HandLandmarker
        Initialized hand landmarker (VIDEO mode, full frame).
    hand_crop_lm : HandLandmarker
        Initialized hand landmarker (IMAGE mode, pose-guided crop fallback).
    pose_lm : PoseLandmarker
        Initialized pose landmarker.
    start_timestamp : int, optional
        Starting timestamp in milliseconds for VIDEO mode (default: 0).
        Important for maintaining temporal consistency across multiple videos.

    Returns
    -------
    tuple[list[np.ndarray] | None, int]
        - List of keypoint arrays (one per frame), or None if video can't be opened
        - End timestamp in milliseconds (for next video's start_timestamp)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, start_timestamp

    keypoints_sequence: list[np.ndarray] = []
    timestamp_ms = start_timestamp

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms += 33  # ~30 fps

        face_result, hand_result, pose_result = detect_landmarks(
            frame, face_lm, hand_lm, hand_crop_lm, pose_lm, timestamp_ms
        )

        keypoints = extract_keypoints(face_result, hand_result, pose_result)
        keypoints_sequence.append(keypoints)

    cap.release()
    return (keypoints_sequence if keypoints_sequence else None), timestamp_ms


# ---------------------------------------------------------------------------
# Multiprocessing helpers
# ---------------------------------------------------------------------------
# Per-worker state populated once by _init_worker, then reused by every
# _process_one_video call dispatched to that worker.
_wk: dict = {}


def _init_worker(model_dir_str: str, apply_interpolation: bool) -> None:
    """Create MediaPipe landmarkers once per worker process.

    Called automatically by ``Pool(initializer=...)``.  The landmarkers
    are stored in the module-global ``_wk`` dict so
    ``_process_one_video`` can reuse them for every video dispatched to
    this worker.  When the worker process exits the OS reclaims all
    resources.
    """
    import atexit
    from contextlib import ExitStack

    global _wk
    stack = ExitStack()
    face_lm, hand_lm, hand_crop_lm, pose_lm = stack.enter_context(
        create_landmarkers(model_dir_str)
    )
    _wk = {
        "face_lm": face_lm,
        "hand_lm": hand_lm,
        "hand_crop_lm": hand_crop_lm,
        "pose_lm": pose_lm,
        "ts": 0,
        "apply_interp": apply_interpolation,
        "_stack": stack,
    }
    atexit.register(stack.close)


def _process_one_video(
    task: tuple[str, Path, Path, str],
) -> tuple[str, str, int]:
    """Process a single video inside a worker process.

    Parameters
    ----------
    task : tuple
        ``(sign_class, video_file, output_path, video_key)``

    Returns
    -------
    tuple[str, str, int]
        ``(status, video_key, frame_count)`` where *status* is
        ``"ok"`` or ``"failed"``.
    """
    global _wk
    _sign_class, video_file, output_path, video_key = task

    kp_seq, _wk["ts"] = process_video(
        video_file,
        _wk["face_lm"],
        _wk["hand_lm"],
        _wk["hand_crop_lm"],
        _wk["pose_lm"],
        _wk["ts"],
    )

    if kp_seq is None or len(kp_seq) == 0:
        return ("failed", video_key, 0)

    arr = np.array(kp_seq, dtype=np.float32)
    if _wk["apply_interp"] and arr.shape[1] == EXPECTED_DIM:
        try:
            arr = interpolate_missing_keypoints(arr).astype(np.float32)
        except Exception as e:
            import warnings
            warnings.warn(f"Interpolation failed for {video_key}: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    return ("ok", video_key, len(kp_seq))


# ---------------------------------------------------------------------------
# Main extraction routine
# ---------------------------------------------------------------------------
def run_extraction(
    cfg: TrainConfig | None = None,
    *,
    num_workers: int = 1,
) -> dict[str, int]:
    """Process all videos and save keypoints.

    The raw video layout and output paths are determined by the active
    :class:`DatasetInfo` (selected via ``cfg.dataset``).

    Parameters
    ----------
    cfg : TrainConfig, optional
        If *None* the full training config is used.
    num_workers : int, optional
        Number of parallel worker processes (default ``1`` = sequential).
        Each worker spawns its own set of MediaPipe landmarkers (~300 MB RAM).

    Returns
    -------
    dict
        Statistics: ``processed``, ``failed``, ``total_frames``.
    """
    if cfg is None:
        cfg = TrainConfig.full()

    ds_info = cfg.dataset_info
    processed_dir = ds_info.processed_dir
    actions = cfg.actions

    for action in actions:
        (processed_dir / action).mkdir(parents=True, exist_ok=True)

    processed_log_path = processed_dir / "processing_log.json"
    if processed_log_path.exists():
        with open(processed_log_path) as f:
            processed_videos: set[str] = set(json.load(f))
        print(f"Resuming: {len(processed_videos)} videos already processed")
    else:
        processed_videos = set()

    tasks: list[tuple[str, Path, Path, str]] = []
    classes_list = list(actions)
    for sample_id, class_name, video_path in ds_info.iter_raw_videos(classes_list):
        output_path = ds_info.output_npy_path(class_name, sample_id)
        video_key = f"{class_name}/{sample_id}"
        if video_key in processed_videos and output_path.exists():
            continue
        tasks.append((class_name, video_path, output_path, video_key))

    print(f"Processing {len(actions)} sign classes from {ds_info.display_name}...")
    print(f"Videos to process: {len(tasks)} (skipping already done)")
    print("Using VIDEO mode for better temporal landmark tracking")
    if num_workers > 1:
        print(f"Workers: {num_workers} (parallel multiprocessing)")
    print(f"Output directory: {processed_dir}\n")

    stats: dict[str, int] = {"processed": 0, "failed": 0, "total_frames": 0}

    # ------------------------------------------------------------------
    # Parallel path: per-video dispatch with persistent worker landmarkers
    # ------------------------------------------------------------------
    if tasks and num_workers > 1:
        print(
            f"Dispatching {len(tasks)} videos across "
            f"{num_workers} workers...\n"
        )

        pbar = tqdm(
            total=len(tasks),
            desc=f"Extracting ({num_workers} workers)",
        )
        with Pool(
            num_workers,
            initializer=_init_worker,
            initargs=(str(MP_MODEL_DIR), cfg.apply_interpolation),
        ) as pool:
            for status, video_key, n_frames in pool.imap_unordered(
                _process_one_video, tasks
            ):
                if status == "ok":
                    stats["processed"] += 1
                    stats["total_frames"] += n_frames
                    processed_videos.add(video_key)
                else:
                    stats["failed"] += 1

                pbar.update(1)
                pbar.set_postfix(
                    done=stats["processed"],
                    failed=stats["failed"],
                    frames=stats["total_frames"],
                )

                if (stats["processed"] + stats["failed"]) % 50 == 0:
                    with open(processed_log_path, "w") as f:
                        json.dump(list(processed_videos), f)
        pbar.close()

    # ------------------------------------------------------------------
    # Sequential path: single-process (num_workers == 1)
    # ------------------------------------------------------------------
    elif tasks:
        global_timestamp = 0
        with create_landmarkers(MP_MODEL_DIR) as (face_lm, hand_lm, hand_crop_lm, pose_lm):
            pbar = tqdm(tasks, desc="Processing videos")
            for sign_class, video_file, output_path, video_key in pbar:
                keypoints_seq, global_timestamp = process_video(
                    video_file, face_lm, hand_lm, hand_crop_lm, pose_lm, global_timestamp
                )

                if keypoints_seq is None or len(keypoints_seq) == 0:
                    stats["failed"] += 1
                    continue

                keypoints_array = np.array(keypoints_seq, dtype=np.float32)

                if cfg.apply_interpolation and keypoints_array.shape[1] == EXPECTED_DIM:
                    try:
                        keypoints_array = interpolate_missing_keypoints(
                            keypoints_array
                        ).astype(np.float32)
                    except Exception as e:
                        import warnings
                        warnings.warn(f"Interpolation failed for {video_key}: {e}")

                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, keypoints_array)

                stats["processed"] += 1
                stats["total_frames"] += len(keypoints_seq)
                processed_videos.add(video_key)

                pbar.set_postfix(
                    done=stats["processed"],
                    failed=stats["failed"],
                    frames=stats["total_frames"],
                )

                if stats["processed"] % 50 == 0:
                    with open(processed_log_path, "w") as f:
                        json.dump(list(processed_videos), f)

    with open(processed_log_path, "w") as f:
        json.dump(list(processed_videos), f)

    print("\nProcessing complete!")
    print(f"  Processed: {stats['processed']} videos")
    print(f"  Failed: {stats['failed']} videos")
    print(f"  Total frames extracted: {stats['total_frames']}")
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from ..dataset.registry import DATASET_CHOICES

    parser = argparse.ArgumentParser(
        description="Extract keypoints from sign-language videos"
    )
    parser.add_argument(
        "--dataset", choices=DATASET_CHOICES, default="bosphorus",
        help="Dataset to extract from (default: bosphorus)",
    )
    parser.add_argument("--test", action="store_true", help="Use 10-class test config")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1 = sequential)",
    )
    args = parser.parse_args()

    config = (
        TrainConfig.test(dataset=args.dataset)
        if args.test
        else TrainConfig.full(dataset=args.dataset)
    )
    run_extraction(config, num_workers=args.num_workers)
