"""
BosphorusSign22k dataset configuration.

Raw layout::

    data/BosphorusSign22k/raw/{class_name}/{User_X_NNN}.mp4
    data/BosphorusSign22k/processed/{class_name}/{User_X_NNN}.npy

Classes are derived from subdirectory names. Splits are computed by
signer ID (``User_2`` through ``User_7``).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from .base import DatasetInfo

# Regex to extract signer ID from filenames like "User_2_001.npy"
_SIGNER_RE = re.compile(r"^(User_\d+)_")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class BosphorusSign22kInfo(DatasetInfo):
    """Dataset info for the BosphorusSign22k corpus."""

    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir / "data" / "BosphorusSign22k"

    # -- identity ----------------------------------------------------------
    @property
    def name(self) -> str:
        return "bosphorus"

    @property
    def display_name(self) -> str:
        return "BosphorusSign22k"

    # -- paths -------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self._base / "raw"

    @property
    def processed_dir(self) -> Path:
        return self._base / "processed"

    @property
    def split_dir(self) -> Path:
        return self._base / "split"

    # -- classes -----------------------------------------------------------
    def class_names(self) -> list[str]:
        for d in (self.raw_dir, self.processed_dir):
            if d.exists():
                names = sorted(p.name for p in d.iterdir() if p.is_dir())
                if names:
                    return names
        return []

    # -- splitting ---------------------------------------------------------
    @property
    def default_split_mode(self) -> str:
        return "signer"

    def signer_split_map(self) -> dict[str, list[str]]:
        return {
            "train": ["User_3", "User_4", "User_5", "User_6"],
            "val": ["User_2"],
            "test": ["User_7"],
        }

    def extract_signer(self, filename: str) -> str | None:
        m = _SIGNER_RE.match(filename)
        return m.group(1) if m else None

    # -- extraction --------------------------------------------------------
    def iter_raw_videos(self, classes: list[str] | None = None) -> Iterator[tuple[str, str, Path]]:
        target_classes = classes if classes is not None else self.class_names()
        for class_name in sorted(target_classes):
            class_dir = self.raw_dir / class_name
            if not class_dir.exists():
                continue
            for video_path in sorted(class_dir.iterdir()):
                if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                    yield video_path.stem, class_name, video_path
