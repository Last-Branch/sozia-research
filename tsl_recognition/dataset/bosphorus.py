"""
BosphorusSign22k dataset configuration.

Raw layout::

    data/BosphorusSign22k/raw/{ClassID}/{User_X_NNN}.mp4
    data/BosphorusSign22k/processed/{ClassName_tr}/{User_X_NNN}.npy

Raw directories use numeric ClassIDs (e.g., ``0001/``).  Turkish class
names are resolved from ``BosphorusSign22k_classes.csv`` (ClassID →
ClassName_tr).  Splits are computed by signer ID (``User_2`` through
``User_7``).
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from pathlib import Path

from .base import DatasetInfo

# Regex to extract signer ID from filenames like "User_2_001.npy"
_SIGNER_RE = re.compile(r"^(User_\d+)_")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

_CLASS_CSV = "BosphorusSign22k_classes.csv"


class BosphorusSign22kInfo(DatasetInfo):
    """Dataset info for the BosphorusSign22k corpus."""

    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir / "data" / "BosphorusSign22k"
        self._class_map: dict[str, str] | None = None  # ClassID → ClassName_tr

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

    # -- CSV helpers -------------------------------------------------------
    def _load_class_map(self) -> dict[str, str]:
        """Load ClassID → ClassName_tr from BosphorusSign22k_classes.csv."""
        if self._class_map is not None:
            return self._class_map
        csv_path = self._base / _CLASS_CSV
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{_CLASS_CSV} not found at {csv_path}. "
                f"See the 'Data setup' section of the README."
            )
        self._class_map = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["ClassName_tr"]
                if name in self._class_map.values():
                    raise ValueError(
                        f"Duplicate ClassName_tr {name!r} in {csv_path} "
                        f"(ClassIDs collide on the same processed/ directory)"
                    )
                self._class_map[row["ClassID"]] = name
        return self._class_map

    # -- classes -----------------------------------------------------------
    def class_names(self) -> list[str]:
        """Return Turkish class names sorted by ClassID."""
        class_map = self._load_class_map()
        return [class_map[cid] for cid in sorted(class_map.keys())]

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
    def iter_raw_videos(
        self, classes: list[str] | None = None
    ) -> Iterator[tuple[str, str, Path]]:
        """Yield (sample_id, ClassName_tr, video_path) for all raw videos.

        Raw directories use numeric ClassIDs (``0001/``, ``0002/``, …);
        this method maps them to Turkish class names via the classes CSV.
        """
        class_map = self._load_class_map()  # ClassID → ClassName_tr
        allowed = set(classes) if classes is not None else None

        for class_id, class_name in sorted(class_map.items()):
            if allowed is not None and class_name not in allowed:
                continue
            class_dir = self.raw_dir / class_id
            if not class_dir.exists():
                continue
            for video_path in sorted(class_dir.iterdir()):
                if (
                    video_path.is_file()
                    and video_path.suffix.lower() in VIDEO_EXTENSIONS
                ):
                    yield video_path.stem, class_name, video_path
