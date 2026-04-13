"""
AUTSL dataset configuration.

Raw layout::

    data/AUTSL/train/signer{N}_sample{M}_color.mp4
    data/AUTSL/val/signer{N}_sample{M}_color.mp4
    data/AUTSL/test/signer{N}_sample{M}_color.mp4

Labels come from CSV files::

    data/AUTSL/train_labels.csv           sample_name,class_id
    data/AUTSL/validation_labels.csv      sample_name,class_id
    data/AUTSL/test_labels.csv            sample_name,class_id
    data/AUTSL/SignList_ClassId_TR_EN.csv  ClassId,TR,EN

Processed keypoints follow the same class-folder convention as
BosphorusSign22k::

    data/AUTSL/processed/{class_name}/{signer_sample}.npy

Splits are predefined by the dataset authors (signer-disjoint).
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from pathlib import Path

from .base import DatasetInfo

_SIGNER_RE = re.compile(r"^(signer\d+)_")

_LABEL_CSV = {
    "train": "train_labels.csv",
    "val": "validation_labels.csv",
    "test": "test_labels.csv",
}

_SPLIT_SUBDIR = {
    "train": "train",
    "val": "val",
    "test": "test",
}


class AUTSLInfo(DatasetInfo):
    """Dataset info for the AUTSL (Turkish Sign Language) corpus."""

    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir / "data" / "AUTSL"
        self._class_map: dict[int, str] | None = None
        self._label_cache: dict[str, dict[str, int]] = {}

    # -- identity ----------------------------------------------------------
    @property
    def name(self) -> str:
        return "autsl"

    @property
    def display_name(self) -> str:
        return "AUTSL"

    # -- paths -------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self._base

    @property
    def processed_dir(self) -> Path:
        return self._base / "processed"

    @property
    def split_dir(self) -> Path:
        return self._base / "split"

    # -- internal CSV helpers ----------------------------------------------
    def _load_class_map(self) -> dict[int, str]:
        """Load ClassId -> class_name (Turkish) from SignList CSV."""
        if self._class_map is not None:
            return self._class_map
        csv_path = self._base / "SignList_ClassId_TR_EN.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"SignList_ClassId_TR_EN.csv not found at {csv_path}. "
                f"See the 'Data setup' section of the README."
            )
        self._class_map = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["TR"]
                if name in self._class_map.values():
                    raise ValueError(
                        f"Duplicate TR name {name!r} in {csv_path} "
                        f"(ClassIds collide on the same processed/ directory)"
                    )
                self._class_map[int(row["ClassId"])] = name
        return self._class_map

    def _load_split_labels(self, split: str) -> dict[str, int]:
        """Load sample_name -> class_id from a split's label CSV."""
        if split in self._label_cache:
            return self._label_cache[split]
        csv_path = self._base / _LABEL_CSV[split]
        labels: dict[str, int] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) == 2:
                    labels[parts[0]] = int(parts[1])
        self._label_cache[split] = labels
        return labels

    # -- classes -----------------------------------------------------------
    def class_names(self) -> list[str]:
        class_map = self._load_class_map()
        return [class_map[i] for i in sorted(class_map.keys())]

    # -- splitting ---------------------------------------------------------
    @property
    def default_split_mode(self) -> str:
        return "predefined"

    def extract_signer(self, filename: str) -> str | None:
        m = _SIGNER_RE.match(filename)
        return m.group(1) if m else None

    def predefined_split_entries(self) -> dict[str, list[dict]]:
        """Build split entries from the official AUTSL CSVs."""
        class_map = self._load_class_map()
        splits: dict[str, list[dict]] = {}

        for split in ("train", "val", "test"):
            labels = self._load_split_labels(split)
            entries: list[dict] = []
            for sample_name, class_id in labels.items():
                class_name = class_map[class_id]
                signer = self.extract_signer(sample_name)
                entries.append(
                    {
                        "sample_id": sample_name,
                        "class_name": class_name,
                        "class_id": class_id,
                        "signer": signer or "unknown",
                        "split_origin": split,
                    }
                )
            splits[split] = entries

        return splits

    # -- extraction --------------------------------------------------------
    def iter_raw_videos(
        self, classes: list[str] | None = None
    ) -> Iterator[tuple[str, str, Path]]:
        class_map = self._load_class_map()
        allowed_classes = set(classes) if classes is not None else None

        for split in ("train", "val", "test"):
            labels = self._load_split_labels(split)
            split_dir = self._base / _SPLIT_SUBDIR[split]
            if not split_dir.exists():
                continue
            for sample_name, class_id in labels.items():
                class_name = class_map[class_id]
                if allowed_classes is not None and class_name not in allowed_classes:
                    continue
                video_path = split_dir / f"{sample_name}_color.mp4"
                if video_path.exists():
                    yield sample_name, class_name, video_path
