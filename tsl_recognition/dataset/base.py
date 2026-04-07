"""
Abstract base class for dataset-specific configuration.

Every dataset supported by the TSL recognition pipeline implements
``DatasetInfo``. The base class defines the interface that extraction,
splitting, training, and evaluation depend on, so adding a new dataset
is just a matter of creating a subclass and registering it in registry.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path


class DatasetInfo(ABC):
    """Abstract base describing everything the pipeline needs to know about
    a dataset: paths, class labels, raw-video layout, and split strategy."""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used on the CLI (e.g. ``"bosphorus"``, ``"autsl"``)."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable dataset name (e.g. ``"BosphorusSign22k"``)."""

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def raw_dir(self) -> Path:
        """Root directory that contains the original (unprocessed) data."""

    @property
    @abstractmethod
    def processed_dir(self) -> Path:
        """Directory for extracted ``.npy`` keypoint files.

        Convention: ``processed/{class_name}/{sample}.npy`` so that
        ``scan_dataset`` works uniformly across datasets.
        """

    @property
    @abstractmethod
    def split_dir(self) -> Path:
        """Directory for persistent split manifest JSON files."""

    # ------------------------------------------------------------------
    # Class labels
    # ------------------------------------------------------------------
    @abstractmethod
    def class_names(self) -> list[str]:
        """Return the canonical sorted list of class names.

        For folder-organized datasets this is the list of subdirectory
        names; for CSV-labeled datasets it comes from the label file.
        """

    @property
    def num_classes(self) -> int:
        """Total number of classes in the dataset."""
        return len(self.class_names())

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def default_split_mode(self) -> str:
        """Default split strategy: ``"signer"``, ``"random"``, or ``"predefined"``."""

    def signer_split_map(self) -> dict[str, list[str]] | None:
        """Mapping of ``{"train": [...], "val": [...], "test": [...]}``
        with signer IDs for signer-independent splitting.

        Return *None* if signer-based splitting is not applicable.
        """
        return None

    def extract_signer(self, filename: str) -> str | None:
        """Extract the signer identifier from a ``.npy`` filename.

        Returns *None* if the filename does not encode a signer ID.
        """
        return None

    # ------------------------------------------------------------------
    # Predefined splits (for datasets shipped with official splits)
    # ------------------------------------------------------------------
    def predefined_split_entries(self) -> dict[str, list[dict]] | None:
        """Return pre-existing split assignments.

        Each value is a list of dicts with at least::

            {"sample_id": str, "class_name": str, "signer": str}

        Return *None* if the dataset does not ship with predefined splits.
        """
        return None

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    @abstractmethod
    def iter_raw_videos(self, classes: list[str] | None = None) -> Iterator[tuple[str, str, Path]]:
        """Yield ``(sample_id, class_name, video_path)`` for extraction.

        Parameters
        ----------
        classes : list[str] | None
            If given, restrict to these class names. *None* means all.
        """

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def output_npy_path(self, class_name: str, sample_id: str) -> Path:
        """Canonical path for a processed ``.npy`` file.

        Default: ``processed_dir / class_name / sample_id.npy``.
        Override if your dataset needs a different convention.
        """
        return self.processed_dir / class_name / f"{sample_id}.npy"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, classes={self.num_classes})"
