"""
Dataset registry.

To add a new dataset:

1. Create ``tsl_recognition/dataset/my_dataset.py`` with a class that inherits
   :class:`DatasetInfo`.
2. Import it here and add it to ``DATASET_REGISTRY``.
3. That's it -- the CLI, extraction, splitting, and training pipelines
   will pick it up automatically via ``--dataset my_dataset``.
"""

from __future__ import annotations

from pathlib import Path

from .autsl import AUTSLInfo
from .base import DatasetInfo
from .bosphorus import BosphorusSign22kInfo

# Maps CLI name -> concrete DatasetInfo constructor.
# Each constructor receives ``base_dir: Path`` (the repository root).
DATASET_REGISTRY: dict[str, type[DatasetInfo]] = {
    "bosphorus": BosphorusSign22kInfo,
    "autsl": AUTSLInfo,
}

DATASET_CHOICES = sorted(DATASET_REGISTRY.keys())


def get_dataset_info(name: str, base_dir: Path) -> DatasetInfo:
    """Look up a dataset by CLI name and return an initialised instance.

    Parameters
    ----------
    name : str
        One of the keys in ``DATASET_REGISTRY``.
    base_dir : Path
        Repository root (passed to the ``DatasetInfo`` constructor).

    Raises
    ------
    ValueError
        If *name* is not found in the registry.
    """
    cls = DATASET_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown dataset: {name!r}. "
            f"Available: {DATASET_CHOICES}"
        )
    return cls(base_dir)
