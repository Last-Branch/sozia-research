"""
Lip-reading pipeline: face-landmark word classifier trained on TSL sign datasets.

The model uses the 249-dim face-landmark subset (dims 132..381) of the 507-dim
keypoint vectors already extracted by the TSL recognition pipeline.  Training
uses sign class names (Turkish words) as word-level supervision, exploiting the
natural mouthing behaviour of TSL signers.

Quick start::

    python -m lip_reading train                    # full run on BosphorusSign22k
    python -m lip_reading train --dataset autsl    # full run on AUTSL
    python -m lip_reading train --test             # 10-class smoke test
"""

from .config import FACE_SLICE, LIP_FEATURE_DIM, LipTrainConfig
from .dataset import LipReadingDataset, build_lip_loaders
from .train import train

__all__ = [
    "LipTrainConfig",
    "LIP_FEATURE_DIM",
    "FACE_SLICE",
    "LipReadingDataset",
    "build_lip_loaders",
    "train",
]
