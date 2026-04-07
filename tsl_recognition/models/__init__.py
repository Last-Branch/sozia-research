"""
Model registry for the TSL recognition pipeline.

Provides ``SignClassifier`` (shared base class), ``MODEL_REGISTRY``
(name -> class mapping) and a ``build_model`` factory so the rest of the
codebase never needs to import individual architecture classes directly.
"""

from __future__ import annotations

import torch.nn as nn

from .base import SignClassifier as SignClassifier
from .gru import ActionGRU

# ---------------------------------------------------------------------------
# Registry: short CLI name  ->  model class
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "gru": ActionGRU,
}


def build_model(
    arch: str,
    input_size: int,
    num_classes: int,
    model_size: str = "small",
    dropout: float = 0.4,
) -> nn.Module:
    """Instantiate a model by its registry name.

    Parameters
    ----------
    arch : str
        Key in ``MODEL_REGISTRY`` (e.g. ``"gru"``).
    input_size : int
        Feature dimension per time-step.
    num_classes : int
        Number of output classes.
    model_size : str
        ``"small"``, ``"large"``, or ``"xlarge"`` preset.
    dropout : float
        Dropout probability.

    Returns
    -------
    nn.Module
        The constructed model, ready for ``.to(device)``.

    Raises
    ------
    ValueError
        If *arch* is not found in the registry.
    """
    if arch not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown architecture '{arch}'. Available: {available}"
        )
    cls = MODEL_REGISTRY[arch]
    return cls(
        input_size=input_size,
        num_classes=num_classes,
        model_size=model_size,
        dropout=dropout,
    )
