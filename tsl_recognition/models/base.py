"""
SignClassifier — shared base class for all sign language classifiers.

Provides:
    - ``SIZES`` class attribute (each subclass overrides with its own presets)
    - ``_get_config(model_size)`` helper that falls back to ``"small"``
    - ``_build_head(...)`` to construct the shared FC classification head
    - ``_classify(x)`` to run the head
    - Abstract ``_encode(x, lengths)`` that each subclass implements
    - Concrete ``forward(x, lengths)`` that calls ``_encode`` then ``_classify``
"""

from __future__ import annotations

import abc
from typing import ClassVar

import torch
import torch.nn as nn


class SignClassifier(nn.Module, abc.ABC):
    """Abstract base for sign language classifiers.

    Subclasses must:
      1. Override ``SIZES`` with their own architecture presets.
      2. Build encoder layers in ``__init__`` and call
         ``self.head = self._build_head(...)`` to create the shared FC head.
      3. Implement ``_encode(x, lengths)`` returning a ``(batch, features)``
         tensor that feeds into the classification head.
    """

    SIZES: ClassVar[dict[str, dict]] = {}

    def __init__(self, model_size: str = "small") -> None:
        super().__init__()
        self.model_size = model_size

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _get_config(cls, model_size: str) -> dict:
        """Return the config dict for *model_size*, falling back to ``"small"``."""
        return cls.SIZES.get(model_size, cls.SIZES["small"])

    @staticmethod
    def _build_head(
        in_features: int,
        fc_sizes: list[int],
        num_classes: int,
        dropout: float,
    ) -> nn.Sequential:
        """Build the shared FC classification head.

        Architecture::

            fc1 → bn1 → relu → drop → fc2 → bn2 → relu → drop → fc3
        """
        return nn.Sequential(
            nn.Linear(in_features, fc_sizes[0]),
            nn.BatchNorm1d(fc_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_sizes[0], fc_sizes[1]),
            nn.BatchNorm1d(fc_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_sizes[1], num_classes),
        )

    # ------------------------------------------------------------------
    # Forward pipeline
    # ------------------------------------------------------------------

    def _classify(self, x: torch.Tensor) -> torch.Tensor:
        """Run the classification head on encoder output."""
        return self.head(x)

    @abc.abstractmethod
    def _encode(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode an input sequence to a fixed-size representation.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, seq_len, features)``.
        lengths : torch.Tensor, optional
            Actual (non-padded) sequence lengths, shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Encoded representation of shape ``(batch, encoder_out_features)``.
        """
        ...

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: encode the sequence then classify.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, seq_len, features)``.
        lengths : torch.Tensor, optional
            Actual (non-padded) sequence lengths, shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_classes)``.
        """
        return self._classify(self._encode(x, lengths))
