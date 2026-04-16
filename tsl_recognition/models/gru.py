"""
ActionGRU — GRU-based sign language classifier.

GRU cells have fewer gates than LSTM (reset + update vs. input/forget/output),
making them lighter and potentially faster to train while retaining competitive
accuracy on sequence tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import SignClassifier


class ActionGRU(SignClassifier):
    """GRU classifier with 4–6 stacked layers.

    Uses the same SignClassifier interface with GRU cells. At least four
    layers are used to give the model sufficient depth.
    """

    SIZES = {
        "small": {"gru_hidden": 256, "gru_layers": 4, "fc": [512, 256]},
        "large": {"gru_hidden": 512, "gru_layers": 5, "fc": [1024, 512]},
        "xlarge": {"gru_hidden": 1024, "gru_layers": 6, "fc": [2048, 1024]},
    }

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        model_size: str = "small",
        dropout: float = 0.4,
    ):
        super().__init__(model_size=model_size)
        cfg = self._get_config(model_size)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=cfg["gru_hidden"],
            num_layers=cfg["gru_layers"],
            batch_first=True,
            dropout=dropout if cfg["gru_layers"] > 1 else 0,
            bidirectional=False,
        )

        self.head = self._build_head(
            cfg["gru_hidden"],
            cfg["fc"],
            num_classes,
            dropout,
        )

    def _encode(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode via GRU, returning the hidden state at the last real frame."""
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths.cpu().clamp(min=1),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            gru_out, _ = self.gru(x)

        if lengths is not None:
            batch_idx = torch.arange(x.size(0), device=x.device)
            return gru_out[batch_idx, lengths.to(x.device) - 1, :]
        return gru_out[:, -1, :]
