"""1D CNN binary distinguisher for blind 64-bit ciphertext-pair features."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn

INPUT_BITS = 64
WORD_CHANNELS = 4
WORD_LENGTH = 16


class BitPairReshape(nn.Module):
    """Map flat (B, 64) features to (B, 4, 16) word channels [L0, R0, L1, R1]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != INPUT_BITS:
            raise ValueError(f"expected last dim {INPUT_BITS}, got {x.shape}")
        return x.view(x.size(0), WORD_CHANNELS, WORD_LENGTH)


class CnnDistinguisher(nn.Module):
    """Conv1d stack over four 16-bit word channels; outputs one logit per sample."""

    def __init__(
        self,
        channels: Sequence[int] = (32, 64, 128),
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if len(channels) < 2 or len(channels) > 3:
            raise ValueError("channels must have length 2 or 3")
        self.reshape = BitPairReshape()
        in_ch = WORD_CHANNELS
        conv_layers: list[nn.Module] = []
        for out_ch in channels:
            conv_layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            in_ch = out_ch
        self.features = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B,) for input (B, 64) or (B, 4, 16)."""
        return self.forward_logits(x).squeeze(-1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2 and x.shape[-1] == INPUT_BITS:
            x = self.reshape(x)
        elif not (x.dim() == 3 and x.shape[1:] == (WORD_CHANNELS, WORD_LENGTH)):
            raise ValueError(
                f"expected (B, {INPUT_BITS}) or (B, {WORD_CHANNELS}, {WORD_LENGTH}), got {tuple(x.shape)}"
            )
        h = self.features(x)
        h = self.pool(h).squeeze(-1)
        return self.classifier(h)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid probabilities for class 1 (real differential pair)."""
        self.eval()
        logits = self.forward_logits(
            x if x.dim() > 1 else x.unsqueeze(0)
        ).squeeze(-1)
        return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(channels: Iterable[int] = (32, 64, 128)) -> CnnDistinguisher:
    return CnnDistinguisher(channels=tuple(channels))
