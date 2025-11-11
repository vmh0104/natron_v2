"""Natron Transformer architecture for multi-task financial modeling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class NatronModelConfig:
    feature_dim: int = 100
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    buy_head_dim: int = 1
    sell_head_dim: int = 1
    direction_classes: int = 3
    regime_classes: int = 6


class NatronEncoder(nn.Module):
    """Shared Transformer encoder used across training phases."""

    def __init__(self, config: NatronModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.feature_dim, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = True,
    ) -> torch.Tensor:
        """Encode a sequence and optionally return the full sequence of embeddings."""
        x = self.input_projection(x)
        encoded = self.encoder(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)
        if return_sequence:
            return encoded
        return encoded[:, -1]


class NatronTransformer(nn.Module):
    """Transformer encoder with multi-task output heads."""

    def __init__(self, config: NatronModelConfig, encoder: Optional[NatronEncoder] = None) -> None:
        super().__init__()
        self.config = config
        self.encoder = encoder or NatronEncoder(config)

        self.buy_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.buy_head_dim),
        )
        self.sell_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.sell_head_dim),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.direction_classes),
        )
        self.regime_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.regime_classes),
        )

    def encode(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return sequence embeddings from the shared encoder."""
        return self.encoder(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, return_sequence=True)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning logits for each prediction head."""
        encoded = self.encode(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        pooled = encoded[:, -1]

        buy_logits = self.buy_head(pooled)
        sell_logits = self.sell_head(pooled)
        direction_logits = self.direction_head(pooled)
        regime_logits = self.regime_head(pooled)

        if return_hidden:
            return buy_logits, sell_logits, direction_logits, regime_logits, encoded
        return buy_logits, sell_logits, direction_logits, regime_logits

    def get_encoder(self) -> NatronEncoder:
        """Return the underlying encoder module."""
        return self.encoder


__all__ = ["NatronTransformer", "NatronEncoder", "NatronModelConfig"]
