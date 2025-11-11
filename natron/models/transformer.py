"""Transformer architecture definitions for Natron."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from natron.config import ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), dim=-1)


class NatronTransformer(nn.Module):
    """Transformer backbone with multi-task heads."""

    def __init__(self, feature_dim: int, config: ModelConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(feature_dim, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model) if config.use_positional_encoding else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation=config.activation,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.d_model)

        hidden_dim = config.d_model
        head_hidden = hidden_dim // 2

        def head():
            return nn.Sequential(
                nn.Linear(hidden_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(head_hidden, 1),
            )

        def head_multiclass(classes: int):
            return nn.Sequential(
                nn.Linear(hidden_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(head_hidden, classes),
            )

        self.buy_head = head()
        self.sell_head = head()
        self.direction_head = head_multiclass(3)
        self.regime_head = head_multiclass(6)

        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.projection_head = ProjectionHead(hidden_dim, config.projection_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        hidden = self.encoder(x)
        hidden = self.layer_norm(hidden)
        pooled = hidden.mean(dim=1)
        pooled = self.dropout(pooled)
        return hidden, pooled

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden, pooled = self.encode(x)
        outputs = {
            "hidden": hidden,
            "pooled": pooled,
            "buy": self.buy_head(pooled),
            "sell": self.sell_head(pooled),
            "direction": self.direction_head(pooled),
            "regime": self.regime_head(pooled),
        }
        return outputs

    def forward_pretraining(
        self,
        masked_inputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        hidden, _ = self.encode(masked_inputs)
        reconstructed = self.reconstruction_head(hidden)
        return {"reconstruction": reconstructed, "hidden": hidden, "mask": mask}

    def project(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.projection_head(pooled)
