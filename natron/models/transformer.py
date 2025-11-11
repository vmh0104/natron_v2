from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class TransformerConfig:
    input_dim: int
    d_model: int = 256
    n_heads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    max_positions: int = 128
    classification_dropout: float = 0.1
    use_cls_token: bool = True


@dataclass(slots=True)
class TransformerOutput:
    sequence_output: torch.Tensor
    cls_output: torch.Tensor
    buy_logits: torch.Tensor
    sell_logits: torch.Tensor
    direction_logits: torch.Tensor
    regime_logits: torch.Tensor
    reconstruction: torch.Tensor | None = None
    projection: torch.Tensor | None = None


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class NatronTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.cfg = config

        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(config.d_model, max_len=config.max_positions)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.classification_dropout)

        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        else:
            self.register_parameter("cls_token", None)

        # Multi-task heads
        self.buy_head = nn.Linear(config.d_model, 1)
        self.sell_head = nn.Linear(config.d_model, 1)
        self.direction_head = nn.Linear(config.d_model, 3)
        self.regime_head = nn.Linear(config.d_model, 6)

        # Pre-training heads
        self.reconstruction_head = nn.Linear(config.d_model, config.input_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        for head in [self.buy_head, self.sell_head, self.direction_head, self.regime_head, self.reconstruction_head]:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> TransformerOutput:
        batch_size, seq_len, _ = inputs.shape

        x = self.input_proj(inputs)
        if self.cfg.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (1, 0), value=0)
        x = self.pos_encoder(x)

        if attention_mask is not None:
            # expect mask == 1 for valid positions. Convert to boolean padding mask.
            padding_mask = attention_mask == 0
        else:
            padding_mask = None

        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        encoded = self.norm(encoded)

        if self.cfg.use_cls_token:
            cls_output = encoded[:, 0]
            sequence_output = encoded[:, 1:]
        else:
            cls_output = encoded.mean(dim=1)
            sequence_output = encoded

        cls_output = self.dropout(cls_output)

        buy_logits = self.buy_head(cls_output).squeeze(-1)
        sell_logits = self.sell_head(cls_output).squeeze(-1)
        direction_logits = self.direction_head(cls_output)
        regime_logits = self.regime_head(cls_output)
        reconstruction = self.reconstruction_head(sequence_output) if return_sequence else None
        projection = F.normalize(self.projection_head(cls_output), dim=-1)

        return TransformerOutput(
            sequence_output=sequence_output,
            cls_output=cls_output,
            buy_logits=buy_logits,
            sell_logits=sell_logits,
            direction_logits=direction_logits,
            regime_logits=regime_logits,
            reconstruction=reconstruction,
            projection=projection,
        )

    def encode(self, inputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(inputs, attention_mask=attention_mask).cls_output
