from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch.nn as nn


@dataclass
class HeadConfig:
    in_dim: int
    hidden_dim: int = 128
    dropout: float = 0.1


class ClassificationHead(nn.Module):
    def __init__(self, config: HeadConfig, out_dim: int, activation: str = "sigmoid") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.in_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, out_dim),
        )
        self.activation = activation

    def forward(self, x):
        logits = self.net(x)
        return logits


def build_task_heads(d_model: int) -> Dict[str, nn.Module]:
    head_cfg = HeadConfig(in_dim=d_model)
    return nn.ModuleDict(
        {
            "buy": ClassificationHead(head_cfg, out_dim=1, activation="sigmoid"),
            "sell": ClassificationHead(head_cfg, out_dim=1, activation="sigmoid"),
            "direction": ClassificationHead(head_cfg, out_dim=3, activation="softmax"),
            "regime": ClassificationHead(head_cfg, out_dim=6, activation="softmax"),
        }
    )
