from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class MaskingConfig:
    mask_prob: float = 0.15


def mask_sequence(
    x: torch.Tensor,
    mask_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len, feat = x.size()
    mask = torch.rand(batch, seq_len, device=x.device) < mask_prob
    masked_x = x.clone().masked_fill(mask.unsqueeze(-1), 0.0)
    return masked_x, mask, x


def masked_mse_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(reconstructed, target, reduction="mean")


def info_nce_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    batch = z_i.shape[0]
    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)
    labels = torch.arange(batch, device=z_i.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(2 * batch, dtype=torch.bool, device=z_i.device)
    similarity_matrix = similarity_matrix / temperature
    similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

    loss_i = nn.functional.cross_entropy(similarity_matrix[:batch], labels)
    loss_j = nn.functional.cross_entropy(similarity_matrix[batch:], labels)
    return (loss_i + loss_j) / 2
