"""Pretraining module for Natron masked modeling and contrastive learning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from natron.models.transformer import NatronEncoder, NatronModelConfig


@dataclass
class PretrainingConfig:
    masking_ratio: float = 0.15
    projection_dim: int = 128
    temperature: float = 0.1


class NatronPretrainingModel(nn.Module):
    """Wraps the shared encoder with reconstruction and contrastive heads."""

    def __init__(self, encoder: NatronEncoder, pretrain_cfg: PretrainingConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.pretrain_cfg = pretrain_cfg

        feature_dim = encoder.config.feature_dim
        d_model = encoder.config.d_model

        self.mask_token = nn.Parameter(torch.zeros(feature_dim))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        self.reconstruction_head = nn.Linear(d_model, feature_dim)
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(encoder.config.dropout),
            nn.Linear(d_model, pretrain_cfg.projection_dim),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstructed tokens, normalized projections, and hidden states."""
        masked_inputs = self.apply_mask(inputs, mask)
        hidden_seq = self.encoder(masked_inputs, return_sequence=True)
        reconstructed = self.reconstruction_head(hidden_seq)

        pooled = hidden_seq[:, -1]
        proj_a = F.normalize(self.projector(pooled), dim=-1)
        proj_b = F.normalize(self.projector(pooled), dim=-1)

        return reconstructed, inputs, mask, (proj_a, proj_b)

    def apply_mask(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Replace masked positions with a learned mask token."""
        mask = mask.unsqueeze(-1)
        mask_token = self.mask_token.to(inputs.device)
        return torch.where(mask, mask_token, inputs)

    @staticmethod
    def reconstruction_loss(reconstructed: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        if mask.sum() == 0:
            return F.mse_loss(reconstructed, targets)
        return F.mse_loss(reconstructed[mask], targets[mask])

    def contrastive_loss(self, proj_pair: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        proj_a, proj_b = proj_pair
        temperature = self.pretrain_cfg.temperature
        logits = torch.matmul(proj_a, proj_b.t()) / temperature
        labels = torch.arange(proj_a.size(0), device=proj_a.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_a + loss_b)


__all__ = ["NatronPretrainingModel", "PretrainingConfig"]
