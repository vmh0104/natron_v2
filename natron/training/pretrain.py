from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW

from ..models import NatronTransformer
from .losses import info_nce_loss, masked_mse_loss


@dataclass(slots=True)
class PretrainConfig:
    mask_ratio: float = 0.15
    mask_value: float = 0.0
    temperature: float = 0.07
    recon_weight: float = 1.0
    contrastive_weight: float = 0.2
    lr: float = 1e-4
    weight_decay: float = 1e-5


class NatronPretrainModule(pl.LightningModule):
    def __init__(self, model: NatronTransformer, config: PretrainConfig) -> None:
        super().__init__()
        self.model = model
        self.cfg = config
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  # pragma: no cover - Lightning handles
        return self.model(x, return_sequence=True).__dict__

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  # pragma: no cover - training loop side effect
        sequences, _ = batch
        sequences = sequences.float()
        view1, mask1 = self._apply_mask(sequences)
        view2, mask2 = self._apply_mask(sequences)

        out1 = self.model(view1, return_sequence=True)
        out2 = self.model(view2, return_sequence=True)
        if out1.reconstruction is None or out2.reconstruction is None or out1.projection is None or out2.projection is None:
            raise RuntimeError("Model must return reconstruction and projection during pretraining")

        mask1_expanded = mask1.unsqueeze(-1).expand_as(sequences)
        mask2_expanded = mask2.unsqueeze(-1).expand_as(sequences)
        recon_loss1 = masked_mse_loss(out1.reconstruction, sequences, mask1_expanded)
        recon_loss2 = masked_mse_loss(out2.reconstruction, sequences, mask2_expanded)
        recon_loss = 0.5 * (recon_loss1 + recon_loss2)

        contrastive_loss = info_nce_loss(out1.projection, out2.projection, temperature=self.cfg.temperature)
        loss = self.cfg.recon_weight * recon_loss + self.cfg.contrastive_weight * contrastive_loss

        self.log("train/recon_loss", recon_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/contrastive_loss", contrastive_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  # pragma: no cover
        sequences, _ = batch
        sequences = sequences.float()
        view1, mask1 = self._apply_mask(sequences)
        view2, mask2 = self._apply_mask(sequences)

        out1 = self.model(view1, return_sequence=True)
        out2 = self.model(view2, return_sequence=True)
        mask1_expanded = mask1.unsqueeze(-1).expand_as(sequences)
        mask2_expanded = mask2.unsqueeze(-1).expand_as(sequences)
        recon_loss1 = masked_mse_loss(out1.reconstruction, sequences, mask1_expanded)
        recon_loss2 = masked_mse_loss(out2.reconstruction, sequences, mask2_expanded)
        recon_loss = 0.5 * (recon_loss1 + recon_loss2)

        contrastive_loss = info_nce_loss(out1.projection, out2.projection, temperature=self.cfg.temperature)
        loss = self.cfg.recon_weight * recon_loss + self.cfg.contrastive_weight * contrastive_loss

        self.log("val/recon_loss", recon_loss, prog_bar=True)
        self.log("val/contrastive_loss", contrastive_loss, prog_bar=True)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):  # pragma: no cover - Lightning handles schedule
        optimizer = AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        return optimizer

    def _apply_mask(self, sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = sequences.shape
        mask = torch.rand(batch_size, seq_len, device=sequences.device) < self.cfg.mask_ratio
        mask[:, -1] = False  # keep last token as label anchor
        for i in range(batch_size):
            if not mask[i].any():
                idx = torch.randint(0, seq_len - 1, (1,), device=sequences.device)
                mask[i, idx] = True
        masked_inputs = sequences.clone()
        masked_inputs[mask] = self.cfg.mask_value
        return masked_inputs, mask
