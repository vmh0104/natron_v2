from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..models import NatronTransformer
from .losses import multitask_loss
from .metrics import compute_metrics


@dataclass(slots=True)
class SupervisedConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5


class NatronSupervisedModule(pl.LightningModule):
    def __init__(self, model: NatronTransformer, config: SupervisedConfig) -> None:
        super().__init__()
        self.model = model
        self.cfg = config
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  # pragma: no cover
        return vars(self.model(x))

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  # pragma: no cover
        sequences, targets = batch
        sequences = sequences.float()
        outputs = self.model(sequences)
        outputs_dict = vars(outputs)
        loss = multitask_loss(outputs_dict, targets)
        metrics = compute_metrics(outputs_dict, targets)
        self.log("train/loss", loss, prog_bar=True)
        for name, value in metrics.items():
            self.log(f"train/{name}", value, prog_bar=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  # pragma: no cover
        sequences, targets = batch
        sequences = sequences.float()
        outputs = self.model(sequences)
        outputs_dict = vars(outputs)
        loss = multitask_loss(outputs_dict, targets)
        metrics = compute_metrics(outputs_dict, targets)
        self.log("val/loss", loss, prog_bar=True)
        for name, value in metrics.items():
            self.log(f"val/{name}", value, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  # pragma: no cover
        sequences, targets = batch
        sequences = sequences.float()
        outputs = self.model(sequences)
        outputs_dict = vars(outputs)
        loss = multitask_loss(outputs_dict, targets)
        metrics = compute_metrics(outputs_dict, targets)
        self.log("test/loss", loss)
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
        return loss

    def configure_optimizers(self):  # pragma: no cover
        optimizer = AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.cfg.scheduler_factor,
            patience=self.cfg.scheduler_patience,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "frequency": 1,
            },
        }

    def on_after_backward(self) -> None:  # pragma: no cover
        if self.cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
