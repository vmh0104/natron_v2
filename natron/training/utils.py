"""Shared training utilities for Natron."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from natron.config import TrainingConfig


def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )


def build_scheduler(optimizer: Optimizer, cfg: TrainingConfig) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        verbose=True,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader, device: torch.device) -> float:
    model.eval()
    running_loss = 0.0
    count = 0
    criterion = nn.MSELoss()
    for batch in dataloader:
        inputs = batch["inputs"].to(device)
        masked_inputs = batch["masked_inputs"].to(device)
        mask = batch["mask"].to(device)
        outputs = model.forward_pretraining(masked_inputs, mask)
        recon = outputs["reconstruction"]
        loss = criterion(recon[mask], inputs[mask])
        running_loss += loss.item()
        count += 1
    return running_loss / max(count, 1)
