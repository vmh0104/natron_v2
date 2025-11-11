"""Pretraining utilities for Natron Transformer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from natron.config import ModelConfig, TrainingConfig
from natron.training.utils import build_optimizer, build_scheduler, save_checkpoint


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Compute InfoNCE loss between two batches of projected representations."""
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    representations = torch.cat([z1, z2], dim=0)
    similarity = torch.matmul(representations, representations.t()) / temperature

    # Mask self-similarity
    mask = torch.eye(2 * batch_size, device=z1.device).bool()
    similarity = similarity.masked_fill(mask, float("-inf"))

    positives = torch.cat([torch.sum(z1 * z2, dim=-1), torch.sum(z2 * z1, dim=-1)], dim=0) / temperature
    labels = torch.arange(2 * batch_size, device=z1.device)
    labels = (labels + batch_size) % (2 * batch_size)

    logits = similarity
    logits[torch.arange(2 * batch_size), labels] = positives

    loss = F.cross_entropy(logits, labels)
    return loss


@dataclass
class PretrainingState:
    epoch: int
    reconstruction_loss: float
    contrastive_loss: float
    total_loss: float


class NatronPretrainer:
    def __init__(
        self,
        model,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        logger,
        device: torch.device,
    ):
        self.model = model
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.logger = logger
        self.device = device
        self.optimizer = build_optimizer(model, train_cfg)
        self.scheduler = build_scheduler(self.optimizer, train_cfg)
        self.scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.amp)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        self.model.to(self.device)
        best_loss = float("inf")

        for epoch in range(1, self.train_cfg.max_epochs_pretrain + 1):
            train_state = self._train_epoch(train_loader, epoch)
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step(train_state.total_loss)

            self.logger.info(
                f"[Pretrain][Epoch {epoch}] "
                f"recon={train_state.reconstruction_loss:.4f} "
                f"contrast={train_state.contrastive_loss:.4f} "
                f"total={train_state.total_loss:.4f} "
                f"val={val_loss:.4f}" if val_loss is not None else ""
            )

            if val_loss is not None and val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    Path(self.train_cfg.checkpoint_dir) / "pretrain_best.pt",
                )

    def _train_epoch(self, loader: DataLoader, epoch: int) -> PretrainingState:
        self.model.train()
        recon_loss_meter = 0.0
        contrast_loss_meter = 0.0
        total_meter = 0.0
        count = 0

        for batch in loader:
            inputs = batch["inputs"].to(self.device)
            masked_inputs = batch["masked_inputs"].to(self.device)
            augmented = batch["augmented"].to(self.device)
            mask = batch["mask"].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.train_cfg.amp):
                outputs = self.model.forward_pretraining(masked_inputs, mask)
                reconstruction = outputs["reconstruction"]
                recon_loss = F.mse_loss(reconstruction[mask], inputs[mask])

                _, pooled_inputs = self.model.encode(inputs)
                _, pooled_aug = self.model.encode(augmented)
                proj_inputs = self.model.project(pooled_inputs)
                proj_aug = self.model.project(pooled_aug)
                contrastive_loss = info_nce_loss(
                    proj_inputs,
                    proj_aug,
                    self.model_cfg.contrastive_temperature,
                )

                loss = recon_loss + contrastive_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            recon_loss_meter += recon_loss.item()
            contrast_loss_meter += contrastive_loss.item()
            total_meter += loss.item()
            count += 1

        return PretrainingState(
            epoch=epoch,
            reconstruction_loss=recon_loss_meter / max(count, 1),
            contrastive_loss=contrast_loss_meter / max(count, 1),
            total_loss=total_meter / max(count, 1),
        )

    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch["inputs"].to(self.device)
                masked_inputs = batch["masked_inputs"].to(self.device)
                augmented = batch["augmented"].to(self.device)
                mask = batch["mask"].to(self.device)

                outputs = self.model.forward_pretraining(masked_inputs, mask)
                reconstruction = outputs["reconstruction"]
                recon_loss = F.mse_loss(reconstruction[mask], inputs[mask])

                _, pooled_inputs = self.model.encode(inputs)
                _, pooled_aug = self.model.encode(augmented)
                proj_inputs = self.model.project(pooled_inputs)
                proj_aug = self.model.project(pooled_aug)
                contrastive_loss = info_nce_loss(
                    proj_inputs,
                    proj_aug,
                    self.model_cfg.contrastive_temperature,
                )
                loss = recon_loss + contrastive_loss
                total_loss += loss.item()
                count += 1
        return total_loss / max(count, 1)
