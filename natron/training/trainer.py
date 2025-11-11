from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from natron.data.augmentations import SequenceAugmentations
from natron.models.pretraining import info_nce_loss, mask_sequence, masked_mse_loss
from natron.training.callbacks import EarlyStopping
from natron.training.metrics import multitask_loss
from natron.utils.torch_utils import autocast, get_device, save_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    epochs: int
    lr: float
    weight_decay: float
    use_amp: bool = True
    mask_prob: float = 0.15
    temperature: float = 0.1
    patience: int = 10


class PretrainingTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        config: TrainerConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.device = device or get_device()
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.augment = SequenceAugmentations()
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    def train(self) -> None:
        self.model.train()
        for epoch in range(1, self.config.epochs + 1):
            epoch_loss = 0.0
            for batch in self.dataloader:
                sequences = batch[0] if isinstance(batch, (list, tuple)) else batch
                sequences = sequences.to(self.device)

                masked_x, mask, target = mask_sequence(sequences, self.config.mask_prob)
                masked_x = masked_x.to(self.device)
                mask = mask.to(self.device)
                target = target.to(self.device)

                aug1, aug2 = self._build_augmented_views(sequences)

                self.optimizer.zero_grad()

                with autocast(self.config.use_amp, self.device):
                    recon = self.model.masked_reconstruction(masked_x, mask)
                    if recon.numel() == 0:
                        continue
                    target_mask = target.masked_select(mask.unsqueeze(-1).expand_as(target))
                    target_masked = target_mask.view(-1, target.size(-1))
                    recon_loss = masked_mse_loss(recon, target_masked)

                    z1 = self.model.contrastive_projection(aug1)
                    z2 = self.model.contrastive_projection(aug2)
                    contrastive = info_nce_loss(z1, z2, temperature=self.config.temperature)

                    loss = recon_loss + contrastive

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()

            logger.info("Pretraining epoch %d | loss: %.4f", epoch, epoch_loss / len(self.dataloader))

    def _build_augmented_views(self, sequences: torch.Tensor) -> tuple:
        seq_np = sequences.detach().cpu().numpy()
        aug1 = np.stack([self.augment.jitter(self.augment.scaling(seq)) for seq in seq_np])
        aug2 = np.stack([self.augment.time_mask(self.augment.jitter(seq)) for seq in seq_np])
        aug1_tensor = torch.from_numpy(aug1).to(self.device, dtype=torch.float32)
        aug2_tensor = torch.from_numpy(aug2).to(self.device, dtype=torch.float32)
        return aug1_tensor, aug2_tensor


class SupervisedTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainerConfig,
        class_weights: Dict[str, torch.Tensor],
        checkpoint_path,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or get_device()
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=1e-4,
        )
        self.early_stopping = EarlyStopping(patience=config.patience, min_delta=1e-4)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
        self.class_weights = {k: v.to(self.device) for k, v in class_weights.items()}
        self.checkpoint_path = checkpoint_path

    def train(self) -> None:
        best_val_loss = float("inf")

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._run_epoch(self.train_loader, training=True)
            val_loss = self._run_epoch(self.val_loader, training=False)

            self.scheduler.step(val_loss)
            self.early_stopping.step(val_loss)

            logger.info(
                "Supervised epoch %d | train loss: %.4f | val loss: %.4f",
                epoch,
                train_loss,
                val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    {
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    self.checkpoint_path,
                    is_best=True,
                )

            if self.early_stopping.should_stop:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        self.model.train(training)
        total_loss = 0.0
        for batch in loader:
            sequences, targets = batch
            sequences = sequences.to(self.device)
            targets = {
                key: value.to(self.device)
                for key, value in targets.items()
            }

            if training:
                self.optimizer.zero_grad()

            with autocast(self.config.use_amp, self.device):
                outputs, _ = self.model(sequences)
                loss_dict = multitask_loss(outputs, targets, self.class_weights)
                loss = loss_dict["total"]

            if training:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss += loss.item()
        return total_loss / len(loader)
