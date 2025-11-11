"""Supervised fine-tuning routines for Natron."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from natron.config import TrainingConfig
from natron.utils.metrics import compute_classification_metrics, multitask_loss_dict
from natron.training.utils import build_optimizer, build_scheduler, save_checkpoint


@dataclass
class SupervisedState:
    epoch: int
    loss: float
    metrics: Dict[str, float]


class NatronFineTuner:
    def __init__(
        self,
        model,
        train_cfg: TrainingConfig,
        logger,
        device: torch.device,
    ):
        self.model = model
        self.train_cfg = train_cfg
        self.logger = logger
        self.device = device
        self.optimizer = build_optimizer(model, train_cfg)
        self.scheduler = build_scheduler(self.optimizer, train_cfg)
        self.scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.amp)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        self.model.to(self.device)
        best_loss = float("inf")

        for epoch in range(1, self.train_cfg.max_epochs_supervised + 1):
            train_state = self._train_epoch(train_loader, epoch)
            val_loss, val_metrics = self._evaluate(val_loader)
            self.scheduler.step(val_loss)

            self.logger.info(
                f"[Finetune][Epoch {epoch}] "
                f"loss={train_state.loss:.4f} val_loss={val_loss:.4f} "
                f"buy_auc={val_metrics['buy_auc']:.3f} sell_auc={val_metrics['sell_auc']:.3f} "
                f"dir_acc={val_metrics['direction_acc']:.3f} regime_acc={val_metrics['regime_acc']:.3f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    Path(self.train_cfg.checkpoint_dir) / "finetune_best.pt",
                )

    def _train_epoch(self, loader: DataLoader, epoch: int) -> SupervisedState:
        self.model.train()
        running_loss = 0.0
        count = 0
        for batch in loader:
            inputs = batch["inputs"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

            with torch.cuda.amp.autocast(enabled=self.train_cfg.amp):
                outputs = self.model(inputs)
                loss, _ = multitask_loss_dict(outputs, targets)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            count += 1

        return SupervisedState(epoch=epoch, loss=running_loss / max(count, 1), metrics={})

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        losses = []
        buy_probs, buy_targets = [], []
        sell_probs, sell_targets = [], []
        direction_preds, direction_targets = [], []
        regime_preds, regime_targets = [], []

        for batch in loader:
            inputs = batch["inputs"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch["targets"].items()}
            outputs = self.model(inputs)

            loss, _ = multitask_loss_dict(outputs, targets)
            losses.append(loss.item())

            buy_probs.extend(torch.sigmoid(outputs["buy"]).squeeze(-1).cpu().numpy())
            buy_targets.extend(targets["buy"].cpu().numpy())

            sell_probs.extend(torch.sigmoid(outputs["sell"]).squeeze(-1).cpu().numpy())
            sell_targets.extend(targets["sell"].cpu().numpy())

            direction_preds.extend(outputs["direction"].argmax(dim=-1).cpu().numpy())
            direction_targets.extend(targets["direction"].cpu().numpy())

            regime_preds.extend(outputs["regime"].argmax(dim=-1).cpu().numpy())
            regime_targets.extend(targets["regime"].cpu().numpy())

        buy_auc = compute_classification_metrics(buy_targets, np.array(buy_probs) > 0.5, buy_probs).auc_roc or 0.0
        sell_auc = compute_classification_metrics(sell_targets, np.array(sell_probs) > 0.5, sell_probs).auc_roc or 0.0
        direction_acc = (np.array(direction_preds) == np.array(direction_targets)).mean()
        regime_acc = (np.array(regime_preds) == np.array(regime_targets)).mean()

        metrics = {
            "buy_auc": buy_auc,
            "sell_auc": sell_auc,
            "direction_acc": float(direction_acc),
            "regime_acc": float(regime_acc),
        }
        return float(np.mean(losses)), metrics

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.model.eval()
        outputs = self.model(inputs.to(self.device))
        outputs["buy_prob"] = torch.sigmoid(outputs["buy"])
        outputs["sell_prob"] = torch.sigmoid(outputs["sell"])
        outputs["direction_prob"] = torch.softmax(outputs["direction"], dim=-1)
        outputs["regime_prob"] = torch.softmax(outputs["regime"], dim=-1)
        return outputs

    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        return self._evaluate(loader)
