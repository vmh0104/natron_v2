"""Phase 2 — Supervised fine-tuning for multi-task predictions."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from natron.data.datamodule import DataModuleConfig, NatronDataModule
from natron.models.transformer import NatronEncoder, NatronModelConfig, NatronTransformer
from natron.utils.config import NatronConfig
from natron.utils.logging import get_logger


logger = get_logger(__name__)


def run_finetuning(
    config: NatronConfig,
    device: torch.device,
    encoder: Optional[NatronEncoder] = None,
    data_module: Optional[NatronDataModule] = None,
) -> NatronTransformer:
    """Execute supervised fine-tuning, returning the trained multi-task model."""
    phase_cfg = config.phases.get("finetuning", {})
    epochs = phase_cfg.get("epochs", 100)
    batch_size = phase_cfg.get("batch_size", 32)
    lr = phase_cfg.get("lr", 1e-4)
    weight_decay = phase_cfg.get("weight_decay", 1e-5)
    patience = phase_cfg.get("scheduler", {}).get("patience", 5)
    factor = phase_cfg.get("scheduler", {}).get("factor", 0.5)
    grad_clip = phase_cfg.get("grad_clip", 1.0)

    if data_module is None:
        data_module = NatronDataModule(Path(config.data_path), DataModuleConfig(batch_size=batch_size))
        data_module.prepare()
    elif data_module.sequences is None:
        data_module.prepare()

    model_config = NatronModelConfig(feature_dim=data_module.feature_dim)
    multitask_model = NatronTransformer(model_config, encoder=encoder).to(device)

    optimizer = AdamW(multitask_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=patience, factor=factor, verbose=True)

    train_loader = data_module.get_supervised_dataloader("train", batch_size=batch_size, shuffle=True)
    val_loader = data_module.get_supervised_dataloader("val", batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state: Optional[dict] = None

    for epoch in range(1, epochs + 1):
        train_metrics = _run_supervised_epoch(multitask_model, train_loader, optimizer, device, grad_clip, training=True)
        val_metrics = _run_supervised_epoch(multitask_model, val_loader, optimizer, device, grad_clip, training=False)

        scheduler.step(val_metrics["loss"])
        logger.info(
            "Finetune epoch %s | train_loss=%.4f | val_loss=%.4f | dir_acc=%.3f | regime_acc=%.3f",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["direction_acc"],
            val_metrics["regime_acc"],
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = multitask_model.state_dict()
            _save_checkpoint(config, "finetune", best_state)

    if best_state is not None:
        multitask_model.load_state_dict(best_state)
    return multitask_model


def _run_supervised_epoch(
    model: NatronTransformer,
    dataloader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    grad_clip: float,
    training: bool = True,
) -> Dict[str, float]:
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0
    direction_correct = 0
    regime_correct = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="finetune" if training else "finetune-val", leave=False):
        inputs = batch["inputs"].to(device)
        buy_target = batch["buy"].to(device)
        sell_target = batch["sell"].to(device)
        direction_target = batch["direction"].to(device)
        regime_target = batch["regime"].to(device)

        with torch.set_grad_enabled(training):
            buy_logits, sell_logits, direction_logits, regime_logits = model(inputs)
            buy_logits = buy_logits.squeeze(-1)
            sell_logits = sell_logits.squeeze(-1)

            buy_loss = F.binary_cross_entropy_with_logits(buy_logits, buy_target)
            sell_loss = F.binary_cross_entropy_with_logits(sell_logits, sell_target)
            direction_loss = F.cross_entropy(direction_logits, direction_target)
            regime_loss = F.cross_entropy(regime_logits, regime_target)

            loss = buy_loss + sell_loss + direction_loss + regime_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        direction_pred = direction_logits.argmax(dim=1)
        regime_pred = regime_logits.argmax(dim=1)
        direction_correct += (direction_pred == direction_target).sum().item()
        regime_correct += (regime_pred == regime_target).sum().item()
        total_samples += direction_target.size(0)

    metrics = {
        "loss": total_loss / max(total_batches, 1),
        "direction_acc": direction_correct / max(total_samples, 1),
        "regime_acc": regime_correct / max(total_samples, 1),
    }
    return metrics


def _save_checkpoint(config: NatronConfig, suffix: str, state: dict) -> None:
    path = Path(config.checkpoint_dir) / f"{config.run_name}_{suffix}.pt"
    torch.save({"model": state}, path)
    logger.info("Saved %s checkpoint to %s", suffix, path)


__all__ = ["run_finetuning"]
