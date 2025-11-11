"""Phase 1 — Masked modeling and contrastive pretraining."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from natron.data.datamodule import DataModuleConfig, NatronDataModule
from natron.models.pretraining import NatronPretrainingModel, PretrainingConfig
from natron.models.transformer import NatronEncoder, NatronModelConfig
from natron.utils.config import NatronConfig
from natron.utils.logging import get_logger


logger = get_logger(__name__)


def run_pretraining(
    config: NatronConfig,
    device: torch.device,
    data_module: Optional[NatronDataModule] = None,
) -> NatronEncoder:
    """Execute masked reconstruction and contrastive pretraining, returning the encoder."""
    phase_cfg = config.phases.get("pretraining", {})
    masking_ratio = phase_cfg.get("masking_ratio", 0.15)
    projection_dim = phase_cfg.get("projection_dim", 128)
    temperature = phase_cfg.get("temperature", 0.1)
    epochs = phase_cfg.get("epochs", 50)
    batch_size = phase_cfg.get("batch_size", 64)
    lr = phase_cfg.get("lr", 3e-4)
    weight_decay = phase_cfg.get("weight_decay", 1e-5)
    grad_clip = phase_cfg.get("grad_clip", 1.0)

    if data_module is None:
        data_module = NatronDataModule(Path(config.data_path), DataModuleConfig(batch_size=batch_size))
        data_module.prepare()
    elif data_module.sequences is None:
        data_module.prepare()

    train_loader = data_module.get_pretrain_dataloader("train", batch_size=batch_size, shuffle=True)
    val_loader = data_module.get_pretrain_dataloader("val", batch_size=batch_size, shuffle=False)

    model_config = NatronModelConfig(feature_dim=data_module.feature_dim)
    encoder = NatronEncoder(model_config).to(device)
    pretrain_cfg = PretrainingConfig(masking_ratio=masking_ratio, projection_dim=projection_dim, temperature=temperature)
    model = NatronPretrainingModel(encoder, pretrain_cfg).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state: Optional[dict] = None

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, device, grad_clip, training=True)
        val_loss = _run_epoch(model, val_loader, optimizer, device, grad_clip, training=False)

        logger.info("Pretraining epoch %s | train_loss=%.4f | val_loss=%.4f", epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = encoder.state_dict()
            _save_checkpoint(config, "pretrain", {"encoder": best_state})

    if best_state is not None:
        encoder.load_state_dict(best_state)
    return encoder


def _run_epoch(
    model: NatronPretrainingModel,
    dataloader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    grad_clip: float,
    training: bool = True,
) -> float:
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in tqdm(dataloader, desc="pretrain" if training else "pretrain-val", leave=False):
        batch = batch.to(device)
        mask = _generate_mask(batch.shape, model.pretrain_cfg.masking_ratio, device)

        with torch.set_grad_enabled(training):
            reconstructed, targets, mask_tensor, proj_pair = model(batch, mask)
            recon_loss = model.reconstruction_loss(reconstructed, targets, mask_tensor)
            contrastive_loss = model.contrastive_loss(proj_pair)
            loss = recon_loss + contrastive_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def _generate_mask(shape: Tuple[int, int, int], ratio: float, device: torch.device) -> torch.Tensor:
    batch_size, seq_len, _ = shape
    mask = torch.rand(batch_size, seq_len, device=device) < ratio
    # Ensure at least one masked token per sequence
    any_masked = mask.any(dim=1, keepdim=True)
    random_indices = torch.randint(0, seq_len, (batch_size, 1), device=device)
    mask = mask | torch.zeros_like(mask).scatter_(1, random_indices, ~any_masked)
    return mask


def _save_checkpoint(config: NatronConfig, suffix: str, state: dict) -> None:
    path = Path(config.checkpoint_dir) / f"{config.run_name}_{suffix}.pt"
    torch.save(state, path)
    logger.info("Saved %s checkpoint to %s", suffix, path)


__all__ = ["run_pretraining"]
