"""Coordination logic for multi-phase Natron training."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from natron.data.datamodule import DataModuleConfig, NatronDataModule
from natron.models.transformer import NatronEncoder, NatronModelConfig, NatronTransformer
from natron.training import finetune, pretrain, reinforcement
from natron.training.reinforcement import NatronRLPolicy
from natron.utils.config import NatronConfig
from natron.utils.logging import get_logger


logger = get_logger(__name__)


def run_training(config: NatronConfig, resume_checkpoint: Optional[Path] = None) -> None:
    """Run all configured training phases sequentially."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    data_module = NatronDataModule(Path(config.data_path), DataModuleConfig())
    data_module.prepare()

    encoder: NatronEncoder | None = None
    multitask_model: NatronTransformer | None = None
    rl_policy: NatronRLPolicy | None = None

    if config.phases.get("pretraining", {}).get("enabled", True):
        logger.info("Starting Phase 1 — Unsupervised Pretraining")
        encoder = pretrain.run_pretraining(config, device, data_module=data_module)
    else:
        logger.info("Skipping Phase 1 — Pretraining disabled in config")
        encoder = NatronEncoder(NatronModelConfig(feature_dim=data_module.feature_dim)).to(device)

    if config.phases.get("finetuning", {}).get("enabled", True):
        logger.info("Starting Phase 2 — Supervised Fine-Tuning")
        multitask_model = finetune.run_finetuning(config, device, encoder=encoder, data_module=data_module)
    else:
        logger.info("Skipping Phase 2 — Fine-tuning disabled in config")
        multitask_model = NatronTransformer(NatronModelConfig(feature_dim=data_module.feature_dim), encoder=encoder).to(device)

    if config.phases.get("reinforcement", {}).get("enabled", False):
        logger.info("Starting Phase 3 — Reinforcement Learning")
        rl_policy = reinforcement.run_reinforcement(config, device, policy_model=multitask_model, data_module=data_module)
    else:
        logger.info("Skipping Phase 3 — Reinforcement disabled in config")
        rl_policy = NatronRLPolicy(multitask_model.get_encoder(), train_encoder=False)

    _save_final_artifacts(config, multitask_model, rl_policy, data_module)
    logger.info("Training pipeline complete")


def _save_final_artifacts(
    config: NatronConfig,
    model: NatronTransformer,
    rl_policy: NatronRLPolicy,
    data_module: NatronDataModule,
) -> None:
    path = Path(config.model_dir) / "natron_v2.pt"
    scaler = data_module.scaler
    feature_columns = list(data_module.features.columns) if data_module.features is not None else []
    checkpoint = {
        "model_state": _cpu_state_dict(model),
        "model_config": model.config.__dict__,
        "feature_columns": feature_columns,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "rl_policy_state": _cpu_state_dict(rl_policy),
        "phases": config.phases,
        "window_length": data_module.config.window_length,
    }
    torch.save(checkpoint, path)
    logger.info("Saved final Natron model bundle to %s", path)


def _cpu_state_dict(module: torch.nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


__all__ = ["run_training"]
