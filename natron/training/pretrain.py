from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from natron.models.transformer import NatronTransformer
from natron.training.trainer import PretrainingTrainer, TrainerConfig
from natron.utils.logging import configure_logging
from natron.utils.torch_utils import get_device, save_checkpoint


def run_pretraining(
    sequences: torch.Tensor,
    model: NatronTransformer,
    output_path: Path,
    config: dict,
) -> None:
    configure_logging()
    dataset = TensorDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)

    trainer_cfg = TrainerConfig(
        epochs=config["epochs"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        mask_prob=config["mask_prob"],
        temperature=config["temperature"],
        use_amp=config.get("use_amp", True),
    )

    trainer = PretrainingTrainer(model, dataloader, trainer_cfg, device=get_device())
    trainer.train()

    save_checkpoint(
        {
            "model_state": model.state_dict(),
            "config": config,
        },
        output_path,
        is_best=True,
    )
