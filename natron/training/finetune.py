from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from natron.data.sequence import NatronDataset
from natron.models.transformer import NatronTransformer
from natron.training.trainer import SupervisedTrainer, TrainerConfig
from natron.utils.logging import configure_logging
from natron.utils.torch_utils import get_device, load_checkpoint


def _to_tensor(weight) -> torch.Tensor:
    if isinstance(weight, (float, int)):
        return torch.tensor(weight, dtype=torch.float32)
    return torch.tensor(weight, dtype=torch.float32)


def run_finetuning(
    train: Tuple[np.ndarray, Dict[str, np.ndarray]],
    val: Tuple[np.ndarray, Dict[str, np.ndarray]],
    model: NatronTransformer,
    checkpoint_path: Path,
    config: dict,
    class_weights: dict,
    pretrained_path: Optional[Path] = None,
) -> None:
    configure_logging()

    device = get_device()
    if pretrained_path and pretrained_path.exists():
        state_dict, _ = load_checkpoint(pretrained_path, map_location=device)
        model.load_state_dict(state_dict["model_state"])

    train_dataset = NatronDataset(*train)
    val_dataset = NatronDataset(*val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    trainer_cfg = TrainerConfig(
        epochs=config["epochs"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        use_amp=config.get("use_amp", True),
        patience=config.get("scheduler", {}).get("patience", 5),
    )

    weights = {
        "buy": _to_tensor(class_weights.get("buy", 1.0)),
        "sell": _to_tensor(class_weights.get("sell", 1.0)),
        "direction": _to_tensor(class_weights.get("direction", [1.0, 1.0, 1.0])),
        "regime": _to_tensor(class_weights.get("regime", [1.0] * 6)),
    }

    trainer = SupervisedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_cfg,
        class_weights=weights,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    trainer.train()
