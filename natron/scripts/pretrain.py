"""CLI entrypoint for Natron Transformer pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from natron.config import load_config
from natron.data.loader import DataModule, create_dataloader
from natron.models.transformer import NatronTransformer
from natron.training.pretrain import NatronPretrainer
from natron.utils import create_logger, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Natron Transformer Pretraining")
    parser.add_argument("--config", type=str, default="natron/configs/natron_base.yaml")
    parser.add_argument("--output", type=str, default="artifacts/model/pretrain.pt")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or cfg.training.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.training.seed)

    logger = create_logger("natron.pretrain", cfg.training.log_dir)
    logger.info("Preparing data...")
    module = DataModule(cfg)
    data_bundle = module.prepare()
    datasets = data_bundle["datasets"]

    train_loader = create_dataloader(
        datasets["pretrain_train"],
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        seed=cfg.training.seed,
    )
    val_loader = create_dataloader(
        datasets["pretrain_val"],
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        seed=cfg.training.seed,
    )

    feature_dim = data_bundle["features"].shape[1]
    model = NatronTransformer(feature_dim=feature_dim, config=cfg.model)
    logger.info("Starting pretraining...")
    trainer = NatronPretrainer(model, cfg.model, cfg.training, logger, device)
    trainer.fit(train_loader, val_loader)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)

    scaler = data_bundle["scaler"]
    scaler_path = output_path.with_suffix(".scaler.npz")
    import numpy as np

    np.savez(scaler_path, mean=scaler.mean, std=scaler.std)
    logger.info("Pretraining completed. Model saved to %s", args.output)


if __name__ == "__main__":
    main()
