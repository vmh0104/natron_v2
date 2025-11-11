"""CLI entrypoint for Natron Transformer supervised fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from natron.config import load_config
from natron.data.loader import DataModule, create_dataloader
from natron.models.transformer import NatronTransformer
from natron.training.finetune import NatronFineTuner
from natron.utils import create_logger, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Natron Transformer Fine-Tuning")
    parser.add_argument("--config", type=str, default="natron/configs/natron_base.yaml")
    parser.add_argument("--pretrained", type=str, default=None, help="Optional path to pretrained checkpoint")
    parser.add_argument("--output", type=str, default="artifacts/model/natron_v2.pt")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or cfg.training.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.training.seed)

    logger = create_logger("natron.finetune", cfg.training.log_dir)
    logger.info("Preparing data...")
    module = DataModule(cfg)
    data_bundle = module.prepare()
    datasets = data_bundle["datasets"]

    train_loader = create_dataloader(
        datasets["supervised_train"],
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        seed=cfg.training.seed,
    )
    val_loader = create_dataloader(
        datasets["supervised_val"],
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        seed=cfg.training.seed,
    )
    test_loader = create_dataloader(
        datasets["supervised_test"],
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        seed=cfg.training.seed,
    )

    feature_dim = data_bundle["features"].shape[1]
    model = NatronTransformer(feature_dim=feature_dim, config=cfg.model)

    if args.pretrained:
        logger.info("Loading pretrained weights from %s", args.pretrained)
        state = torch.load(args.pretrained, map_location="cpu")
        incompat = model.load_state_dict(state, strict=False)
        if incompat.missing_keys:
            logger.warning("Missing keys in checkpoint: %s", incompat.missing_keys)
        if incompat.unexpected_keys:
            logger.warning("Unexpected keys in checkpoint: %s", incompat.unexpected_keys)

    logger.info("Starting supervised fine-tuning...")
    trainer = NatronFineTuner(model, cfg.training, logger, device)
    trainer.fit(train_loader, val_loader)
    test_loss, test_metrics = trainer.evaluate(test_loader)
    logger.info("Test loss: %.4f | Metrics: %s", test_loss, test_metrics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)

    scaler = data_bundle["scaler"]
    scaler_path = output_path.with_suffix(".scaler.npz")
    import numpy as np

    np.savez(scaler_path, mean=scaler.mean, std=scaler.std)
    logger.info("Fine-tuned model saved to %s", args.output)


if __name__ == "__main__":
    main()
