#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from natron.config.loader import load_natron_config
from natron.data.sequence import NatronDataset
from natron.evaluation.metrics import classification_report
from natron.models.transformer import NatronTransformer
from natron.training.finetune import run_finetuning
from natron.utils.data_utils import temporal_train_val_test_split
from natron.utils.logging import configure_logging
from natron.utils.seed import seed_everything
from natron.utils.torch_utils import get_device, load_checkpoint


def evaluate(model: NatronTransformer, test_loader: DataLoader) -> None:
    device = get_device()
    model.to(device)
    model.eval()

    buy_preds, buy_targets = [], []
    sell_preds, sell_targets = [], []
    direction_preds, direction_targets = [], []
    regime_preds, regime_targets = [], []

    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            outputs, _ = model(sequences)

            buy_preds.extend(torch.sigmoid(outputs["buy"]).cpu().round().numpy())
            buy_targets.extend(targets["buy"].cpu().numpy())

            sell_preds.extend(torch.sigmoid(outputs["sell"]).cpu().round().numpy())
            sell_targets.extend(targets["sell"].cpu().numpy())

            direction_preds.extend(torch.argmax(outputs["direction"], dim=-1).cpu().numpy())
            direction_targets.extend(targets["direction"].cpu().numpy())

            regime_preds.extend(torch.argmax(outputs["regime"], dim=-1).cpu().numpy())
            regime_targets.extend(targets["regime"].cpu().numpy())

    buy_targets_arr = np.array(buy_targets).astype(int)
    buy_preds_arr = np.array(buy_preds).astype(int)
    sell_targets_arr = np.array(sell_targets).astype(int)
    sell_preds_arr = np.array(sell_preds).astype(int)
    direction_targets_arr = np.array(direction_targets).astype(int)
    direction_preds_arr = np.array(direction_preds).astype(int)
    regime_targets_arr = np.array(regime_targets).astype(int)
    regime_preds_arr = np.array(regime_preds).astype(int)

    print("Buy head:", classification_report(buy_targets_arr, buy_preds_arr))
    print("Sell head:", classification_report(sell_targets_arr, sell_preds_arr))
    print("Direction head:", classification_report(direction_targets_arr, direction_preds_arr))
    print("Regime head:", classification_report(regime_targets_arr, regime_preds_arr))


def main(config_path: str | None) -> None:
    configure_logging()
    cfg = load_natron_config(Path(config_path) if config_path else None)
    paths = cfg.get("paths")
    hardware = cfg.get("hardware")
    if hardware and "seed" in hardware:
        seed_everything(hardware["seed"])
    artifacts = Path(paths["artifacts"])

    sequences = np.load(artifacts / "sequences.npy")
    targets_npz = np.load(artifacts / "targets.npz")
    targets = {k: targets_npz[k] for k in targets_npz.files}

    splits = cfg.get("data")["train"]
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = temporal_train_val_test_split(
        sequences,
        targets,
        val_ratio=splits["validation_split"],
        test_ratio=splits["test_split"],
    )

    model = NatronTransformer(
        input_dim=sequences.shape[-1],
        model_cfg=cfg.get("model"),
    )

    run_finetuning(
        train=(train_x, train_y),
        val=(val_x, val_y),
        model=model,
        checkpoint_path=Path(paths["supervised_model"]),
        config=cfg.get("supervised"),
        class_weights=cfg.get("supervised")["class_weights"],
        pretrained_path=Path(paths["pretrained_encoder"]),
    )

    checkpoint, _ = load_checkpoint(Path(paths["supervised_model"]), map_location=get_device())
    model.load_state_dict(checkpoint["model_state"])

    test_dataset = NatronDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=cfg.get("supervised")["batch_size"], shuffle=False)
    evaluate(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Natron supervised fine-tuning")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(args.config)
