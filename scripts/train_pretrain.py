#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from natron.config.loader import load_natron_config
from natron.models.transformer import NatronTransformer
from natron.training.pretrain import run_pretraining
from natron.utils.logging import configure_logging
from natron.utils.seed import seed_everything


def main(config_path: str | None) -> None:
    configure_logging()
    cfg = load_natron_config(Path(config_path) if config_path else None)
    paths = cfg.get("paths")
    hardware = cfg.get("hardware")
    if hardware and "seed" in hardware:
        seed_everything(hardware["seed"])
    artifacts = Path(paths["artifacts"])
    sequences_path = artifacts / "sequences.npy"

    if not sequences_path.exists():
        raise FileNotFoundError("Sequences not found. Run scripts/prepare_data.py first.")

    sequences = torch.from_numpy(np.load(sequences_path)).float()

    model = NatronTransformer(
        input_dim=sequences.shape[-1],
        model_cfg=cfg.get("model"),
    )

    run_pretraining(
        sequences=sequences,
        model=model,
        output_path=Path(paths["pretrained_encoder"]),
        config=cfg.get("pretraining"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Natron pretraining")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(args.config)
