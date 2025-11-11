#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from natron.config.loader import load_natron_config
from natron.data.pipeline import DataPipeline
from natron.utils.logging import configure_logging
from natron.utils.seed import seed_everything


def main(config_path: str | None) -> None:
    configure_logging()
    cfg = load_natron_config(Path(config_path) if config_path else None)
    paths = cfg.get("paths")
    hardware = cfg.get("hardware")
    if hardware and "seed" in hardware:
        seed_everything(hardware["seed"])
    data_path = Path(paths["raw_data"])
    artifacts = Path(paths["artifacts"])
    artifacts.mkdir(exist_ok=True, parents=True)

    pipeline = DataPipeline(
        data_path=data_path,
        sequence_length=cfg.get("data")["sequence_length"],
        neutral_buffer=cfg.get("data")["features"]["neutral_buffer"],
    )
    raw, features, labels, (sequences, targets) = pipeline.run()

    features.to_parquet(artifacts / "features.parquet")
    labels.to_parquet(artifacts / "labels.parquet")
    np.save(artifacts / "sequences.npy", sequences)
    np.savez(
        artifacts / "targets.npz",
        buy=targets["buy"],
        sell=targets["sell"],
        direction=targets["direction"],
        regime=targets["regime"],
    )
    raw["close"].to_csv(artifacts / "close_prices.csv", index=False)

    print(f"Saved artifacts to {artifacts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Natron data.")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML.")
    args = parser.parse_args()
    main(args.config)
