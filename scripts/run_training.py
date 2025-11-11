#!/usr/bin/env python
"""Entry point for running the full Natron training pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from natron.pipelines.train_pipeline import run_training
from natron.utils.config import load_config
from natron.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Natron multi-phase training pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/natron.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint to resume from",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.log_dir, config.run_name)
    run_training(config, resume_checkpoint=args.resume)


if __name__ == "__main__":
    main()
