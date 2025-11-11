"""Logging utilities for the Natron project."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: Path, run_name: str, log_level: int = logging.INFO) -> None:
    """Configure root logging handlers."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    handlers = [
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ]

    logging.basicConfig(level=log_level, format=fmt, handlers=handlers, force=True)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger instance."""
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
