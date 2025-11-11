"""Logging utilities for Natron."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def create_logger(
    name: str = "natron",
    log_dir: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and configure a project-wide logger.

    Parameters
    ----------
    name:
        Logger name, defaults to 'natron'.
    log_dir:
        Optional directory to write logs. If provided, logs are written to both file and stdout.
    level:
        Logging level, defaults to logging.INFO.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
