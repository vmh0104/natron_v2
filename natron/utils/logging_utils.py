from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
