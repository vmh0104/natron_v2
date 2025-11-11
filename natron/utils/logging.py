import logging
import os
from typing import Optional


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Configure Python logging with both console and optional file handlers.

    Parameters
    ----------
    level:
        Logging level for the root logger.
    log_file:
        Optional path to a log file. If provided, a `FileHandler` is attached.
    overwrite:
        When True, truncates the existing log file instead of appending.
    """
    root_logger = logging.getLogger()

    if getattr(configure_logging, "_configured", False):
        root_logger.setLevel(level)
        return

    root_logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_mode = "w" if overwrite else "a"
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    configure_logging._configured = True  # type: ignore[attr-defined]
