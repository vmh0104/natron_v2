"""Utility functions for loading and validating Natron configuration files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class NatronConfig:
    """Container for Natron configuration values."""

    data_path: Path
    model_dir: Path
    log_dir: Path
    checkpoint_dir: Path
    run_name: str
    device: str
    seed: int
    phases: Dict[str, Any]

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "NatronConfig":
        required_keys = {
            "data_path",
            "model_dir",
            "log_dir",
            "checkpoint_dir",
            "run_name",
            "device",
            "seed",
            "phases",
        }
        missing = required_keys - cfg.keys()
        if missing:
            raise ValueError(f"Missing configuration keys: {missing}")

        def as_path(value: Any) -> Path:
            path = Path(value).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            return path

        return cls(
            data_path=as_path(cfg["data_path"]),
            model_dir=as_path(cfg["model_dir"]),
            log_dir=as_path(cfg["log_dir"]),
            checkpoint_dir=as_path(cfg["checkpoint_dir"]),
            run_name=str(cfg["run_name"]),
            device=str(cfg["device"]),
            seed=int(cfg["seed"]),
            phases=dict(cfg["phases"]),
        )


def load_config(path: str | Path) -> NatronConfig:
    """Load a YAML configuration file and return a :class:`NatronConfig`."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping at the root")
    return NatronConfig.from_dict(data)


__all__ = ["NatronConfig", "load_config"]
