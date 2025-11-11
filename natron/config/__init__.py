from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result


DEFAULT_CONFIG_PATH = Path(__file__).with_name("defaults.yaml")


def load_default_config(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    if extra:
        cfg = merge_dict(cfg, extra)
    return cfg
