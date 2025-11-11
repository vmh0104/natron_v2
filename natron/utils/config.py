from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class Config:
    data: Dict[str, Any]
    source: Optional[Path] = None

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def merge(self, other: Dict[str, Any]) -> "Config":
        merged = copy.deepcopy(self.data)
        _recursive_update(merged, other)
        return Config(data=merged, source=self.source)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)


def load_config(path: Path, overrides: Optional[Dict[str, Any]] = None) -> Config:
    with path.open("r") as handle:
        data = yaml.safe_load(handle) or {}
    if overrides:
        _recursive_update(data, overrides)
    return Config(data=data, source=path)


def _recursive_update(base: Dict[str, Any], updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _recursive_update(base[key], value)
        else:
            base[key] = value
