from pathlib import Path
from typing import Any, Dict, Optional

from natron.config.defaults import get_default_config
from natron.utils.config import Config, load_config


def load_natron_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    defaults = get_default_config()
    config = Config(data=defaults, source=None)

    if config_path and config_path.exists():
        loaded = load_config(config_path, overrides)
        config = config.merge(loaded.data)
        config.source = config_path
    elif overrides:
        config = config.merge(overrides)

    return config
