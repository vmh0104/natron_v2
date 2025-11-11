"""
Configuration utilities for the Natron Transformer project.

This module defines dataclasses that mirror the YAML configuration files
and provides helper functions to load and validate configuration objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    data_path: str = "data/data_export.csv"
    cache_dir: str = "artifacts/cache"
    sequence_length: int = 96
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class FeatureConfig:
    rolling_windows: tuple[int, ...] = (5, 10, 14, 20, 50, 100)
    stochastic_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0
    atr_window: int = 14
    hurst_windows: tuple[int, ...] = (20, 50, 100)
    volatility_windows: tuple[int, ...] = (10, 20, 50)
    volume_windows: tuple[int, ...] = (5, 10, 20)
    market_profile_bins: int = 24
    technical_noise: float = 1e-4


@dataclass
class LabelConfig:
    neutral_buffer: float = 0.001
    volume_spike_multiplier: float = 1.5
    trend_window: int = 48
    adx_window: int = 14
    atr_window: int = 14
    atr_percentile: float = 0.9
    balance_target: float = 0.35
    min_class_fraction: float = 0.1
    max_class_fraction: float = 0.45
    stochastic_jitter: float = 0.02
    seed: int = 42


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    use_positional_encoding: bool = True
    masking_ratio: float = 0.15
    projection_dim: int = 128
    contrastive_temperature: float = 0.07


@dataclass
class TrainingConfig:
    batch_size: int = 128
    max_epochs_pretrain: int = 50
    max_epochs_supervised: int = 100
    gradient_clip_norm: float = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    amp: bool = True
    checkpoint_dir: str = "artifacts/checkpoints"
    log_dir: str = "artifacts/logs"
    device: str = "cuda"
    seed: int = 42


@dataclass
class RLConfig:
    enabled: bool = False
    algorithm: str = "ppo"
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    policy_lr: float = 3e-5
    value_lr: float = 1e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    num_steps: int = 2048
    num_epochs: int = 10
    minibatch_size: int = 256
    reward_alpha: float = 0.01
    reward_beta: float = 0.02


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    socket_host: str = "0.0.0.0"
    socket_port: int = 5555
    model_path: str = "artifacts/model/natron_v2.pt"
    device: str = "cuda"


@dataclass
class NatronConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    api: APIConfig = field(default_factory=APIConfig)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "NatronConfig":
        """Instantiate configuration from a dictionary, applying defaults."""
        def build(section_cls, key: str):
            section = cfg.get(key, {})
            if not isinstance(section, dict):
                raise TypeError(f"Config section '{key}' must be a mapping, got {type(section)}")
            return section_cls(**section)

        return cls(
            data=build(DataConfig, "data"),
            features=build(FeatureConfig, "features"),
            labels=build(LabelConfig, "labels"),
            model=build(ModelConfig, "model"),
            training=build(TrainingConfig, "training"),
            rl=build(RLConfig, "rl"),
            api=build(APIConfig, "api"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as a nested dictionary."""
        return asdict(self)


def load_config(path: Optional[str | Path] = None) -> NatronConfig:
    """
    Load a Natron configuration from YAML file or return defaults if not provided.

    Parameters
    ----------
    path:
        Optional path to a YAML configuration file. If None, default configuration
        values are used.
    """
    if path is None:
        return NatronConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg_dict: Dict[str, Any] = yaml.safe_load(f) or {}

    return NatronConfig.from_dict(cfg_dict)
