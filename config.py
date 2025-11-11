"""
Natron Transformer - Configuration Module
Centralized configuration for multi-task financial trading model
"""

import os
from dataclasses import dataclass
from typing import Optional
import yaml
import torch


@dataclass
class DataConfig:
    """Data processing configuration"""
    csv_path: str = "data_export.csv"
    sequence_length: int = 96
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    neutral_buffer: float = 0.001  # For direction labeling
    lookforward: int = 3  # Candles to look ahead for direction
    
    # Feature settings
    n_features: int = 100
    normalize: bool = True
    
    # Batch processing
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Transformer model architecture configuration"""
    input_dim: int = 100
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    
    # Task-specific heads
    direction_classes: int = 3  # Up, Down, Neutral
    regime_classes: int = 6  # 6 market regimes
    
    # Model output
    use_positional_encoding: bool = True
    max_seq_length: int = 96


@dataclass
class PretrainConfig:
    """Phase 1: Unsupervised pretraining configuration"""
    enabled: bool = True
    epochs: int = 50
    mask_ratio: float = 0.15
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    
    # Contrastive learning
    temperature: float = 0.07
    contrastive_weight: float = 0.3
    reconstruction_weight: float = 0.7
    
    # Checkpoint
    save_every: int = 10
    checkpoint_dir: str = "checkpoints/pretrain"


@dataclass
class TrainConfig:
    """Phase 2: Supervised training configuration"""
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Loss weights for multi-task learning
    buy_weight: float = 1.0
    sell_weight: float = 1.0
    direction_weight: float = 1.5
    regime_weight: float = 1.2
    
    # Class balancing
    use_class_weights: bool = True
    focal_loss: bool = True  # Use focal loss for imbalanced classes
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Learning rate scheduling
    scheduler: str = "reduce_on_plateau"  # or "cosine"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Early stopping
    early_stopping_patience: int = 15
    
    # Checkpoint
    save_best: bool = True
    checkpoint_dir: str = "checkpoints/supervised"


@dataclass
class RLConfig:
    """Phase 3: Reinforcement learning configuration"""
    enabled: bool = False
    algorithm: str = "ppo"  # or "sac"
    episodes: int = 1000
    steps_per_episode: int = 500
    
    # Reward function
    profit_reward: float = 1.0
    turnover_penalty: float = 0.01
    drawdown_penalty: float = 0.05
    
    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    
    learning_rate: float = 3e-4
    checkpoint_dir: str = "checkpoints/rl"


@dataclass
class InferenceConfig:
    """Inference and deployment configuration"""
    model_path: str = "model/natron_v2.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    confidence_threshold: float = 0.6
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    api_debug: bool = False
    
    # Socket server for MT5
    socket_host: str = "0.0.0.0"
    socket_port: int = 8765
    max_latency_ms: int = 50


@dataclass
class NatronConfig:
    """Main configuration container"""
    data: DataConfig
    model: ModelConfig
    pretrain: PretrainConfig
    train: TrainConfig
    rl: RLConfig
    inference: InferenceConfig
    
    # Global settings
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    
    # Logging
    log_dir: str = "logs"
    log_level: str = "INFO"
    wandb_enabled: bool = False
    wandb_project: str = "natron-transformer"
    
    # Directories
    output_dir: str = "output"
    model_dir: str = "model"


def load_config(config_path: Optional[str] = None) -> NatronConfig:
    """Load configuration from YAML file or return defaults"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return NatronConfig(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            pretrain=PretrainConfig(**config_dict.get('pretrain', {})),
            train=TrainConfig(**config_dict.get('train', {})),
            rl=RLConfig(**config_dict.get('rl', {})),
            inference=InferenceConfig(**config_dict.get('inference', {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ['data', 'model', 'pretrain', 'train', 'rl', 'inference']}
        )
    
    # Return default configuration
    return NatronConfig(
        data=DataConfig(),
        model=ModelConfig(),
        pretrain=PretrainConfig(),
        train=TrainConfig(),
        rl=RLConfig(),
        inference=InferenceConfig()
    )


def save_config(config: NatronConfig, save_path: str):
    """Save configuration to YAML file"""
    config_dict = {
        'data': config.data.__dict__,
        'model': config.model.__dict__,
        'pretrain': config.pretrain.__dict__,
        'train': config.train.__dict__,
        'rl': config.rl.__dict__,
        'inference': config.inference.__dict__,
        'seed': config.seed,
        'device': config.device,
        'mixed_precision': config.mixed_precision,
        'log_dir': config.log_dir,
        'log_level': config.log_level,
        'wandb_enabled': config.wandb_enabled,
        'wandb_project': config.wandb_project,
        'output_dir': config.output_dir,
        'model_dir': config.model_dir
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


# Default global config instance
default_config = load_config()
