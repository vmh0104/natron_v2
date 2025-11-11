"""
Main training pipeline for Natron Transformer V2
End-to-end pipeline: Data → Features → Labels → Sequences → Training
"""

import pandas as pd
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
import os
import argparse
from pathlib import Path

from src.feature_engine import FeatureEngine
from src.label_generator import LabelGeneratorV2
from src.sequence_creator import SequenceCreator
from src.model import NatronTransformer
from src.trainer import PretrainingTrainer, SupervisedTrainer


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: dict) -> tuple:
    """Load and prepare data"""
    print("📊 Loading data...")
    data_path = os.path.join(config['paths']['data_dir'], config['data']['input_file'])
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")
    
    # Ensure required columns exist
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def generate_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Generate technical features"""
    print("\n🔧 Generating features...")
    feature_engine = FeatureEngine()
    features_df = feature_engine.generate_all_features(df)
    
    # Count features
    feature_cols = [col for col in features_df.columns if col != 'time']
    print(f"Generated {len(feature_cols)} features")
    
    return features_df


def generate_labels(features_df: pd.DataFrame, df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Generate labels"""
    print("\n🏷️  Generating labels...")
    
    # Merge original OHLCV data for labeling
    labeling_df = pd.merge(df[['time', 'open', 'high', 'low', 'close', 'volume']], 
                          features_df, on='time', how='inner')
    
    label_generator = LabelGeneratorV2(
        neutral_buffer=config['labeling']['neutral_buffer'],
        buy_threshold=config['labeling']['buy_threshold'],
        sell_threshold=config['labeling']['sell_threshold'],
        balance_labels=config['labeling']['balance_labels'],
        stochastic_perturbation=config['labeling']['stochastic_perturbation']
    )
    
    labels_df = label_generator.generate_labels(labeling_df)
    
    return labels_df


def create_datasets(features_df: pd.DataFrame, labels_df: pd.DataFrame, config: dict):
    """Create train/val/test datasets"""
    print("\n📦 Creating sequences...")
    
    sequence_creator = SequenceCreator(
        sequence_length=config['data']['sequence_length']
    )
    
    train_dataset, val_dataset, test_dataset = sequence_creator.create_datasets(
        features_df,
        labels_df,
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        random_seed=config['data']['random_seed']
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def create_model(config: dict, input_dim: int) -> NatronTransformer:
    """Create model"""
    print("\n🧠 Creating model...")
    
    model = NatronTransformer(
        input_dim=input_dim,
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        activation=config['model']['activation'],
        max_seq_length=config['model']['max_seq_length']
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    return model


def train_pretraining(model: NatronTransformer, train_dataset, config: dict):
    """Phase 1: Pretraining"""
    print("\n🔥 Phase 1: Pretraining...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    trainer = PretrainingTrainer(model, config)
    save_dir = os.path.join(config['paths']['checkpoint_dir'], 'pretrain')
    
    trainer.train(
        train_loader,
        num_epochs=config['training']['num_epochs_pretrain'],
        save_dir=save_dir
    )
    
    print("✅ Pretraining complete!")


def train_supervised(model: NatronTransformer, train_dataset, val_dataset, config: dict):
    """Phase 2: Supervised fine-tuning"""
    print("\n🎯 Phase 2: Supervised Fine-tuning...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    trainer = SupervisedTrainer(model, config)
    save_dir = os.path.join(config['paths']['checkpoint_dir'], 'supervised')
    
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['training']['num_epochs_supervised'],
        save_dir=save_dir
    )
    
    print("✅ Supervised training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train Natron Transformer V2')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--skip-pretrain', action='store_true', help='Skip pretraining phase')
    parser.add_argument('--pretrain-only', action='store_true', help='Only run pretraining')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create directories
    for path_key in ['model_dir', 'checkpoint_dir', 'log_dir', 'data_dir']:
        os.makedirs(config['paths'][path_key], exist_ok=True)
    
    # Prepare data
    df = prepare_data(config)
    features_df = generate_features(df, config)
    labels_df = generate_labels(features_df, df, config)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(features_df, labels_df, config)
    
    # Get input dimension
    sample_x, _ = train_dataset[0]
    input_dim = sample_x.shape[1]
    print(f"\nInput dimension: {input_dim}")
    
    # Create model
    model = create_model(config, input_dim)
    
    # Training phases
    if not args.skip_pretrain:
        train_pretraining(model, train_dataset, config)
    
    if not args.pretrain_only:
        # Load pretrained weights if available
        pretrain_path = os.path.join(config['paths']['checkpoint_dir'], 'pretrain', 'best_pretrain.pt')
        if os.path.exists(pretrain_path) and not args.skip_pretrain:
            print("\n📥 Loading pretrained weights...")
            checkpoint = torch.load(pretrain_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        train_supervised(model, train_dataset, val_dataset, config)
    
    print("\n🎉 Training complete!")
    print(f"Model saved to: {os.path.join(config['paths']['checkpoint_dir'], 'supervised', 'natron_v2.pt')}")


if __name__ == "__main__":
    main()
