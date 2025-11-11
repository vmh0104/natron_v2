#!/usr/bin/env python3
"""
Natron Transformer - Main Training Pipeline
End-to-End Multi-Task Financial Trading Model

Usage:
    python main.py --mode train          # Full training pipeline
    python main.py --mode pretrain       # Phase 1 only
    python main.py --mode supervised     # Phase 2 only
    python main.py --mode api            # Start inference API
    python main.py --mode test           # Test inference
"""

import argparse
import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from feature_engine import FeatureEngine
from label_generator import LabelGeneratorV2
from dataset import create_dataloaders
from model import create_model
from train import Phase1Trainer, Phase2Trainer, Phase3Trainer, load_pretrained_weights


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: dict):
    """
    Prepare data: Load CSV → Extract Features → Generate Labels
    
    Returns:
        features_df, labels_df, original_df
    """
    print("\n" + "="*70)
    print("📊 DATA PREPARATION")
    print("="*70)
    
    # Load data
    data_path = config['data']['input_file']
    
    if not os.path.exists(data_path):
        print(f"\n⚠️  Warning: {data_path} not found.")
        print("   Generating synthetic data for demonstration...")
        
        # Generate synthetic data
        n = 5000
        dates = pd.date_range('2023-01-01', periods=n, freq='1H')
        
        # Create realistic price movement
        trend = np.linspace(0, 50, n)
        cycle = 10 * np.sin(np.linspace(0, 10 * np.pi, n))
        noise = np.cumsum(np.random.randn(n) * 0.5)
        base_price = 100 + trend + cycle + noise
        
        df = pd.DataFrame({
            'time': dates,
            'open': base_price + np.random.randn(n) * 0.3,
            'high': base_price + np.random.rand(n) * 2,
            'low': base_price - np.random.rand(n) * 2,
            'close': base_price + np.random.randn(n) * 0.3,
            'volume': np.random.randint(1000, 10000, n)
        })
        
        # Save for reference
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/data_export.csv', index=False)
        print(f"   ✅ Synthetic data saved to data/data_export.csv")
    else:
        print(f"\n📥 Loading data from {data_path}")
        df = pd.read_csv(data_path)
    
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Date range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    
    # Extract features
    print("\n🔧 Extracting features...")
    engine = FeatureEngine()
    features = engine.extract_all_features(df)
    
    # Generate labels
    print("\n🏷️  Generating labels...")
    label_gen = LabelGeneratorV2(
        neutral_buffer=config['labeling']['neutral_buffer'],
        lookforward=config['labeling']['lookforward'],
        volume_threshold=config['labeling']['volume_threshold'],
        balance_threshold=config['labeling']['balance_threshold'],
        adaptive_balancing=config['labeling']['adaptive_balancing']
    )
    labels = label_gen.generate_labels(df, features)
    
    print("\n✅ Data preparation complete!")
    print(f"   Features: {features.shape}")
    print(f"   Labels: {labels.shape}")
    
    return features, labels, df


def train_phase1(config: dict, features: pd.DataFrame, device: torch.device):
    """Phase 1: Pretraining"""
    print("\n" + "="*70)
    print("🔄 STARTING PHASE 1: PRETRAINING")
    print("="*70)
    
    # Create pretrain dataloaders
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        features,
        None,  # No labels for pretraining
        config,
        pretrain=True
    )
    
    # Create pretrain model
    model = create_model(config, features.shape[1], pretrain=True)
    
    # Train
    trainer = Phase1Trainer(model, train_loader, val_loader, config, device)
    trainer.train()
    
    # Save scaler
    os.makedirs('model', exist_ok=True)
    import pickle
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Scaler saved to model/scaler.pkl")
    
    return model, scaler


def train_phase2(config: dict, features: pd.DataFrame, labels: pd.DataFrame, 
                 device: torch.device, pretrain_path: str = None):
    """Phase 2: Supervised Fine-Tuning"""
    print("\n" + "="*70)
    print("🎯 STARTING PHASE 2: SUPERVISED FINE-TUNING")
    print("="*70)
    
    # Create supervised dataloaders
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        features,
        labels,
        config,
        pretrain=False
    )
    
    # Create supervised model
    model = create_model(config, features.shape[1], pretrain=False)
    
    # Load pretrained weights if available
    if pretrain_path and os.path.exists(pretrain_path):
        model = load_pretrained_weights(model, pretrain_path)
    
    # Train
    trainer = Phase2Trainer(model, train_loader, val_loader, config, device)
    trainer.train()
    
    # Save final model
    final_path = config['api']['model_path']
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config
    }
    torch.save(checkpoint, final_path)
    print(f"\n✅ Final model saved to {final_path}")
    
    # Save scaler
    import pickle
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return model


def train_phase3(config: dict, model, device: torch.device):
    """Phase 3: Reinforcement Learning"""
    if not config['rl']['enabled']:
        print("\n⏭️  Phase 3 (RL) disabled in config. Skipping.")
        return
    
    trainer = Phase3Trainer(model, config, device)
    trainer.train()


def run_full_training(config: dict):
    """Run complete training pipeline: Phase 1 → Phase 2 → Phase 3"""
    print("\n" + "="*70)
    print("🚀 NATRON TRANSFORMER - FULL TRAINING PIPELINE")
    print("="*70)
    
    # Set device
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Set seed
    set_seed(config['system']['seed'])
    print(f"🎲 Random seed: {config['system']['seed']}")
    
    # Prepare data
    features, labels, df = prepare_data(config)
    
    # Phase 1: Pretraining (if enabled)
    pretrain_path = os.path.join(config['pretrain']['checkpoint_dir'], 'best_pretrain.pt')
    
    if config['pretrain']['enabled']:
        pretrain_model, scaler = train_phase1(config, features, device)
    else:
        print("\n⏭️  Phase 1 (Pretraining) disabled. Skipping.")
        pretrain_path = None
    
    # Phase 2: Supervised Fine-Tuning
    model = train_phase2(config, features, labels, device, pretrain_path)
    
    # Phase 3: Reinforcement Learning (optional)
    train_phase3(config, model, device)
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n✅ Model saved to: {config['api']['model_path']}")
    print(f"✅ Scaler saved to: model/scaler.pkl")
    print(f"\n📡 To start the inference API, run:")
    print(f"   python main.py --mode api")


def run_api():
    """Start inference API server"""
    from src.api import run_server
    run_server()


def test_inference(config: dict):
    """Test inference on sample data"""
    print("\n" + "="*70)
    print("🧪 TESTING INFERENCE")
    print("="*70)
    
    from src.api import NatronAPI
    
    # Initialize API
    api = NatronAPI()
    
    # Create sample data (96 candles)
    print("\n📊 Creating sample 96-candle sequence...")
    n = 96
    dates = pd.date_range('2024-01-01', periods=n, freq='1H')
    
    base_price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    df = pd.DataFrame({
        'time': dates,
        'open': base_price + np.random.randn(n) * 0.2,
        'high': base_price + np.random.rand(n),
        'low': base_price - np.random.rand(n),
        'close': base_price,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    # Make prediction
    print("\n🔮 Making prediction...")
    result = api.predict(df)
    
    # Display results
    print("\n" + "="*70)
    print("📊 PREDICTION RESULTS")
    print("="*70)
    print(f"\n🟢 BUY Probability:  {result['buy_prob']:.3f}")
    print(f"🔴 SELL Probability: {result['sell_prob']:.3f}")
    print(f"\n📈 Direction: {result['direction']}")
    print(f"   Probabilities:")
    for k, v in result['direction_probs'].items():
        print(f"      {k:8s}: {v:.3f}")
    print(f"\n🌊 Regime: {result['regime']}")
    print(f"   Probabilities:")
    for k, v in result['regime_probs'].items():
        print(f"      {k:12s}: {v:.3f}")
    print(f"\n⭐ Confidence: {result['confidence']:.3f}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Natron Transformer Training Pipeline')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'pretrain', 'supervised', 'api', 'test'],
                       help='Execution mode')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.mode == 'train':
        # Full training pipeline
        run_full_training(config)
    
    elif args.mode == 'pretrain':
        # Phase 1 only
        device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
        set_seed(config['system']['seed'])
        features, _, _ = prepare_data(config)
        train_phase1(config, features, device)
    
    elif args.mode == 'supervised':
        # Phase 2 only
        device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
        set_seed(config['system']['seed'])
        features, labels, _ = prepare_data(config)
        pretrain_path = os.path.join(config['pretrain']['checkpoint_dir'], 'best_pretrain.pt')
        train_phase2(config, features, labels, device, pretrain_path)
    
    elif args.mode == 'api':
        # Start API server
        run_api()
    
    elif args.mode == 'test':
        # Test inference
        test_inference(config)


if __name__ == "__main__":
    main()
