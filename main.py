#!/usr/bin/env python3
"""
Natron Transformer - Main Training Pipeline
End-to-End training from data to deployed model
"""

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from feature_engine import FeatureEngine
from label_generator import LabelGeneratorV2
from sequence_creator import SequenceCreator, split_data, create_dataloaders
from model import NatronTransformer, count_parameters
from pretrain import PretrainEngine
from train_supervised import SupervisedTrainer
from train_rl import PPOAgent, TradingEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration file"""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(config: dict) -> pd.DataFrame:
    """Load OHLCV data"""
    data_path = config['data']['input_file']
    logger.info(f"Loading data from {data_path}")
    
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Creating sample data for demonstration...")
        
        # Generate sample data
        n_samples = 5000
        np.random.seed(42)
        
        df = pd.DataFrame({
            'time': pd.date_range('2020-01-01', periods=n_samples, freq='15min'),
            'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
            'high': 100 + np.cumsum(np.random.randn(n_samples) * 0.5) + np.abs(np.random.randn(n_samples) * 0.3),
            'low': 100 + np.cumsum(np.random.randn(n_samples) * 0.5) - np.abs(np.random.randn(n_samples) * 0.3),
            'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
            'volume': np.random.randint(1000, 10000, n_samples)
        })
        
        # Save sample data
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.info(f"Sample data created and saved to {data_path}")
    else:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} candles")
    
    return df


def prepare_data(df: pd.DataFrame, config: dict):
    """
    Prepare data: features, labels, sequences.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler
    """
    logger.info("\n" + "="*60)
    logger.info("📊 DATA PREPARATION")
    logger.info("="*60)
    
    # 1. Generate features
    feature_engine = FeatureEngine()
    features = feature_engine.generate_features(df)
    
    # 2. Generate labels
    label_generator = LabelGeneratorV2(config['labeling'])
    labels = label_generator.generate_labels(df, features)
    
    # 3. Create sequences
    sequence_creator = SequenceCreator(
        sequence_length=config['data']['sequence_length']
    )
    
    X, y = sequence_creator.create_sequences(features, labels, fit_scaler=True)
    
    # 4. Split data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y,
        test_split=config['data']['test_split'],
        val_split=config['data']['validation_split'],
        shuffle=False
    )
    
    # 5. Save scaler
    scaler_path = 'model/scaler.pkl'
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    sequence_creator.save_scaler(scaler_path)
    
    logger.info("✅ Data preparation complete")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, sequence_creator


def phase1_pretrain(config: dict, train_loader, val_loader, device: str):
    """Phase 1: Pretraining"""
    logger.info("\n" + "="*60)
    logger.info("🧠 PHASE 1: PRETRAINING")
    logger.info("="*60)
    
    # Create model
    model = NatronTransformer(
        n_features=config['features']['total_features'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        max_seq_length=config['model']['max_seq_length']
    )
    
    logger.info(f"Model created with {count_parameters(model):,} parameters")
    
    # Train
    pretrain_engine = PretrainEngine(model, config, device)
    pretrain_engine.train(train_loader, val_loader)
    
    return model


def phase2_supervised(config: dict, model, train_loader, val_loader, device: str):
    """Phase 2: Supervised Fine-Tuning"""
    logger.info("\n" + "="*60)
    logger.info("🎯 PHASE 2: SUPERVISED FINE-TUNING")
    logger.info("="*60)
    
    # Load pretrained weights if available
    pretrain_path = Path(config['pretrain']['checkpoint_dir']) / 'pretrain_best.pt'
    
    trainer = SupervisedTrainer(model, config, device)
    
    if pretrain_path.exists() and config['supervised']['load_pretrained']:
        trainer.load_pretrained(str(pretrain_path))
    else:
        logger.info("Training from scratch (no pretrained weights)")
    
    # Train
    trainer.train(train_loader, val_loader)
    
    return model


def phase3_rl(config: dict, model, X_train, y_train, device: str):
    """Phase 3: Reinforcement Learning"""
    logger.info("\n" + "="*60)
    logger.info("🎮 PHASE 3: REINFORCEMENT LEARNING")
    logger.info("="*60)
    
    # Create environment
    env = TradingEnvironment(X_train, y_train, config)
    
    # Create agent
    agent = PPOAgent(model, config, device)
    
    # Train
    n_episodes = config['rl']['episodes']
    agent.train(env, n_episodes)
    
    return model


def save_final_model(model, config: dict):
    """Save final production model"""
    output_path = config['api']['model_path']
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config
    }
    
    torch.save(checkpoint, output_path)
    logger.info(f"💾 Final model saved to {output_path}")


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Natron Transformer Training Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--skip-pretrain', action='store_true',
                       help='Skip pretraining phase')
    parser.add_argument('--skip-rl', action='store_true',
                       help='Skip RL phase')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data file (overrides config)')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("🧠 NATRON TRANSFORMER")
    print("Multi-Task Financial Trading Model")
    print("="*60 + "\n")
    
    # Load config
    config = load_config(args.config)
    
    # Override data path if provided
    if args.data:
        config['data']['input_file'] = args.data
    
    # Set device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    logger.info(f"🖥️  Using device: {device}")
    
    # Set seed
    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
    logger.info(f"🌱 Random seed: {seed}")
    
    # Create directories
    for directory in ['model', 'logs', 'data']:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    df = load_data(config)
    X_train, y_train, X_val, y_val, X_test, y_test, sequence_creator = prepare_data(df, config)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=config['supervised']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    # Phase 1: Pretraining (optional)
    if not args.skip_pretrain:
        model = phase1_pretrain(config, train_loader, val_loader, device)
    else:
        logger.info("⏭️  Skipping pretraining")
        model = NatronTransformer(
            n_features=config['features']['total_features'],
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            num_encoder_layers=config['model']['num_encoder_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            max_seq_length=config['model']['max_seq_length']
        )
    
    # Phase 2: Supervised Training
    model = phase2_supervised(config, model, train_loader, val_loader, device)
    
    # Phase 3: Reinforcement Learning (optional)
    if not args.skip_rl:
        model = phase3_rl(config, model, X_train, y_train, device)
    else:
        logger.info("⏭️  Skipping RL training")
    
    # Save final model
    save_final_model(model, config)
    
    # Summary
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved to: {config['api']['model_path']}")
    print(f"Scaler saved to: model/scaler.pkl")
    print(f"\nTo start the API server:")
    print(f"  python src/api_server.py")
    print(f"\nTo test predictions:")
    print(f"  curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' -d @test_request.json")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
