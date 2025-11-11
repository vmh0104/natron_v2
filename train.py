"""
Main Training Pipeline - End-to-End Training Orchestration
"""

import os
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse

from src.feature_engine import FeatureEngine
from src.label_generator import LabelGeneratorV2
from src.sequence_creator import SequenceCreator, SequenceDataset
from src.model import create_model
from src.pretraining import PretrainingTrainer
from src.supervised_training import SupervisedTrainer


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: dict) -> tuple:
    """Load and prepare data"""
    print("=" * 60)
    print("📊 Data Preparation")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(config['paths']['data_dir'], config['data']['input_file'])
    print(f"\n📁 Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    
    # Ensure required columns exist
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Feature extraction
    print("\n🔧 Extracting features...")
    feature_engine = FeatureEngine()
    features_df = feature_engine.extract_all_features(df)
    print(f"✅ Extracted {features_df.shape[1]} features")
    
    # Label generation
    print("\n🏷️  Generating labels...")
    label_gen = LabelGeneratorV2(
        buy_threshold=config['labels']['buy_threshold'],
        sell_threshold=config['labels']['sell_threshold'],
        neutral_buffer=config['labels']['direction_neutral_buffer'],
        balance_classes=config['labels']['balance_classes'],
        stochastic_perturbation=config['labels'].get('stochastic_perturbation', 0.05)
    )
    labels_df = label_gen.generate_labels(df, features_df)
    
    # Sequence creation
    print("\n🔗 Creating sequences...")
    seq_creator = SequenceCreator(
        sequence_length=config['data']['sequence_length'],
        normalize=config['features']['normalize']
    )
    sequences, labels = seq_creator.create_sequences(features_df, labels_df)
    
    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = seq_creator.split_data(
        sequences, labels,
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        random_seed=config['data']['random_seed']
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test, seq_creator


def train_pipeline(config: dict, skip_pretraining: bool = False):
    """Main training pipeline"""
    print("\n" + "=" * 60)
    print("🚀 Natron Transformer V2 - Training Pipeline")
    print("=" * 60)
    
    # Set random seeds
    torch.manual_seed(config['data']['random_seed'])
    np.random.seed(config['data']['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['data']['random_seed'])
    
    # Device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, seq_creator = prepare_data(config)
    
    # Create datasets
    train_dataset = SequenceDataset(X_train, y_train, config['data']['sequence_length'])
    val_dataset = SequenceDataset(X_val, y_val, config['data']['sequence_length'])
    test_dataset = SequenceDataset(X_test, y_test, config['data']['sequence_length'])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Model config
    model_config = {
        'num_features': X_train.shape[2],
        'd_model': config['model']['d_model'],
        'nhead': config['model']['nhead'],
        'num_layers': config['model']['num_layers'],
        'dim_feedforward': config['model']['dim_feedforward'],
        'dropout': config['model']['dropout'],
        'max_seq_length': config['model']['max_seq_length'],
        'activation': config['model']['activation']
    }
    
    # Create model
    print("\n🧠 Creating model...")
    model = create_model(model_config)
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Phase 1: Pretraining
    if not skip_pretraining:
        print("\n" + "=" * 60)
        print("📚 Phase 1: Pretraining")
        print("=" * 60)
        
        pretrain_config = {
            **config['training'],
            **config['pretraining']
        }
        pretrain_config.update(model_config)
        
        pretrainer = PretrainingTrainer(model, device, pretrain_config)
        pretrain_save_path = os.path.join(config['paths']['model_dir'], 'natron_pretrained.pt')
        pretrainer.train(train_loader, config['training']['num_epochs_pretrain'], pretrain_save_path)
        
        # Load pretrained weights
        checkpoint = torch.load(pretrain_save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded pretrained weights")
    
    # Phase 2: Supervised Fine-Tuning
    print("\n" + "=" * 60)
    print("🎯 Phase 2: Supervised Fine-Tuning")
    print("=" * 60)
    
    supervised_config = {
        **config['training'],
        **model_config
    }
    
    trainer = SupervisedTrainer(model, device, supervised_config)
    model_save_path = os.path.join(config['paths']['model_dir'], 'natron_v2.pt')
    trainer.train(train_loader, val_loader, 
                  config['training']['num_epochs_supervised'], 
                  model_save_path)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {model_save_path}")
    
    # Save feature scaler for inference
    scaler_path = os.path.join(config['paths']['model_dir'], 'feature_scaler.pkl')
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(seq_creator.feature_scaler, f)
    print(f"Feature scaler saved to: {scaler_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Natron Transformer V2")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--skip-pretraining", action="store_true", help="Skip pretraining phase")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Create directories
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)
    os.makedirs(config['paths']['data_dir'], exist_ok=True)
    
    train_pipeline(config, skip_pretraining=args.skip_pretraining)
