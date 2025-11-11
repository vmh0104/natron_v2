"""
Main Training Pipeline: End-to-End Training Script
"""
import os
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from feature_engine import FeatureEngine
from label_generator import LabelGeneratorV2
from sequence_creator import SequenceCreator
from model import NatronTransformer
from pretraining import PretrainingTrainer
from supervised_training import SupervisedTrainer


def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config: dict):
    """Create necessary directories"""
    os.makedirs(config['output']['model_dir'], exist_ok=True)
    os.makedirs(config['output']['log_dir'], exist_ok=True)
    os.makedirs(config['output']['checkpoint_dir'], exist_ok=True)


def main():
    """Main training pipeline"""
    print("="*70)
    print("🧠 Natron Transformer V2 - Multi-Task Financial Trading Model")
    print("="*70)
    
    # Load configuration
    config = load_config()
    setup_directories(config)
    
    # Set random seeds
    np.random.seed(config['data']['random_seed'])
    torch.manual_seed(config['data']['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['data']['random_seed'])
    
    # Device setup
    if config['device']['cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['device_id']}")
        print(f"\n✓ Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print(f"\n✓ Using CPU")
    
    # Step 1: Load data
    print(f"\n📊 Step 1: Loading data from {config['data']['input_file']}")
    df = pd.read_csv(config['data']['input_file'])
    print(f"   Loaded {len(df)} candles")
    
    # Ensure required columns
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Step 2: Feature Engineering
    print(f"\n🔧 Step 2: Feature Engineering (~{config['features']['num_features']} features)")
    feature_engine = FeatureEngine()
    features_df = feature_engine.generate_all_features(df)
    print(f"   Generated {features_df.shape[1]} features")
    
    # Step 3: Label Generation
    print(f"\n🏷️  Step 3: Label Generation (V2 - Bias-Reduced)")
    label_generator = LabelGeneratorV2(
        neutral_buffer=config['labeling']['neutral_buffer'],
        lookahead_candles=config['labeling']['lookahead_candles'],
        buy_threshold=config['labeling']['buy_threshold'],
        sell_threshold=config['labeling']['sell_threshold'],
        balance_labels=config['labeling']['balance_labels'],
        stochastic_perturbation=config['labeling']['stochastic_perturbation']
    )
    labels_df = label_generator.generate_labels(df, features_df)
    
    # Step 4: Sequence Creation
    print(f"\n📦 Step 4: Creating sequences (length={config['data']['sequence_length']})")
    train_dataset, val_dataset, test_dataset = SequenceCreator.create_datasets(
        features_df=features_df,
        labels_df=labels_df,
        sequence_length=config['data']['sequence_length'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        random_seed=config['data']['random_seed'],
        normalize=config['features']['normalize']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Step 5: Initialize Model
    print(f"\n🧠 Step 5: Initializing Natron Transformer")
    model = NatronTransformer(
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        activation=config['model']['activation'],
        max_seq_length=config['model']['max_seq_length'],
        num_features=features_df.shape[1]
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Step 6: Phase 1 - Pretraining
    if config['training']['pretrain']['enabled']:
        pretrain_checkpoint = os.path.join(
            config['output']['checkpoint_dir'],
            'pretrained_model.pt'
        )
        
        pretrainer = PretrainingTrainer(
            model=model,
            device=device,
            mask_probability=config['training']['pretrain']['mask_probability'],
            contrastive_temperature=config['training']['pretrain']['contrastive_temperature'],
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        pretrainer.train(
            train_loader=train_loader,
            num_epochs=config['training']['pretrain']['epochs'],
            save_path=pretrain_checkpoint
        )
        
        # Load pretrained weights
        checkpoint = torch.load(pretrain_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded pretrained weights from {pretrain_checkpoint}")
    
    # Step 7: Phase 2 - Supervised Fine-Tuning
    if config['training']['supervised']['enabled']:
        model_path = os.path.join(
            config['output']['model_dir'],
            config['output']['model_name']
        )
        
        trainer = SupervisedTrainer(
            model=model,
            device=device,
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            loss_weights=config['loss_weights']
        )
        
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['supervised']['epochs'],
            early_stopping_patience=config['training']['supervised']['early_stopping_patience'],
            save_path=model_path
        )
        
        print(f"\n✅ Training complete! Model saved to: {model_path}")
    
    # Step 8: Final Evaluation (optional)
    print(f"\n📈 Step 8: Final Evaluation on Test Set")
    trainer.model.eval()
    test_metrics, test_accuracies = trainer.validate(test_loader)
    
    print(f"\nTest Set Results:")
    print(f"  Loss: {test_metrics['total']:.6f}")
    print(f"  Accuracies:")
    print(f"    Buy:       {test_accuracies['buy']:.4f}")
    print(f"    Sell:      {test_accuracies['sell']:.4f}")
    print(f"    Direction: {test_accuracies['direction']:.4f}")
    print(f"    Regime:    {test_accuracies['regime']:.4f}")
    
    print("\n" + "="*70)
    print("🎉 Natron Transformer V2 Training Pipeline Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
