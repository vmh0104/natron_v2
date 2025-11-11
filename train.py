#!/usr/bin/env python3
"""
Natron Transformer - Main Training Pipeline
End-to-end training for multi-task financial trading model
"""

import torch
import yaml
import argparse
import os
import sys
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.sequence_creator import prepare_data_pipeline
from models.transformer import create_natron_model
from training.pretrain import PretrainTrainer
from training.supervised import SupervisedTrainer, load_pretrained_encoder
from training.rl_trainer import RLTrainer


def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║           🧠 NATRON TRANSFORMER V2                               ║
║           Multi-Task Financial Trading Model                     ║
║                                                                  ║
║           Phase 1: Pretraining (Unsupervised)                   ║
║           Phase 2: Supervised Fine-Tuning                        ║
║           Phase 3: Reinforcement Learning (Optional)             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def setup_environment(config):
    """Setup training environment"""
    # Set random seeds
    seed = config.get('system', {}).get('seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set device
    device = config.get('system', {}).get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"🖥️  Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create directories
    checkpoint_dir = config.get('system', {}).get('checkpoint_dir', 'model')
    log_dir = config.get('system', {}).get('log_dir', 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    return device


def main():
    parser = argparse.ArgumentParser(description='Natron Transformer Training')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data', type=str, default='data_export.csv',
                        help='Path to OHLCV data CSV')
    parser.add_argument('--skip-pretrain', action='store_true',
                        help='Skip pretraining phase')
    parser.add_argument('--skip-supervised', action='store_true',
                        help='Skip supervised training phase')
    parser.add_argument('--skip-rl', action='store_true',
                        help='Skip RL training phase')
    parser.add_argument('--load-pretrained', type=str, default=None,
                        help='Load pretrained encoder from checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Load configuration
    print(f"📋 Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print("   ✅ Configuration loaded\n")
    
    # Setup environment
    device = setup_environment(config)
    
    # ========================================
    # DATA PREPARATION
    # ========================================
    
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)
    
    train_loader, val_loader, test_loader, sequence_creator = prepare_data_pipeline(
        args.data,
        config
    )
    
    # Get number of features
    sample_batch = next(iter(train_loader))
    n_features = sample_batch[0].shape[2]
    print(f"✅ Number of features: {n_features}\n")
    
    # ========================================
    # MODEL CREATION
    # ========================================
    
    print("\n" + "="*70)
    print("STEP 2: MODEL INITIALIZATION")
    print("="*70 + "\n")
    
    model = create_natron_model(config, n_features)
    model = model.to(device)
    
    checkpoint_dir = config.get('system', {}).get('checkpoint_dir', 'model')
    
    # ========================================
    # PHASE 1: PRETRAINING (Optional)
    # ========================================
    
    pretrain_config = config.get('training', {}).get('pretrain', {})
    if pretrain_config.get('enabled', True) and not args.skip_pretrain:
        print("\n" + "="*70)
        print("STEP 3: PHASE 1 - PRETRAINING")
        print("="*70)
        
        pretrain_trainer = PretrainTrainer(model, config, device)
        model = pretrain_trainer.train(train_loader, val_loader, checkpoint_dir)
        
    elif args.load_pretrained:
        print(f"\n📂 Loading pretrained encoder from {args.load_pretrained}...")
        model = load_pretrained_encoder(model, args.load_pretrained)
    else:
        print("\n⏭️  Skipping pretraining phase")
    
    # ========================================
    # PHASE 2: SUPERVISED TRAINING
    # ========================================
    
    supervised_config = config.get('training', {}).get('supervised', {})
    if supervised_config.get('enabled', True) and not args.skip_supervised:
        print("\n" + "="*70)
        print("STEP 4: PHASE 2 - SUPERVISED FINE-TUNING")
        print("="*70)
        
        supervised_trainer = SupervisedTrainer(model, config, device)
        
        save_every = config.get('system', {}).get('save_every', 10)
        model = supervised_trainer.train(
            train_loader,
            val_loader,
            checkpoint_dir,
            save_every
        )
    else:
        print("\n⏭️  Skipping supervised training phase")
    
    # ========================================
    # PHASE 3: REINFORCEMENT LEARNING (Optional)
    # ========================================
    
    rl_config = config.get('training', {}).get('rl', {})
    if rl_config.get('enabled', False) and not args.skip_rl:
        print("\n" + "="*70)
        print("STEP 5: PHASE 3 - REINFORCEMENT LEARNING")
        print("="*70)
        
        # Get training data as numpy array
        train_data = []
        for sequences, _ in train_loader:
            train_data.append(sequences.numpy())
        train_data = np.concatenate(train_data, axis=0)
        
        rl_trainer = RLTrainer(model, config, device)
        rl_trainer.train(train_data, checkpoint_dir)
    else:
        print("\n⏭️  Skipping RL training phase (optional)")
    
    # ========================================
    # FINAL EVALUATION
    # ========================================
    
    print("\n" + "="*70)
    print("STEP 6: FINAL EVALUATION ON TEST SET")
    print("="*70 + "\n")
    
    from training.supervised import SupervisedTrainer
    evaluator = SupervisedTrainer(model, config, device)
    test_loss, test_metrics = evaluator.validate(test_loader)
    
    print("📊 TEST SET RESULTS:")
    print("="*70)
    print(f"Loss: {test_loss:.4f}\n")
    print(f"Buy Accuracy:       {test_metrics['buy_acc']:.3f} | F1: {test_metrics['buy_f1']:.3f}")
    print(f"Sell Accuracy:      {test_metrics['sell_acc']:.3f} | F1: {test_metrics['sell_f1']:.3f}")
    print(f"Direction Accuracy: {test_metrics['direction_acc']:.3f} | F1: {test_metrics['direction_f1']:.3f}")
    print(f"Regime Accuracy:    {test_metrics['regime_acc']:.3f} | F1: {test_metrics['regime_f1']:.3f}")
    print("="*70 + "\n")
    
    # ========================================
    # TRAINING COMPLETE
    # ========================================
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📁 Model saved to: {checkpoint_dir}/natron_v2.pt")
    print(f"📁 Scaler saved to: {checkpoint_dir}/scaler.pkl")
    print("\n🚀 Next steps:")
    print("   1. Start API server:")
    print(f"      python src/inference/api_server.py --model {checkpoint_dir}/natron_v2.pt")
    print("\n   2. Test predictions:")
    print("      curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' -d @sample_request.json")
    print("\n   3. Integrate with MQL5 EA for live trading")
    print("="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
