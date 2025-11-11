"""
Natron Transformer - Main Orchestrator
End-to-End Pipeline: Data → Pretrain → Train → RL → Deploy
"""

import argparse
import os
import sys
import torch
import numpy as np

from config import load_config, save_config
from feature_engine import load_and_prepare_data
from label_generator_v2 import create_labels
from dataset import create_dataloaders, create_pretraining_dataloader, save_scaler
from pretrain import run_pretraining
from train import run_training
from rl_trainer import train_rl
from api_server import run_server


def setup_directories(config):
    """Create necessary directories"""
    dirs = [
        config.output_dir,
        config.model_dir,
        config.log_dir,
        config.pretrain.checkpoint_dir,
        config.train.checkpoint_dir,
        config.rl.checkpoint_dir
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    
    print("📁 Directories created")


def run_full_pipeline(args):
    """Run complete training pipeline"""
    print("=" * 70)
    print("🧠 NATRON TRANSFORMER - End-to-End Training Pipeline")
    print("=" * 70)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()
    
    # Override config with command line args
    if args.data:
        config.data.csv_path = args.data
    if args.epochs:
        config.train.epochs = args.epochs
    if args.pretrain_epochs:
        config.pretrain.epochs = args.pretrain_epochs
    if args.device:
        config.device = args.device
    if args.batch_size:
        config.data.batch_size = args.batch_size
    
    # Setup
    setup_directories(config)
    
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Save config
    config_path = os.path.join(config.output_dir, 'config.yaml')
    save_config(config, config_path)
    print(f"💾 Configuration saved to {config_path}\n")
    
    # ========== Phase 0: Data Preparation ==========
    print("\n" + "=" * 70)
    print("📊 PHASE 0: Data Preparation")
    print("=" * 70)
    
    raw_df, features_df = load_and_prepare_data(config.data.csv_path)
    print(f"✅ Data loaded: {len(raw_df)} candles, {features_df.shape[1]} features")
    
    # ========== Phase 1: Unsupervised Pretraining ==========
    if config.pretrain.enabled and not args.skip_pretrain:
        print("\n" + "=" * 70)
        print("🔥 PHASE 1: Unsupervised Pretraining")
        print("=" * 70)
        
        run_pretraining(config)
        pretrain_path = os.path.join(config.pretrain.checkpoint_dir, 'best_pretrain.pt')
    else:
        print("\n⏭️  Skipping Phase 1: Pretraining")
        pretrain_path = None
    
    # ========== Phase 2: Supervised Training ==========
    if not args.skip_train:
        print("\n" + "=" * 70)
        print("🎯 PHASE 2: Supervised Fine-Tuning")
        print("=" * 70)
        
        run_training(config, pretrain_path)
        model_path = os.path.join(config.model_dir, 'natron_v2.pt')
    else:
        print("\n⏭️  Skipping Phase 2: Training")
        model_path = args.model_path or os.path.join(config.model_dir, 'natron_v2.pt')
    
    # ========== Phase 3: Reinforcement Learning ==========
    if config.rl.enabled and not args.skip_rl:
        print("\n" + "=" * 70)
        print("💪 PHASE 3: Reinforcement Learning")
        print("=" * 70)
        
        train_rl(config, model_path)
    else:
        print("\n⏭️  Skipping Phase 3: RL Training")
    
    # ========== Complete ==========
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n📦 Artifacts saved to:")
    print(f"   Model: {config.model_dir}")
    print(f"   Checkpoints: {config.train.checkpoint_dir}")
    print(f"   Logs: {config.log_dir}")
    print(f"\n🚀 To start the API server:")
    print(f"   python api_server.py")
    print(f"\n🔥 To run inference:")
    print(f"   python inference.py <path_to_data>")
    print("=" * 70 + "\n")


def run_inference(args):
    """Run inference on new data"""
    print("🔮 Running Inference...")
    
    # Load config
    config = load_config(args.config) if args.config else load_config()
    
    # Load model
    from model import create_model
    from dataset import load_scaler
    from feature_engine import FeatureEngine
    
    model_path = args.model_path or config.inference.model_path
    scaler_path = os.path.join(config.output_dir, 'scaler.pkl')
    
    print(f"📂 Loading model from {model_path}")
    model = create_model(config)
    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()
    
    print(f"📂 Loading scaler from {scaler_path}")
    scaler = load_scaler(scaler_path)
    
    # Load data
    import pandas as pd
    print(f"📂 Loading data from {args.data}")
    df = pd.read_csv(args.data)
    
    # Extract features
    engine = FeatureEngine()
    features = engine.compute_all_features(df)
    
    # Normalize
    features_normalized = scaler.transform(features.values)
    
    # Create sequences
    seq_len = config.data.sequence_length
    n_samples = len(features_normalized) - seq_len
    
    print(f"\n🔮 Running inference on {n_samples} sequences...")
    
    results = []
    
    for i in range(0, n_samples, 100):  # Process in batches
        batch_end = min(i + 100, n_samples)
        batch_sequences = []
        
        for j in range(i, batch_end):
            seq = features_normalized[j:j+seq_len]
            batch_sequences.append(seq)
        
        # Convert to tensor
        batch_tensor = torch.from_numpy(np.array(batch_sequences)).float().to(config.device)
        
        # Inference
        with torch.no_grad():
            outputs = model(batch_tensor)
        
        # Parse outputs
        buy_probs = torch.softmax(outputs['buy'], dim=1)[:, 1].cpu().numpy()
        sell_probs = torch.softmax(outputs['sell'], dim=1)[:, 1].cpu().numpy()
        direction_preds = outputs['direction'].argmax(dim=1).cpu().numpy()
        regime_preds = outputs['regime'].argmax(dim=1).cpu().numpy()
        
        for k in range(len(buy_probs)):
            results.append({
                'index': i + k,
                'buy_prob': float(buy_probs[k]),
                'sell_prob': float(sell_probs[k]),
                'direction': int(direction_preds[k]),
                'regime': int(regime_preds[k])
            })
        
        if (i + 100) % 1000 == 0:
            print(f"   Processed {i + 100}/{n_samples} sequences...")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = args.output or 'inference_results.csv'
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Inference complete!")
    print(f"   Results saved to {output_path}")
    print(f"\n📊 Summary:")
    print(f"   Avg Buy Prob: {results_df['buy_prob'].mean():.3f}")
    print(f"   Avg Sell Prob: {results_df['sell_prob'].mean():.3f}")
    print(f"   Direction Distribution:")
    print(results_df['direction'].value_counts())
    print(f"   Regime Distribution:")
    print(results_df['regime'].value_counts())


def run_api(args):
    """Start API server"""
    config = load_config(args.config) if args.config else load_config()
    
    model_path = args.model_path or config.inference.model_path
    scaler_path = os.path.join(config.output_dir, 'scaler.pkl')
    
    run_server(config, model_path, scaler_path)


def main():
    parser = argparse.ArgumentParser(
        description='Natron Transformer - Multi-Task Financial Trading Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py train --data data_export.csv --epochs 100

  # Run with pretraining
  python main.py train --data data_export.csv --pretrain-epochs 50 --epochs 100

  # Skip pretraining
  python main.py train --data data_export.csv --skip-pretrain

  # Run inference
  python main.py infer --data test_data.csv --model-path model/natron_v2.pt

  # Start API server
  python main.py serve --model-path model/natron_v2.pt

  # Use custom config
  python main.py train --config my_config.yaml --data data_export.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Run training pipeline')
    train_parser.add_argument('--data', type=str, help='Path to data_export.csv')
    train_parser.add_argument('--config', type=str, help='Path to config YAML file')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--pretrain-epochs', type=int, help='Number of pretraining epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device to use')
    train_parser.add_argument('--skip-pretrain', action='store_true', help='Skip pretraining phase')
    train_parser.add_argument('--skip-train', action='store_true', help='Skip supervised training')
    train_parser.add_argument('--skip-rl', action='store_true', help='Skip RL training')
    train_parser.add_argument('--model-path', type=str, help='Path to existing model')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--data', type=str, required=True, help='Path to input data CSV')
    infer_parser.add_argument('--model-path', type=str, help='Path to trained model')
    infer_parser.add_argument('--config', type=str, help='Path to config YAML file')
    infer_parser.add_argument('--output', type=str, help='Path to output CSV file')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--model-path', type=str, help='Path to trained model')
    serve_parser.add_argument('--config', type=str, help='Path to config YAML file')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    serve_parser.add_argument('--port', type=int, default=5000, help='Port number')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        run_full_pipeline(args)
    elif args.command == 'infer':
        run_inference(args)
    elif args.command == 'serve':
        run_api(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
