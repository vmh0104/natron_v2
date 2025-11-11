#!/usr/bin/env python3
"""
Evaluate trained Natron model on test data
"""

import torch
import yaml
import argparse
import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from data.sequence_creator import prepare_data_pipeline
from models.transformer import create_natron_model


def plot_confusion_matrix(cm, labels, title, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  📊 Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_task(predictions, targets, task_name, class_names=None):
    """Evaluate single task"""
    print(f"\n{'='*60}")
    print(f"📊 {task_name.upper()} EVALUATION")
    print('='*60)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(targets, predictions, 
                                target_names=class_names, 
                                zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    print("\nConfusion Matrix:")
    print(cm)
    
    return cm


def main():
    parser = argparse.ArgumentParser(description='Evaluate Natron Model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to test data CSV')
    parser.add_argument('--output-dir', type=str, default='evaluation',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("📊 NATRON MODEL EVALUATION")
    print("="*70 + "\n")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Prepare data
    print("Loading and preparing data...")
    train_loader, val_loader, test_loader, _ = prepare_data_pipeline(
        args.data,
        config
    )
    
    # Get number of features
    sample_batch = next(iter(test_loader))
    n_features = sample_batch[0].shape[2]
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device)
    
    model = create_natron_model(checkpoint.get('config', config), n_features)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ Model loaded\n")
    
    # Collect predictions
    print("Running inference on test set...")
    
    all_predictions = {
        'buy': [],
        'sell': [],
        'direction': [],
        'regime': []
    }
    
    all_targets = {
        'buy': [],
        'sell': [],
        'direction': [],
        'regime': []
    }
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            
            # Predict
            outputs = model(sequences)
            
            # Get predicted classes
            for task in ['buy', 'sell', 'direction', 'regime']:
                preds = torch.argmax(outputs[task], dim=1).cpu().numpy()
                targets = labels[task].cpu().numpy()
                
                all_predictions[task].extend(preds)
                all_targets[task].extend(targets)
    
    # Convert to numpy arrays
    for task in all_predictions.keys():
        all_predictions[task] = np.array(all_predictions[task])
        all_targets[task] = np.array(all_targets[task])
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate each task
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    # Buy
    cm_buy = evaluate_task(
        all_predictions['buy'],
        all_targets['buy'],
        'Buy Signal',
        class_names=['No Buy', 'Buy']
    )
    plot_confusion_matrix(
        cm_buy,
        ['No Buy', 'Buy'],
        'Buy Signal Confusion Matrix',
        save_path=f"{args.output_dir}/buy_confusion_matrix.png"
    )
    
    # Sell
    cm_sell = evaluate_task(
        all_predictions['sell'],
        all_targets['sell'],
        'Sell Signal',
        class_names=['No Sell', 'Sell']
    )
    plot_confusion_matrix(
        cm_sell,
        ['No Sell', 'Sell'],
        'Sell Signal Confusion Matrix',
        save_path=f"{args.output_dir}/sell_confusion_matrix.png"
    )
    
    # Direction
    cm_dir = evaluate_task(
        all_predictions['direction'],
        all_targets['direction'],
        'Direction Prediction',
        class_names=['Down', 'Up', 'Neutral']
    )
    plot_confusion_matrix(
        cm_dir,
        ['Down', 'Up', 'Neutral'],
        'Direction Prediction Confusion Matrix',
        save_path=f"{args.output_dir}/direction_confusion_matrix.png"
    )
    
    # Regime
    regime_names = ['BULL_STRONG', 'BULL_WEAK', 'RANGE', 
                    'BEAR_WEAK', 'BEAR_STRONG', 'VOLATILE']
    cm_regime = evaluate_task(
        all_predictions['regime'],
        all_targets['regime'],
        'Regime Classification',
        class_names=regime_names
    )
    plot_confusion_matrix(
        cm_regime,
        regime_names,
        'Regime Classification Confusion Matrix',
        save_path=f"{args.output_dir}/regime_confusion_matrix.png"
    )
    
    # Summary
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - Confusion matrices (PNG)")
    print(f"  - Classification reports (shown above)")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
