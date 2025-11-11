"""
Utility script to test the Natron Transformer system
"""

import pandas as pd
import numpy as np
import yaml
from src.feature_engine import FeatureEngine
from src.label_generator import LabelGeneratorV2
from src.sequence_creator import SequenceCreator


def test_feature_extraction():
    """Test feature extraction"""
    print("=" * 60)
    print("🧪 Testing Feature Extraction")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=200, freq='15min')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'time': dates,
        'open': 100 + np.cumsum(np.random.randn(200) * 0.1),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(1000, 10000, 200)
    })
    
    df['high'] = df['open'] + np.abs(np.random.randn(200) * 0.2)
    df['low'] = df['open'] - np.abs(np.random.randn(200) * 0.2)
    df['close'] = df['open'] + np.random.randn(200) * 0.1
    
    print(f"\n📊 Sample data shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Extract features
    feature_engine = FeatureEngine()
    features_df = feature_engine.extract_all_features(df)
    
    print(f"\n✅ Extracted {features_df.shape[1]} features")
    print(f"   Feature shape: {features_df.shape}")
    print(f"\n   Sample features:")
    print(features_df.head())
    
    return df, features_df


def test_label_generation(df, features_df):
    """Test label generation"""
    print("\n" + "=" * 60)
    print("🧪 Testing Label Generation")
    print("=" * 60)
    
    label_gen = LabelGeneratorV2(
        buy_threshold=2,
        sell_threshold=2,
        neutral_buffer=0.001,
        balance_classes=True
    )
    
    labels_df = label_gen.generate_labels(df, features_df)
    
    print(f"\n✅ Generated labels shape: {labels_df.shape}")
    print(f"   Columns: {list(labels_df.columns)}")
    
    return labels_df


def test_sequence_creation(features_df, labels_df):
    """Test sequence creation"""
    print("\n" + "=" * 60)
    print("🧪 Testing Sequence Creation")
    print("=" * 60)
    
    seq_creator = SequenceCreator(sequence_length=96, normalize=True)
    sequences, labels = seq_creator.create_sequences(features_df, labels_df)
    
    print(f"\n✅ Created sequences")
    print(f"   Sequences shape: {sequences.shape}")
    print(f"   Labels shape: {labels.shape}")
    
    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = seq_creator.split_data(
        sequences, labels,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15
    )
    
    print(f"\n✅ Data split complete")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val: {X_val.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    
    return sequences, labels


def test_model_creation():
    """Test model creation"""
    print("\n" + "=" * 60)
    print("🧪 Testing Model Creation")
    print("=" * 60)
    
    import torch
    from src.model import create_model
    
    config = {
        'num_features': 100,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'max_seq_length': 96,
        'activation': 'gelu'
    }
    
    model = create_model(config)
    
    # Test forward pass
    batch_size = 4
    seq_len = 96
    num_features = 100
    
    x = torch.randn(batch_size, seq_len, num_features)
    outputs = model(x)
    
    print(f"\n✅ Model created successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n   Forward pass test:")
    print(f"   Input shape: {x.shape}")
    print(f"   Buy prob shape: {outputs['buy_prob'].shape}")
    print(f"   Sell prob shape: {outputs['sell_prob'].shape}")
    print(f"   Direction logits shape: {outputs['direction_logits'].shape}")
    print(f"   Regime logits shape: {outputs['regime_logits'].shape}")
    
    return model


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 Natron Transformer V2 - System Tests")
    print("=" * 60)
    
    # Test 1: Feature extraction
    df, features_df = test_feature_extraction()
    
    # Test 2: Label generation
    labels_df = test_label_generation(df, features_df)
    
    # Test 3: Sequence creation
    sequences, labels = test_sequence_creation(features_df, labels_df)
    
    # Test 4: Model creation
    model = test_model_creation()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
