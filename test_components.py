"""
Utility script to test model inference without full training
"""

import pandas as pd
import numpy as np
import torch
import yaml
from src.model import NatronTransformer
from src.feature_engine import FeatureEngine
from src.sequence_creator import SequenceCreator


def create_dummy_data(n_samples: int = 200) -> pd.DataFrame:
    """Create dummy OHLCV data for testing"""
    np.random.seed(42)
    
    # Generate synthetic price data
    base_price = 1.0850
    prices = []
    current_price = base_price
    
    for i in range(n_samples):
        change = np.random.normal(0, 0.0005)
        current_price += change
        prices.append(current_price)
    
    # Create OHLCV data
    data = []
    for i, close in enumerate(prices):
        high = close + abs(np.random.normal(0, 0.0002))
        low = close - abs(np.random.normal(0, 0.0002))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.randint(1000, 5000)
        
        data.append({
            'time': i,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def test_feature_engine():
    """Test feature generation"""
    print("Testing FeatureEngine...")
    df = create_dummy_data(200)
    
    feature_engine = FeatureEngine()
    features_df = feature_engine.generate_all_features(df)
    
    feature_cols = [col for col in features_df.columns if col != 'time']
    print(f"✅ Generated {len(feature_cols)} features")
    print(f"Feature columns: {feature_cols[:10]}...")
    
    return features_df


def test_label_generator(features_df: pd.DataFrame, df: pd.DataFrame):
    """Test label generation"""
    print("\nTesting LabelGeneratorV2...")
    
    from src.label_generator import LabelGeneratorV2
    
    # Merge for labeling
    labeling_df = pd.merge(df[['time', 'open', 'high', 'low', 'close', 'volume']],
                          features_df, on='time', how='inner')
    
    label_generator = LabelGeneratorV2()
    labels_df = label_generator.generate_labels(labeling_df)
    
    print(f"✅ Generated labels: {list(labels_df.columns)}")
    print(f"Label shapes: {labels_df.shape}")
    
    return labels_df


def test_sequence_creator(features_df: pd.DataFrame, labels_df: pd.DataFrame):
    """Test sequence creation"""
    print("\nTesting SequenceCreator...")
    
    sequence_creator = SequenceCreator(sequence_length=96)
    X, y = sequence_creator.create_sequences(features_df, labels_df)
    
    print(f"✅ Created sequences: X.shape={X.shape}, y.shape={y.shape}")
    
    return X, y


def test_model():
    """Test model forward pass"""
    print("\nTesting NatronTransformer...")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dummy data
    df = create_dummy_data(200)
    feature_engine = FeatureEngine()
    features_df = feature_engine.generate_all_features(df)
    
    # Get input dimension
    feature_cols = [col for col in features_df.columns if col != 'time']
    input_dim = len(feature_cols)
    
    # Create model
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
    
    # Create sequence
    sequence_creator = SequenceCreator(sequence_length=96)
    X, _ = sequence_creator.create_sequences(features_df, None)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(X[:1])  # Batch size 1
        predictions = model(x_tensor)
    
    print(f"✅ Model forward pass successful")
    print(f"Buy prob: {predictions['buy_prob'].item():.4f}")
    print(f"Sell prob: {predictions['sell_prob'].item():.4f}")
    print(f"Direction probs: {predictions['direction'][0].numpy()}")
    print(f"Regime probs: {predictions['regime'][0].numpy()}")
    
    return model


def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 Natron Transformer V2 - Component Tests")
    print("=" * 60)
    
    # Create dummy data
    df = create_dummy_data(200)
    
    # Test components
    features_df = test_feature_engine()
    labels_df = test_label_generator(features_df, df)
    X, y = test_sequence_creator(features_df, labels_df)
    model = test_model()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {num_params:,}")


if __name__ == "__main__":
    main()
