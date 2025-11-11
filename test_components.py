"""
Example: Test Natron Model Prediction
"""
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.feature_engine import FeatureEngine
from src.label_generator import LabelGeneratorV2
from src.sequence_creator import SequenceCreator


def create_sample_data(n_samples: int = 200) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    
    # Generate random walk prices
    returns = np.random.randn(n_samples) * 0.001
    prices = 100 + np.cumsum(returns)
    
    # Create OHLCV data
    data = []
    for i in range(n_samples):
        base_price = prices[i]
        high = base_price * (1 + abs(np.random.randn() * 0.002))
        low = base_price * (1 - abs(np.random.randn() * 0.002))
        open_price = base_price + np.random.randn() * 0.001
        close = base_price + np.random.randn() * 0.001
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'time': f'2024-01-01 {i:02d}:00',
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def test_feature_engine():
    """Test feature extraction"""
    print("🧪 Testing FeatureEngine...")
    df = create_sample_data(200)
    
    feature_engine = FeatureEngine()
    features_df = feature_engine.fit_transform(df)
    
    print(f"✅ Generated {len(feature_engine.get_feature_names())} features")
    print(f"   Shape: {features_df.shape}")
    print(f"   Feature columns: {len(feature_engine.get_feature_names())}")
    
    return features_df


def test_label_generator(features_df: pd.DataFrame):
    """Test label generation"""
    print("\n🧪 Testing LabelGeneratorV2...")
    
    label_generator = LabelGeneratorV2()
    labels_df = label_generator.generate_labels(features_df)
    
    print(f"✅ Generated labels")
    print(f"   Shape: {labels_df.shape}")
    print(f"   Columns: {list(labels_df.columns)}")
    
    return labels_df


def test_sequence_creator(features_df: pd.DataFrame, labels_df: pd.DataFrame):
    """Test sequence creation"""
    print("\n🧪 Testing SequenceCreator...")
    
    sequence_creator = SequenceCreator(sequence_length=96)
    X, y = sequence_creator.create_sequences(features_df, labels_df)
    
    print(f"✅ Created sequences")
    print(f"   X shape: {X.shape}")
    print(f"   y keys: {list(y.keys())}")
    print(f"   y['buy'] shape: {y['buy'].shape}")
    
    return X, y


def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 Natron Transformer V2 - Component Tests")
    print("=" * 60)
    
    # Test feature engine
    features_df = test_feature_engine()
    
    # Test label generator
    labels_df = test_label_generator(features_df)
    
    # Test sequence creator
    X, y = test_sequence_creator(features_df, labels_df)
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("   1. Place your data_export.csv in data/ directory")
    print("   2. Run: python main.py")
    print("   3. After training, test API: python src/api.py")


if __name__ == '__main__':
    main()
