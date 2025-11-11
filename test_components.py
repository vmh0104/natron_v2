"""
Utility script to test Natron components
"""
import pandas as pd
import numpy as np
from feature_engine import FeatureEngine
from label_generator import LabelGeneratorV2


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    
    # Generate random walk prices
    close = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
    high = close + np.abs(np.random.randn(n_samples) * 0.3)
    low = close - np.abs(np.random.randn(n_samples) * 0.3)
    open_price = close + np.random.randn(n_samples) * 0.2
    
    df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n_samples, freq='15min'),
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    return df


def test_feature_engine():
    """Test feature engineering"""
    print("Testing FeatureEngine...")
    df = create_sample_data(500)
    
    engine = FeatureEngine()
    features = engine.generate_all_features(df)
    
    print(f"✓ Generated {features.shape[1]} features")
    print(f"✓ Features shape: {features.shape}")
    print(f"✓ Feature columns: {list(features.columns[:10])}...")
    
    return df, features


def test_label_generator(df, features_df):
    """Test label generation"""
    print("\nTesting LabelGeneratorV2...")
    
    generator = LabelGeneratorV2()
    labels = generator.generate_labels(df, features_df)
    
    print(f"✓ Generated labels shape: {labels.shape}")
    print(f"✓ Label columns: {list(labels.columns)}")
    
    return labels


def main():
    """Run all tests"""
    print("="*60)
    print("🧪 Natron Component Tests")
    print("="*60)
    
    # Test feature engineering
    df, features = test_feature_engine()
    
    # Test label generation
    labels = test_label_generator(df, features)
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)


if __name__ == "__main__":
    main()
