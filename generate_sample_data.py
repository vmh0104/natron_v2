"""
Generate sample OHLCV data for testing
"""

import pandas as pd
import numpy as np
import os


def generate_sample_data(n_candles: int = 1000, output_path: str = "data/data_export.csv"):
    """Generate sample OHLCV data"""
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate timestamps (M15 timeframe)
    dates = pd.date_range('2024-01-01', periods=n_candles, freq='15min')
    
    # Generate realistic price data (random walk with trend)
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.randn(n_candles) * 0.001  # 0.1% volatility
    trend = np.linspace(0, 0.1, n_candles)  # Slight upward trend
    
    close_prices = base_price + np.cumsum(returns + trend / n_candles)
    
    # Generate OHLC from close prices
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    # High/Low with some spread
    spreads = np.abs(np.random.randn(n_candles) * 0.002)
    high_prices = np.maximum(open_prices, close_prices) + spreads
    low_prices = np.minimum(open_prices, close_prices) - spreads
    
    # Generate volume (higher volume on larger moves)
    price_changes = np.abs(close_prices - open_prices)
    base_volume = 10000
    volumes = base_volume + (price_changes / base_price * 50000).astype(int)
    volumes = np.maximum(volumes, 1000)  # Minimum volume
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    # Round to reasonable precision
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].round(5)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {n_candles} candles")
    print(f"   Saved to: {output_path}")
    print(f"\n   Sample data:")
    print(df.head(10))
    print(f"\n   Statistics:")
    print(df.describe())
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample OHLCV data")
    parser.add_argument("--n-candles", type=int, default=1000, help="Number of candles")
    parser.add_argument("--output", type=str, default="data/data_export.csv", help="Output path")
    
    args = parser.parse_args()
    
    generate_sample_data(args.n_candles, args.output)
