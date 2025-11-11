#!/usr/bin/env python3
"""
Generate sample OHLCV data for testing
Creates realistic synthetic trading data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_price_series(
    n_candles: int = 10000,
    initial_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0001
) -> np.ndarray:
    """Generate realistic price series using geometric Brownian motion"""
    
    # Random walk with trend
    returns = np.random.normal(trend, volatility, n_candles)
    
    # Add some autocorrelation (momentum)
    for i in range(1, len(returns)):
        returns[i] += 0.3 * returns[i-1]
    
    # Convert to prices
    prices = initial_price * np.exp(np.cumsum(returns))
    
    return prices


def generate_ohlcv(
    n_candles: int = 10000,
    initial_price: float = 100.0,
    timeframe: str = '15min',
    start_time: str = '2023-01-01 00:00:00'
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data
    
    Args:
        n_candles: Number of candles to generate
        initial_price: Starting price
        timeframe: Candle timeframe (15min, 1H, 4H, 1D)
        start_time: Start timestamp
        
    Returns:
        DataFrame with columns [time, open, high, low, close, volume]
    """
    
    print(f"🔨 Generating {n_candles} candles of {timeframe} data...")
    
    # Generate close prices
    close_prices = generate_price_series(n_candles, initial_price)
    
    # Generate OHLC
    data = []
    
    for i in range(n_candles):
        close = close_prices[i]
        
        # Open is previous close (with small gap)
        if i == 0:
            open_price = initial_price
        else:
            gap = np.random.normal(0, 0.001) * close_prices[i-1]
            open_price = close_prices[i-1] + gap
        
        # High and low
        intrabar_range = abs(close - open_price) * np.random.uniform(1.2, 2.0)
        high = max(open_price, close) + np.random.uniform(0, intrabar_range * 0.5)
        low = min(open_price, close) - np.random.uniform(0, intrabar_range * 0.5)
        
        # Volume (with some randomness)
        base_volume = 5000
        volume_factor = 1 + abs(close - open_price) / open_price * 10  # More volume on big moves
        volume = int(base_volume * volume_factor * np.random.uniform(0.5, 1.5))
        
        data.append({
            'open': round(open_price, 5),
            'high': round(high, 5),
            'low': round(low, 5),
            'close': round(close, 5),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    
    # Add timestamps
    freq_map = {
        '15min': '15T',
        '1H': '1H',
        '4H': '4H',
        '1D': '1D'
    }
    
    df['time'] = pd.date_range(
        start=start_time,
        periods=n_candles,
        freq=freq_map.get(timeframe, '15T')
    )
    
    # Reorder columns
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    
    print(f"  ✅ Generated {len(df)} candles")
    print(f"  Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    print(f"  Avg volume: {df['volume'].mean():.0f}")
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample OHLCV data')
    parser.add_argument('--output', type=str, default='data_export.csv',
                        help='Output CSV file')
    parser.add_argument('--candles', type=int, default=10000,
                        help='Number of candles to generate')
    parser.add_argument('--price', type=float, default=100.0,
                        help='Initial price')
    parser.add_argument('--timeframe', type=str, default='15min',
                        choices=['15min', '1H', '4H', '1D'],
                        help='Candle timeframe')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("📊 SAMPLE DATA GENERATOR")
    print("="*60 + "\n")
    
    # Generate data
    df = generate_ohlcv(
        n_candles=args.candles,
        initial_price=args.price,
        timeframe=args.timeframe
    )
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    
    print(f"\n✅ Data saved to {args.output}")
    print(f"📊 Shape: {df.shape}")
    print("\n" + "="*60 + "\n")
    
    # Show sample
    print("First 5 rows:")
    print(df.head())


if __name__ == '__main__':
    main()
