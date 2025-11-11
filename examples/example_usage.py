"""
Example usage of Natron Transformer API
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# API endpoint
API_URL = "http://localhost:5000"


def create_sample_candles(n=96):
    """Create sample OHLCV data"""
    dates = pd.date_range(datetime.now() - timedelta(hours=n), periods=n, freq='1H')
    
    # Generate realistic price movement
    base_price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    candles = []
    for i, date in enumerate(dates):
        candle = {
            'time': date.strftime('%Y-%m-%d %H:%M:%S'),
            'open': float(base_price[i] + np.random.randn() * 0.2),
            'high': float(base_price[i] + abs(np.random.randn()) * 0.5),
            'low': float(base_price[i] - abs(np.random.randn()) * 0.5),
            'close': float(base_price[i]),
            'volume': int(np.random.randint(1000, 10000))
        }
        candles.append(candle)
    
    return candles


def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_info():
    """Test info endpoint"""
    print("Testing /info endpoint...")
    response = requests.get(f"{API_URL}/info")
    print(f"Status: {response.status_code}")
    print(f"Model Info: {response.json()}")
    print()


def test_predict():
    """Test prediction endpoint"""
    print("Testing /predict endpoint...")
    
    # Create sample data
    candles = create_sample_candles(96)
    
    # Make request
    payload = {'candles': candles}
    response = requests.post(f"{API_URL}/predict", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        
        print("=" * 70)
        print("PREDICTION RESULTS")
        print("=" * 70)
        print(f"\n🟢 BUY Probability:  {result['buy_prob']:.3f}")
        print(f"🔴 SELL Probability: {result['sell_prob']:.3f}")
        
        print(f"\n📈 Direction: {result['direction']}")
        print(f"   Probabilities:")
        for k, v in result['direction_probs'].items():
            print(f"      {k:8s}: {v:.3f}")
        
        print(f"\n🌊 Regime: {result['regime']}")
        print(f"   Probabilities:")
        for k, v in result['regime_probs'].items():
            print(f"      {k:12s}: {v:.3f}")
        
        print(f"\n⭐ Confidence: {result['confidence']:.3f}")
        print("=" * 70)
        
        # Trading decision logic example
        print("\n💡 TRADING DECISION:")
        if result['buy_prob'] > 0.7 and result['confidence'] > 0.75:
            print("   ✅ STRONG BUY SIGNAL")
        elif result['sell_prob'] > 0.7 and result['confidence'] > 0.75:
            print("   ❌ STRONG SELL SIGNAL")
        elif result['buy_prob'] > 0.6:
            print("   🟢 Moderate Buy Signal")
        elif result['sell_prob'] > 0.6:
            print("   🔴 Moderate Sell Signal")
        else:
            print("   ⏸️  No Clear Signal - Wait")
        
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")


def batch_predict():
    """Example of batch prediction"""
    print("\nTesting batch predictions...")
    
    results = []
    for i in range(5):
        candles = create_sample_candles(96)
        payload = {'candles': candles}
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            results.append({
                'iteration': i + 1,
                'buy_prob': result['buy_prob'],
                'sell_prob': result['sell_prob'],
                'direction': result['direction'],
                'regime': result['regime']
            })
    
    # Display batch results
    print("\nBatch Prediction Results:")
    print("-" * 70)
    for r in results:
        print(f"#{r['iteration']} | Buy: {r['buy_prob']:.2f} | Sell: {r['sell_prob']:.2f} | "
              f"Dir: {r['direction']:7s} | Regime: {r['regime']}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NATRON TRANSFORMER - API USAGE EXAMPLES")
    print("=" * 70 + "\n")
    
    try:
        # Test endpoints
        test_health()
        test_info()
        test_predict()
        batch_predict()
        
        print("\n✅ All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server.")
        print("   Make sure the server is running: python main.py --mode api")
    except Exception as e:
        print(f"❌ Error: {e}")
