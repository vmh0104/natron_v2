#!/usr/bin/env python3
"""
Simple script to test the Natron API
"""

import requests
import json
import time
import numpy as np
from datetime import datetime, timedelta

API_URL = "http://localhost:5000"


def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_info():
    """Test info endpoint"""
    print("\nTesting /info endpoint...")
    response = requests.get(f"{API_URL}/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def generate_sample_data(n_candles=96, start_price=100.0):
    """Generate sample OHLCV data"""
    data = []
    current_time = datetime.now()
    price = start_price
    
    for i in range(n_candles):
        # Random walk
        change = np.random.randn() * 0.5
        price += change
        
        open_price = price
        close_price = price + np.random.randn() * 0.3
        high_price = max(open_price, close_price) + abs(np.random.randn() * 0.2)
        low_price = min(open_price, close_price) - abs(np.random.randn() * 0.2)
        volume = int(np.random.randint(1000, 5000))
        
        candle = {
            "time": (current_time + timedelta(minutes=15*i)).strftime("%Y-%m-%d %H:%M:%S"),
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": volume
        }
        data.append(candle)
        price = close_price
    
    return data


def test_predict():
    """Test predict endpoint"""
    print("\nTesting /predict endpoint...")
    
    # Generate sample data
    data = generate_sample_data(96, start_price=100.0)
    
    request_data = {"data": data}
    
    # Measure latency
    start_time = time.time()
    response = requests.post(
        f"{API_URL}/predict",
        json=request_data,
        headers={"Content-Type": "application/json"}
    )
    latency = (time.time() - start_time) * 1000
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n📊 Prediction Results:")
        print(f"   Buy Probability:    {result['buy_prob']:.4f}")
        print(f"   Sell Probability:   {result['sell_prob']:.4f}")
        print(f"   Direction Up:       {result['direction_up']:.4f}")
        print(f"   Direction Down:     {result['direction_down']:.4f}")
        print(f"   Market Regime:      {result['regime']}")
        print(f"   Regime Confidence:  {result['regime_confidence']:.4f}")
        print(f"   Overall Confidence: {result['confidence']:.4f}")
        print(f"   Server Latency:     {result['latency_ms']:.2f} ms")
        print(f"   Total Latency:      {latency:.2f} ms")
        
        # Interpret signal
        print(f"\n🎯 Signal Interpretation:")
        if result['buy_prob'] > 0.6:
            print("   ✅ STRONG BUY signal")
        elif result['buy_prob'] > 0.5:
            print("   ↗️  Weak buy signal")
        elif result['sell_prob'] > 0.6:
            print("   ❌ STRONG SELL signal")
        elif result['sell_prob'] > 0.5:
            print("   ↘️  Weak sell signal")
        else:
            print("   ⏸️  NEUTRAL - no clear signal")
        
        print(f"   Market State: {result['regime']}")
        
        return True
    else:
        print(f"Error: {response.text}")
        return False


def benchmark_latency(n_requests=10):
    """Benchmark API latency"""
    print(f"\n⏱️  Benchmarking latency ({n_requests} requests)...")
    
    latencies = []
    
    for i in range(n_requests):
        data = generate_sample_data(96)
        request_data = {"data": data}
        
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/predict",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            latencies.append(latency)
            print(f"   Request {i+1}/{n_requests}: {latency:.2f} ms")
        else:
            print(f"   Request {i+1}/{n_requests}: FAILED")
    
    if latencies:
        print(f"\n📈 Latency Statistics:")
        print(f"   Mean:   {np.mean(latencies):.2f} ms")
        print(f"   Median: {np.median(latencies):.2f} ms")
        print(f"   Min:    {np.min(latencies):.2f} ms")
        print(f"   Max:    {np.max(latencies):.2f} ms")
        print(f"   Std:    {np.std(latencies):.2f} ms")


def main():
    """Run all tests"""
    print("="*60)
    print("🧪 Natron Transformer API Test Suite")
    print("="*60)
    
    try:
        # Test health
        if not test_health():
            print("\n❌ Health check failed. Is the server running?")
            print("   Start the server with: python src/api_server.py")
            return
        
        # Test info
        test_info()
        
        # Test prediction
        if not test_predict():
            print("\n❌ Prediction test failed")
            return
        
        # Benchmark
        benchmark_latency(10)
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API server")
        print("   Make sure the server is running:")
        print("   python src/api_server.py")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
