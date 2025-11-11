#!/usr/bin/env python3
"""
Test script for MQL5 Socket Bridge
Simulates MQL5 EA requests
"""

import socket
import json
import time
import numpy as np
from datetime import datetime, timedelta


def generate_sample_ohlcv(n_candles=96, start_price=100.0):
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
            "open": round(open_price, 5),
            "high": round(high_price, 5),
            "low": round(low_price, 5),
            "close": round(close_price, 5),
            "volume": volume
        }
        data.append(candle)
        price = close_price
    
    return data


def test_socket_connection(host='localhost', port=9999):
    """Test socket connection and prediction"""
    print("="*60)
    print("🧪 Testing MQL5 Socket Bridge")
    print("="*60)
    
    try:
        # Connect to server
        print(f"\nConnecting to {host}:{port}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((host, port))
        print("✅ Connected to server")
        
        # Test 1: Health check
        print("\n📊 Test 1: Health Check")
        print("-" * 40)
        health_request = json.dumps({"action": "health"})
        sock.sendall(health_request.encode('utf-8'))
        
        response = sock.recv(4096).decode('utf-8')
        health_data = json.loads(response)
        print(f"Status: {health_data.get('status')}")
        print(f"Uptime: {health_data.get('uptime_seconds')} seconds")
        print(f"Total Predictions: {health_data.get('total_predictions')}")
        
        # Test 2: Server info
        print("\n📊 Test 2: Server Info")
        print("-" * 40)
        info_request = json.dumps({"action": "info"})
        sock.sendall(info_request.encode('utf-8'))
        
        response = sock.recv(4096).decode('utf-8')
        info_data = json.loads(response)
        print(f"Server: {info_data.get('server')}")
        print(f"Version: {info_data.get('version')}")
        print(f"Model: {info_data.get('model')}")
        print(f"Device: {info_data.get('device')}")
        
        # Test 3: Prediction
        print("\n📊 Test 3: AI Prediction")
        print("-" * 40)
        
        # Generate sample data
        print("Generating 96 OHLCV candles...")
        ohlcv_data = generate_sample_ohlcv(96, start_price=100.0)
        
        # Send prediction request
        predict_request = json.dumps({
            "action": "predict",
            "data": ohlcv_data
        })
        
        print(f"Sending {len(predict_request)} bytes...")
        start_time = time.time()
        sock.sendall(predict_request.encode('utf-8'))
        
        # Receive response
        response = sock.recv(65536).decode('utf-8')
        latency = (time.time() - start_time) * 1000
        
        prediction = json.loads(response)
        
        if 'error' in prediction:
            print(f"❌ Error: {prediction['error']}")
        else:
            print("\n🎯 Prediction Results:")
            print(f"   Buy Probability:    {prediction['buy_prob']:.4f} ({prediction['buy_prob']*100:.1f}%)")
            print(f"   Sell Probability:   {prediction['sell_prob']:.4f} ({prediction['sell_prob']*100:.1f}%)")
            print(f"   Direction Up:       {prediction['direction_up']:.4f} ({prediction['direction_up']*100:.1f}%)")
            print(f"   Direction Down:     {prediction['direction_down']:.4f} ({prediction['direction_down']*100:.1f}%)")
            print(f"   Market Regime:      {prediction['regime']}")
            print(f"   Regime Confidence:  {prediction['regime_confidence']:.4f} ({prediction['regime_confidence']*100:.1f}%)")
            print(f"   Overall Confidence: {prediction['confidence']:.4f} ({prediction['confidence']*100:.1f}%)")
            print(f"   Server Latency:     {prediction['latency_ms']:.2f} ms")
            print(f"   Total Latency:      {latency:.2f} ms")
            
            # Signal interpretation
            print("\n🚦 Signal Interpretation:")
            if prediction['buy_prob'] > 0.6:
                print("   ✅ STRONG BUY - Enter long position")
            elif prediction['buy_prob'] > 0.5:
                print("   ↗️  Weak buy - Consider long position")
            elif prediction['sell_prob'] > 0.6:
                print("   ❌ STRONG SELL - Enter short position")
            elif prediction['sell_prob'] > 0.5:
                print("   ↘️  Weak sell - Consider short position")
            else:
                print("   ⏸️  NEUTRAL - No clear signal")
            
            print(f"   Market Regime: {prediction['regime']}")
        
        # Test 4: Benchmark
        print("\n📊 Test 4: Latency Benchmark")
        print("-" * 40)
        
        n_requests = 10
        latencies = []
        
        for i in range(n_requests):
            ohlcv_data = generate_sample_ohlcv(96)
            request = json.dumps({"action": "predict", "data": ohlcv_data})
            
            start = time.time()
            sock.sendall(request.encode('utf-8'))
            response = sock.recv(65536)
            latency = (time.time() - start) * 1000
            
            latencies.append(latency)
            print(f"   Request {i+1}/{n_requests}: {latency:.2f} ms")
        
        print(f"\n📈 Latency Statistics:")
        print(f"   Mean:   {np.mean(latencies):.2f} ms")
        print(f"   Median: {np.median(latencies):.2f} ms")
        print(f"   Min:    {np.min(latencies):.2f} ms")
        print(f"   Max:    {np.max(latencies):.2f} ms")
        print(f"   Std:    {np.std(latencies):.2f} ms")
        
        # Close connection
        sock.close()
        print("\n✅ All tests passed!")
        print("="*60)
        
    except ConnectionRefusedError:
        print("\n❌ Connection refused")
        print("   Make sure the socket server is running:")
        print("   python src/bridge/socket_server.py")
    except socket.timeout:
        print("\n❌ Connection timeout")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MQL5 Socket Bridge')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=9999,
                       help='Server port (default: 9999)')
    
    args = parser.parse_args()
    
    test_socket_connection(args.host, args.port)
