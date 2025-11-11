"""
Natron API Server - Flask-based inference endpoint
Receives 96 OHLCV candles and returns predictions
"""

from flask import Flask, request, jsonify
import torch
import numpy as np
import pandas as pd
import yaml
import pickle
from typing import Dict, List
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import NatronTransformer, create_natron_model
from data.feature_engine import FeatureEngine


app = Flask(__name__)

# Global variables for model and config
model = None
scaler = None
feature_engine = None
config = None
device = None

# Regime names
REGIME_NAMES = {
    0: "BULL_STRONG",
    1: "BULL_WEAK",
    2: "RANGE",
    3: "BEAR_WEAK",
    4: "BEAR_STRONG",
    5: "VOLATILE"
}


def load_model(model_path: str, config_path: str, scaler_path: str):
    """Load trained model and configuration"""
    global model, scaler, feature_engine, config, device
    
    print(f"🔄 Loading model from {model_path}...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"  ✅ Scaler loaded")
    
    # Initialize feature engine
    feature_engine = FeatureEngine()
    print(f"  ✅ Feature engine initialized")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Determine n_features from checkpoint or config
    # We'll use a dummy feature generation to get the count
    dummy_df = pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=200, freq='15min'),
        'open': np.random.randn(200) + 100,
        'high': np.random.randn(200) + 101,
        'low': np.random.randn(200) + 99,
        'close': np.random.randn(200) + 100,
        'volume': np.random.randint(1000, 10000, 200)
    })
    dummy_features = feature_engine.generate_all_features(dummy_df)
    n_features = dummy_features.shape[1]
    
    # Create model
    model = create_natron_model(checkpoint.get('config', config), n_features)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"  ✅ Model loaded (n_features={n_features})")
    print(f"  ✅ Model ready for inference\n")


def preprocess_input(ohlcv_data: List[Dict]) -> torch.Tensor:
    """
    Preprocess raw OHLCV data for model input
    
    Args:
        ohlcv_data: List of dicts with keys: time, open, high, low, close, volume
        
    Returns:
        Tensor of shape (1, 96, n_features)
    """
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv_data)
    
    # Ensure we have exactly 96 candles
    if len(df) < 96:
        raise ValueError(f"Expected 96 candles, got {len(df)}")
    
    # Take last 96
    df = df.tail(96).reset_index(drop=True)
    
    # Generate features
    features = feature_engine.generate_all_features(df)
    
    # Handle NaN
    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Normalize using fitted scaler
    features_normalized = scaler.transform(features)
    
    # Convert to tensor
    sequence = torch.FloatTensor(features_normalized).unsqueeze(0)  # (1, 96, n_features)
    
    return sequence


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'unknown'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Expected JSON format:
    {
        "candles": [
            {"time": "2024-01-01 00:00:00", "open": 100.0, "high": 101.0, 
             "low": 99.0, "close": 100.5, "volume": 5000},
            ...  (96 candles total)
        ]
    }
    
    Returns:
    {
        "buy_prob": 0.71,
        "sell_prob": 0.24,
        "direction_probs": [0.15, 0.69, 0.16],  # [down, up, neutral]
        "direction_pred": "up",
        "regime": "BULL_WEAK",
        "confidence": 0.82,
        "predictions": {
            "buy": 1,
            "sell": 0,
            "direction": 1,
            "regime": 1
        }
    }
    """
    try:
        # Get data
        data = request.get_json()
        
        if 'candles' not in data:
            return jsonify({'error': 'Missing "candles" field'}), 400
        
        candles = data['candles']
        
        if len(candles) < 96:
            return jsonify({'error': f'Need at least 96 candles, got {len(candles)}'}), 400
        
        # Preprocess
        sequence = preprocess_input(candles)
        sequence = sequence.to(device)
        
        # Inference
        with torch.no_grad():
            predictions = model(sequence, return_features=True)
        
        # Extract predictions
        buy_logits = predictions['buy'][0]
        sell_logits = predictions['sell'][0]
        direction_logits = predictions['direction'][0]
        regime_logits = predictions['regime'][0]
        
        # Convert to probabilities
        buy_probs = torch.softmax(buy_logits, dim=0)
        sell_probs = torch.softmax(sell_logits, dim=0)
        direction_probs = torch.softmax(direction_logits, dim=0)
        regime_probs = torch.softmax(regime_logits, dim=0)
        
        # Get predicted classes
        buy_pred = torch.argmax(buy_probs).item()
        sell_pred = torch.argmax(sell_probs).item()
        direction_pred = torch.argmax(direction_probs).item()
        regime_pred = torch.argmax(regime_probs).item()
        
        # Direction names
        direction_names = ['down', 'up', 'neutral']
        
        # Calculate confidence (average of max probabilities)
        confidence = (
            buy_probs[buy_pred].item() +
            sell_probs[sell_pred].item() +
            direction_probs[direction_pred].item() +
            regime_probs[regime_pred].item()
        ) / 4
        
        # Build response
        response = {
            'buy_prob': buy_probs[1].item(),  # Probability of buy
            'sell_prob': sell_probs[1].item(),  # Probability of sell
            'direction_probs': direction_probs.cpu().tolist(),
            'direction_pred': direction_names[direction_pred],
            'regime': REGIME_NAMES[regime_pred],
            'regime_probs': {
                REGIME_NAMES[i]: regime_probs[i].item()
                for i in range(6)
            },
            'confidence': confidence,
            'predictions': {
                'buy': buy_pred,
                'sell': sell_pred,
                'direction': direction_pred,
                'regime': regime_pred
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Expected JSON format:
    {
        "batch": [
            {"candles": [...]},  # Each with 96 candles
            {"candles": [...]},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if 'batch' not in data:
            return jsonify({'error': 'Missing "batch" field'}), 400
        
        batch = data['batch']
        results = []
        
        for item in batch:
            if 'candles' not in item:
                results.append({'error': 'Missing candles'})
                continue
            
            try:
                sequence = preprocess_input(item['candles'])
                sequence = sequence.to(device)
                
                with torch.no_grad():
                    predictions = model(sequence)
                
                # Extract predictions (same as single predict)
                buy_probs = torch.softmax(predictions['buy'][0], dim=0)
                sell_probs = torch.softmax(predictions['sell'][0], dim=0)
                direction_probs = torch.softmax(predictions['direction'][0], dim=0)
                regime_probs = torch.softmax(predictions['regime'][0], dim=0)
                
                buy_pred = torch.argmax(buy_probs).item()
                sell_pred = torch.argmax(sell_probs).item()
                direction_pred = torch.argmax(direction_probs).item()
                regime_pred = torch.argmax(regime_probs).item()
                
                direction_names = ['down', 'up', 'neutral']
                
                confidence = (
                    buy_probs[buy_pred].item() +
                    sell_probs[sell_pred].item() +
                    direction_probs[direction_pred].item() +
                    regime_probs[regime_pred].item()
                ) / 4
                
                results.append({
                    'buy_prob': buy_probs[1].item(),
                    'sell_prob': sell_probs[1].item(),
                    'direction_pred': direction_names[direction_pred],
                    'regime': REGIME_NAMES[regime_pred],
                    'confidence': confidence
                })
                
            except Exception as e:
                results.append({'error': str(e)})
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron API Server')
    parser.add_argument('--model', type=str, default='model/natron_v2.pt',
                        help='Path to trained model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--scaler', type=str, default='model/scaler.pkl',
                        help='Path to scaler file')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to bind to')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Load model
    load_model(args.model, args.config, args.scaler)
    
    # Run server
    print(f"\n🚀 Starting Natron API Server on {args.host}:{args.port}")
    print(f"   Health check: http://{args.host}:{args.port}/health")
    print(f"   Prediction:   http://{args.host}:{args.port}/predict")
    print(f"   Batch:        http://{args.host}:{args.port}/predict_batch\n")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == '__main__':
    main()
