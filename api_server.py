"""
Flask API Server for Natron Transformer Predictions
"""

import os
import yaml
import pickle
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

from src.model import create_model
from src.feature_engine import FeatureEngine


app = Flask(__name__)
CORS(app)

# Global variables
model = None
feature_engine = None
feature_scaler = None
config = None
device = None
regime_names = {
    0: "BULL_STRONG",
    1: "BULL_WEAK",
    2: "RANGE",
    3: "BEAR_WEAK",
    4: "BEAR_STRONG",
    5: "VOLATILE"
}


def load_model(model_path: str, config_path: str = "config.yaml"):
    """Load trained model and scaler"""
    global model, feature_engine, feature_scaler, config, device
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Load feature scaler
    scaler_path = os.path.join(os.path.dirname(model_path), 'feature_scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            feature_scaler = pickle.load(f)
    else:
        print("⚠️  Feature scaler not found, will normalize on-the-fly")
    
    # Create model
    model_config = {
        'num_features': config['features']['num_features'],
        'd_model': config['model']['d_model'],
        'nhead': config['model']['nhead'],
        'num_layers': config['model']['num_layers'],
        'dim_feedforward': config['model']['dim_feedforward'],
        'dropout': config['model']['dropout'],
        'max_seq_length': config['model']['max_seq_length'],
        'activation': config['model']['activation']
    }
    
    model = create_model(model_config)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    # Initialize feature engine
    feature_engine = FeatureEngine()
    
    print(f"✅ Model loaded from {model_path}")
    print(f"   Device: {device}")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint
    
    Expected input:
    {
        "candles": [
            {"time": "...", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 1000},
            ...
        ]  # Must have at least 96 candles
    }
    
    Returns:
    {
        "buy_prob": 0.71,
        "sell_prob": 0.24,
        "direction": "Up",  # "Up", "Down", "Neutral"
        "direction_probs": [0.15, 0.69, 0.16],  # [Down, Up, Neutral]
        "regime": "BULL_WEAK",
        "regime_probs": [0.1, 0.4, 0.2, 0.15, 0.1, 0.05],
        "confidence": 0.82
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        candles = data.get('candles', [])
        
        if len(candles) < config['data']['sequence_length']:
            return jsonify({
                "error": f"Need at least {config['data']['sequence_length']} candles, got {len(candles)}"
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return jsonify({"error": f"Missing required columns: {required_cols}"}), 400
        
        # Extract features
        features_df = feature_engine.extract_all_features(df)
        
        # Take last N candles (sequence_length)
        sequence_length = config['data']['sequence_length']
        features_sequence = features_df.iloc[-sequence_length:].values
        
        # Normalize
        if feature_scaler is not None:
            features_sequence = feature_scaler.transform(features_sequence)
        else:
            # Fallback normalization
            features_sequence = (features_sequence - features_sequence.mean(axis=0)) / (features_sequence.std(axis=0) + 1e-8)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_sequence).unsqueeze(0).to(device)  # (1, seq_len, num_features)
        
        # Predict
        with torch.no_grad():
            outputs = model(features_tensor)
        
        # Extract predictions
        buy_prob = outputs['buy_prob'].item()
        sell_prob = outputs['sell_prob'].item()
        
        # Direction (3-class)
        direction_logits = outputs['direction_logits']
        direction_probs = torch.exp(direction_logits).cpu().numpy()[0]
        direction_idx = np.argmax(direction_probs)
        direction_map = {0: "Down", 1: "Up", 2: "Neutral"}
        direction = direction_map[direction_idx]
        
        # Regime (6-class)
        regime_logits = outputs['regime_logits']
        regime_probs = torch.exp(regime_logits).cpu().numpy()[0]
        regime_idx = np.argmax(regime_probs)
        regime = regime_names[regime_idx]
        
        # Confidence (max probability across all tasks)
        confidence = max(buy_prob, sell_prob, direction_probs.max(), regime_probs.max())
        
        return jsonify({
            "buy_prob": float(buy_prob),
            "sell_prob": float(sell_prob),
            "direction": direction,
            "direction_probs": direction_probs.tolist(),
            "regime": regime,
            "regime_probs": regime_probs.tolist(),
            "confidence": float(confidence)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Natron Transformer API Server")
    parser.add_argument("--model-path", type=str, default="model/natron_v2.pt", help="Path to model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=5000, help="Port")
    
    args = parser.parse_args()
    
    # Load model
    load_model(args.model_path, args.config)
    
    # Run server
    print(f"\n🚀 Starting API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
