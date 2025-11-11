"""
Flask API Server for Natron Transformer Predictions
"""
import os
import yaml
import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, List

from model import NatronTransformer
from feature_engine import FeatureEngine


class NatronPredictor:
    """
    Wrapper class for Natron model predictions
    """
    
    def __init__(self, model_path: str, config_path: str = "config.yaml", device: str = "cuda"):
        self.config = self._load_config(config_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model(model_path)
        self.feature_engine = FeatureEngine()
        
        # Regime mapping
        self.regime_map = {
            0: "BULL_STRONG",
            1: "BULL_WEAK",
            2: "RANGE",
            3: "BEAR_WEAK",
            4: "BEAR_STRONG",
            5: "VOLATILE"
        }
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model(self, model_path: str) -> NatronTransformer:
        """Load trained model"""
        # Initialize model
        model = NatronTransformer(
            d_model=self.config['model']['d_model'],
            nhead=self.config['model']['nhead'],
            num_layers=self.config['model']['num_layers'],
            dim_feedforward=self.config['model']['dim_feedforward'],
            dropout=self.config['model']['dropout'],
            activation=self.config['model']['activation'],
            max_seq_length=self.config['model']['max_seq_length'],
            num_features=self.config['features']['num_features']
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        
        return model
    
    def predict(self, candles: List[Dict]) -> Dict:
        """
        Predict from last 96 candles
        
        Args:
            candles: List of dicts with keys: time, open, high, low, close, volume
            
        Returns:
            Dictionary with predictions
        """
        if len(candles) < self.config['data']['sequence_length']:
            raise ValueError(f"Need at least {self.config['data']['sequence_length']} candles")
        
        # Convert to DataFrame
        df = pd.DataFrame(candles[-self.config['data']['sequence_length']:])
        
        # Generate features
        features_df = self.feature_engine.generate_all_features(df)
        
        # Normalize (using training statistics if available)
        # For simplicity, we'll use basic normalization here
        # In production, load saved normalization stats
        features = features_df.values.astype(np.float32)
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # Convert to tensor
        sequence = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(sequence)
        
        # Extract predictions
        buy_prob = predictions['buy'].item()
        sell_prob = predictions['sell'].item()
        direction_probs = torch.softmax(predictions['direction'], dim=1).cpu().numpy()[0]
        regime_probs = torch.softmax(predictions['regime'], dim=1).cpu().numpy()[0]
        
        # Get predicted regime
        regime_id = np.argmax(regime_probs)
        regime_name = self.regime_map[regime_id]
        
        # Confidence (average of max probabilities)
        confidence = (buy_prob + sell_prob + direction_probs.max() + regime_probs.max()) / 4
        
        return {
            "buy_prob": float(buy_prob),
            "sell_prob": float(sell_prob),
            "direction_up": float(direction_probs[1]),
            "direction_down": float(direction_probs[0]),
            "direction_neutral": float(direction_probs[2]),
            "regime": regime_name,
            "regime_id": int(regime_id),
            "regime_probs": {self.regime_map[i]: float(regime_probs[i]) for i in range(6)},
            "confidence": float(confidence)
        }


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global predictor instance
predictor = None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": predictor is not None})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Expected JSON:
    {
        "candles": [
            {"time": "...", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 1000},
            ...
        ]
    }
    """
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if 'candles' not in data:
            return jsonify({"error": "Missing 'candles' field"}), 400
        
        candles = data['candles']
        
        if not isinstance(candles, list):
            return jsonify({"error": "'candles' must be a list"}), 400
        
        # Validate candle format
        required_keys = ['open', 'high', 'low', 'close', 'volume']
        for candle in candles:
            if not all(key in candle for key in required_keys):
                return jsonify({"error": f"Candle missing required keys: {required_keys}"}), 400
        
        # Make prediction
        result = predictor.predict(candles)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint for multiple sequences
    """
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if 'sequences' not in data:
            return jsonify({"error": "Missing 'sequences' field"}), 400
        
        sequences = data['sequences']
        results = []
        
        for candles in sequences:
            result = predictor.predict(candles)
            results.append(result)
        
        return jsonify({"results": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    """Main function to run Flask server"""
    global predictor
    
    # Load configuration
    config_path = os.getenv('CONFIG_PATH', 'config.yaml')
    model_path = os.getenv('MODEL_PATH', 'models/natron_v2.pt')
    
    if not os.path.exists(model_path):
        print(f"⚠️  Warning: Model file not found at {model_path}")
        print("   Please train the model first using train.py")
        return
    
    # Load predictor
    print(f"🔄 Loading model from {model_path}...")
    try:
        predictor = NatronPredictor(
            model_path=model_path,
            config_path=config_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get server config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    host = config['api']['host']
    port = config['api']['port']
    debug = config['api']['debug']
    
    print(f"\n🚀 Starting Flask API server...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Endpoints:")
    print(f"     GET  /health")
    print(f"     POST /predict")
    print(f"     POST /predict_batch")
    print("\n" + "="*60)
    
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
