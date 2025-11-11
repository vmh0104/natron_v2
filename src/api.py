"""
Natron Inference API - Flask REST API for Real-Time Predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import pandas as pd
import yaml
import pickle
from typing import Dict, List
import os

from model import NatronTransformer
from feature_engine import FeatureEngine


app = Flask(__name__)
CORS(app)


class NatronAPI:
    """Natron Inference API Handler"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize API with model and feature engine.
        
        Args:
            config_path: Path to configuration file
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Device
        self.device = torch.device(
            self.config['api']['device'] 
            if torch.cuda.is_available() 
            else 'cpu'
        )
        
        print(f"🖥️  Using device: {self.device}")
        
        # Load feature engine
        self.feature_engine = FeatureEngine()
        
        # Load scaler
        scaler_path = os.path.join('model', 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Scaler loaded from {scaler_path}")
        else:
            print("⚠️  Warning: Scaler not found. Features will not be normalized.")
            self.scaler = None
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Regime names
        self.regime_names = {
            0: "BULL_STRONG",
            1: "BULL_WEAK",
            2: "RANGE",
            3: "BEAR_WEAK",
            4: "BEAR_STRONG",
            5: "VOLATILE"
        }
        
        # Direction names
        self.direction_names = {
            0: "DOWN",
            1: "UP",
            2: "NEUTRAL"
        }
        
        print("✅ Natron API initialized successfully!")
    
    def _load_model(self) -> NatronTransformer:
        """Load trained model"""
        model_path = self.config['api']['model_path']
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        print(f"📥 Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get number of features from config or checkpoint
        num_features = self.config['features']['num_features']
        
        # Create model
        model = NatronTransformer(
            num_features=num_features,
            d_model=self.config['model']['d_model'],
            nhead=self.config['model']['nhead'],
            num_encoder_layers=self.config['model']['num_encoder_layers'],
            dim_feedforward=self.config['model']['dim_feedforward'],
            dropout=self.config['model']['dropout'],
            max_seq_length=self.config['model']['max_seq_length']
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print("✅ Model loaded successfully!")
        
        return model
    
    def predict(self, ohlcv_data: pd.DataFrame) -> Dict:
        """
        Make prediction on 96-candle OHLCV data.
        
        Args:
            ohlcv_data: DataFrame with columns [time, open, high, low, close, volume]
                        Must have exactly 96 rows
        
        Returns:
            Dictionary with predictions:
                - buy_prob: float [0, 1]
                - sell_prob: float [0, 1]
                - direction: str (UP/DOWN/NEUTRAL)
                - direction_probs: dict with probabilities
                - regime: str (regime name)
                - regime_probs: dict with probabilities
                - confidence: float [0, 1]
        """
        # Validate input
        if len(ohlcv_data) != self.config['data']['sequence_length']:
            raise ValueError(
                f"Expected {self.config['data']['sequence_length']} candles, "
                f"got {len(ohlcv_data)}"
            )
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in ohlcv_data.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Need: {required_cols}")
        
        # Extract features
        features = self.feature_engine.extract_all_features(ohlcv_data)
        
        # Normalize if scaler available
        if self.scaler is not None:
            features_array = self.scaler.transform(features.values)
        else:
            features_array = features.values
        
        # Convert to tensor
        sequence = torch.FloatTensor(features_array).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(sequence)
        
        # Extract predictions
        buy_prob = outputs['buy_prob'].item()
        sell_prob = outputs['sell_prob'].item()
        
        # Direction
        direction_probs = torch.softmax(outputs['direction_logits'], dim=1).cpu().numpy()[0]
        direction_idx = direction_probs.argmax()
        direction = self.direction_names[direction_idx]
        
        # Regime
        regime_probs = torch.softmax(outputs['regime_logits'], dim=1).cpu().numpy()[0]
        regime_idx = regime_probs.argmax()
        regime = self.regime_names[regime_idx]
        
        # Confidence (average of max probabilities)
        confidence = (
            max(buy_prob, 1 - buy_prob) +
            max(sell_prob, 1 - sell_prob) +
            direction_probs.max() +
            regime_probs.max()
        ) / 4.0
        
        # Build response
        result = {
            'buy_prob': float(buy_prob),
            'sell_prob': float(sell_prob),
            'direction': direction,
            'direction_probs': {
                'DOWN': float(direction_probs[0]),
                'UP': float(direction_probs[1]),
                'NEUTRAL': float(direction_probs[2])
            },
            'regime': regime,
            'regime_probs': {
                self.regime_names[i]: float(regime_probs[i])
                for i in range(6)
            },
            'confidence': float(confidence)
        }
        
        return result


# Global API instance
api = None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': api is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    
    Expects JSON with 96 candles:
    {
        "candles": [
            {"time": "2023-01-01 00:00:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000},
            ...
        ]
    }
    
    Returns:
    {
        "buy_prob": 0.71,
        "sell_prob": 0.24,
        "direction": "UP",
        "direction_probs": {"DOWN": 0.15, "UP": 0.69, "NEUTRAL": 0.16},
        "regime": "BULL_WEAK",
        "regime_probs": {...},
        "confidence": 0.82
    }
    """
    try:
        # Parse request
        data = request.get_json()
        
        if 'candles' not in data:
            return jsonify({'error': 'Missing "candles" field'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data['candles'])
        
        # Make prediction
        result = api.predict(df)
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    """
    Prediction endpoint for CSV data.
    
    Expects multipart/form-data with CSV file containing OHLCV data (96 rows).
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Make prediction
        result = api.predict(df)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


@app.route('/info', methods=['GET'])
def model_info():
    """Return model information"""
    return jsonify({
        'model': 'Natron Transformer V2',
        'sequence_length': api.config['data']['sequence_length'],
        'features': api.config['features']['num_features'],
        'd_model': api.config['model']['d_model'],
        'layers': api.config['model']['num_encoder_layers'],
        'heads': api.config['model']['nhead'],
        'device': str(api.device)
    })


def run_server(config_path: str = 'config.yaml'):
    """
    Start the Flask API server.
    
    Args:
        config_path: Path to configuration file
    """
    global api
    
    print("\n" + "="*70)
    print("🚀 NATRON INFERENCE API")
    print("="*70)
    
    # Initialize API
    api = NatronAPI(config_path)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    host = config['api']['host']
    port = config['api']['port']
    
    print(f"\n🌐 Starting server at http://{host}:{port}")
    print("\n📍 Endpoints:")
    print(f"   GET  /health       - Health check")
    print(f"   GET  /info         - Model information")
    print(f"   POST /predict      - Predict from JSON")
    print(f"   POST /predict_csv  - Predict from CSV file")
    print("\n" + "="*70 + "\n")
    
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run_server()
