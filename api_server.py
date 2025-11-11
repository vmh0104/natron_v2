"""
Flask API server for Natron Transformer predictions
Endpoint: /predict - receives last 96 OHLCV candles, returns trading signals
"""

import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import yaml
import os
from typing import Dict, List

from src.model import NatronTransformer
from src.feature_engine import FeatureEngine
from src.sequence_creator import SequenceCreator


class NatronPredictor:
    """Wrapper for Natron model inference"""
    
    def __init__(self, model_path: str, config_path: str = "config.yaml"):
        """Initialize predictor with trained model"""
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        self.device = torch.device(self.config['training']['device'] if torch.cuda.is_available() else 'cpu')
        
        # Determine input dimension (will be set after first prediction)
        self.input_dim = None
        self.model = None
        
        # Load model weights
        if os.path.exists(model_path):
            self._load_model(model_path)
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Initialize feature engine
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
    
    def _load_model(self, model_path: str):
        """Load model from checkpoint"""
        # Try to infer input dimension from config or use default
        # We'll determine it dynamically from first prediction
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle both direct state_dict and checkpoint dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'input_projection.weight' in checkpoint:
            state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Try to get input_dim from model state dict
        if 'input_projection.weight' in state_dict:
            self.input_dim = state_dict['input_projection.weight'].shape[1]
        else:
            # Default to num_features from config
            self.input_dim = self.config['features']['num_features']
        
        self.model = NatronTransformer(
            input_dim=self.input_dim,
            d_model=self.config['model']['d_model'],
            nhead=self.config['model']['nhead'],
            num_layers=self.config['model']['num_layers'],
            dim_feedforward=self.config['model']['dim_feedforward'],
            dropout=self.config['model']['dropout'],
            activation=self.config['model']['activation'],
            max_seq_length=self.config['model']['max_seq_length']
        )
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, ohlcv_data: pd.DataFrame) -> Dict:
        """
        Predict trading signals from OHLCV data
        
        Args:
            ohlcv_data: DataFrame with columns [time, open, high, low, close, volume]
                       Must have at least 96 rows
            
        Returns:
            Dictionary with predictions
        """
        # Ensure we have enough data
        if len(ohlcv_data) < self.config['data']['sequence_length']:
            raise ValueError(f"Need at least {self.config['data']['sequence_length']} candles")
        
        # Use last 96 candles
        recent_data = ohlcv_data.tail(self.config['data']['sequence_length']).copy()
        
        # Generate features
        features_df = self.feature_engine.generate_all_features(recent_data)
        
        # Extract feature columns (exclude 'time')
        feature_cols = [col for col in features_df.columns if col != 'time']
        features = features_df[feature_cols].values
        
        # Normalize (same as training)
        mean = np.nanmean(features, axis=0, keepdims=True)
        std = np.nanstd(features, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        features = (features - mean) / std
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create sequence
        sequence = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # (1, seq_len, features)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(sequence)
        
        # Extract predictions
        buy_prob = predictions['buy_prob'].item()
        sell_prob = predictions['sell_prob'].item()
        direction_probs = predictions['direction'][0].cpu().numpy()
        regime_probs = predictions['regime'][0].cpu().numpy()
        
        # Get direction (Up/Down/Neutral)
        direction_idx = np.argmax(direction_probs)
        direction_map = {0: "Down", 1: "Up", 2: "Neutral"}
        direction = direction_map[direction_idx]
        direction_up_prob = direction_probs[1]
        
        # Get regime
        regime_idx = np.argmax(regime_probs)
        regime = self.regime_map[regime_idx]
        
        # Calculate confidence (average of max probabilities)
        confidence = (buy_prob + sell_prob + direction_probs.max() + regime_probs.max()) / 4.0
        
        return {
            "buy_prob": float(buy_prob),
            "sell_prob": float(sell_prob),
            "direction_up": float(direction_up_prob),
            "direction": direction,
            "regime": regime,
            "confidence": float(confidence)
        }


# Initialize Flask app
app = Flask(__name__)

# Global predictor (initialized on startup)
predictor = None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": predictor is not None})


@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.json
        
        # Validate input
        if 'candles' not in data:
            return jsonify({"error": "Missing 'candles' field"}), 400
        
        candles = data['candles']
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        
        # Validate columns
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing columns: {missing_cols}"}), 400
        
        # Predict
        result = predictor.predict(df)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    """Run Flask server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron Transformer API Server')
    parser.add_argument('--model', type=str, default='models/natron_v2.pt',
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind to')
    
    args = parser.parse_args()
    
    # Initialize predictor
    global predictor
    print("Loading model...")
    predictor = NatronPredictor(args.model, args.config)
    print("Model loaded successfully!")
    
    # Run server
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
