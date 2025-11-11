"""
Flask API Server for Natron Model Predictions
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


class NatronAPI:
    """API server for Natron model predictions"""
    
    def __init__(self, model_path: str, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        self.device = torch.device(self.config['training']['device'] if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize feature engine and sequence creator
        self.feature_engine = FeatureEngine()
        self.sequence_creator = SequenceCreator(sequence_length=self.config['data']['sequence_length'])
        
        # Regime names
        self.regime_names = [
            'BULL_STRONG', 'BULL_WEAK', 'RANGE',
            'BEAR_WEAK', 'BEAR_STRONG', 'VOLATILE'
        ]
    
    def _load_model(self, model_path: str) -> NatronTransformer:
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Infer n_features from config or checkpoint
        n_features = self.config['features']['total_features']
        
        model = NatronTransformer(
            n_features=n_features,
            d_model=self.config['model']['d_model'],
            nhead=self.config['model']['nhead'],
            num_layers=self.config['model']['num_layers'],
            dim_feedforward=self.config['model']['dim_feedforward'],
            dropout=self.config['model']['dropout'],
            sequence_length=self.config['data']['sequence_length']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def predict(self, ohlcv_data: pd.DataFrame) -> Dict:
        """
        Predict from OHLCV data
        
        Args:
            ohlcv_data: DataFrame with columns: time, open, high, low, close, volume
                       Must have at least 96 rows
        
        Returns:
            dict with predictions
        """
        # Generate features
        features_df = self.feature_engine.fit_transform(ohlcv_data)
        
        # Create sequence (last 96 candles)
        if len(features_df) < self.config['data']['sequence_length']:
            raise ValueError(f"Need at least {self.config['data']['sequence_length']} candles")
        
        # Get last sequence
        X_scaled = self.sequence_creator.transform_features(features_df)
        X_sequence = X_scaled[-self.config['data']['sequence_length']:]
        X_tensor = torch.FloatTensor(X_sequence).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(X_tensor)
        
        # Format output
        buy_prob = outputs['buy'].item()
        sell_prob = outputs['sell'].item()
        direction_probs = outputs['direction'].cpu().numpy()[0]
        regime_probs = outputs['regime'].cpu().numpy()[0]
        
        direction_idx = np.argmax(direction_probs)
        regime_idx = np.argmax(regime_probs)
        
        # Confidence (average of max probabilities)
        confidence = (buy_prob + sell_prob + direction_probs[direction_idx] + regime_probs[regime_idx]) / 4
        
        return {
            'buy_prob': float(buy_prob),
            'sell_prob': float(sell_prob),
            'direction_up': float(direction_probs[1]),  # Up probability
            'direction_down': float(direction_probs[0]),  # Down probability
            'direction_neutral': float(direction_probs[2]),  # Neutral probability
            'regime': self.regime_names[regime_idx],
            'regime_probs': {self.regime_names[i]: float(regime_probs[i]) for i in range(6)},
            'confidence': float(confidence)
        }


# Flask app
app = Flask(__name__)

# Global API instance
api_instance = None


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        
        # Parse OHLCV data
        ohlcv_df = pd.DataFrame(data['ohlcv'])
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        
        if not all(col in ohlcv_df.columns for col in required_cols):
            return jsonify({'error': f'Missing required columns. Need: {required_cols}'}), 400
        
        # Predict
        result = api_instance.predict(ohlcv_df)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'natron_v2'})


def create_app(model_path: str, config_path: str):
    """Create Flask app with loaded model"""
    global api_instance
    api_instance = NatronAPI(model_path, config_path)
    return app


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model/natron_v2.pt')
    parser.add_argument('--config_path', type=str, default='config/config.yaml')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    
    args = parser.parse_args()
    
    app = create_app(args.model_path, args.config_path)
    app.run(host=args.host, port=args.port, debug=False)
