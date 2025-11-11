"""
Flask API Server for Natron Transformer
Provides /predict endpoint for real-time trading signals
"""

from flask import Flask, request, jsonify
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import time
import logging
from typing import Dict, List
import pickle

from model import NatronTransformer
from feature_engine import FeatureEngine
from label_generator import REGIME_NAMES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)


class NatronPredictor:
    """
    Prediction engine for Natron model.
    """
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to configuration file
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = self.config['training']['device']
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            self.device = 'cpu'
        
        # Initialize feature engine
        self.feature_engine = FeatureEngine()
        
        # Load scaler
        scaler_path = Path('model/scaler.pkl')
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Scaler loaded from {scaler_path}")
        else:
            logger.warning("Scaler not found, features will not be normalized")
            self.scaler = None
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        logger.info(f"✅ Natron Predictor initialized on {self.device}")
    
    def _load_model(self) -> NatronTransformer:
        """Load trained model"""
        model_path = self.config['api']['model_path']
        
        if not Path(model_path).exists():
            # Try to find latest checkpoint
            supervised_dir = Path(self.config['supervised']['checkpoint_dir'])
            if (supervised_dir / 'supervised_best.pt').exists():
                model_path = str(supervised_dir / 'supervised_best.pt')
            else:
                raise FileNotFoundError(f"Model not found at {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        
        # Create model
        model = NatronTransformer(
            n_features=self.config['features']['total_features'],
            d_model=self.config['model']['d_model'],
            nhead=self.config['model']['nhead'],
            num_encoder_layers=self.config['model']['num_encoder_layers'],
            dim_feedforward=self.config['model']['dim_feedforward'],
            dropout=self.config['model']['dropout'],
            max_seq_length=self.config['model']['max_seq_length']
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        logger.info("✅ Model loaded successfully")
        return model
    
    def predict(self, ohlcv_data: List[Dict]) -> Dict:
        """
        Make prediction from OHLCV data.
        
        Args:
            ohlcv_data: List of 96+ OHLCV candles
                       Each dict: {'time': str, 'open': float, 'high': float, 
                                   'low': float, 'close': float, 'volume': float}
        
        Returns:
            Prediction dictionary with buy_prob, sell_prob, direction, regime, confidence
        """
        start_time = time.time()
        
        # Validate input
        if len(ohlcv_data) < 96:
            raise ValueError(f"Need at least 96 candles, got {len(ohlcv_data)}")
        
        # Take last 96 candles
        ohlcv_data = ohlcv_data[-96:]
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data)
        
        # Generate features
        features = self.feature_engine.generate_features(df)
        
        # Take last sequence
        features_last = features.iloc[-96:].values
        
        # Normalize if scaler available
        if self.scaler is not None:
            features_last = self.scaler.transform(features_last)
        
        # Convert to tensor
        x = torch.from_numpy(features_last).float().unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(x)
        
        # Extract predictions
        buy_prob = outputs['buy'].item()
        sell_prob = outputs['sell'].item()
        
        direction_probs = torch.softmax(outputs['direction'], dim=-1)[0]
        direction_up = direction_probs[1].item()
        direction_down = direction_probs[0].item()
        
        regime_probs = torch.softmax(outputs['regime'], dim=-1)[0]
        regime_id = regime_probs.argmax().item()
        regime_name = REGIME_NAMES[regime_id]
        regime_confidence = regime_probs[regime_id].item()
        
        # Calculate overall confidence
        confidence = max(buy_prob, sell_prob, direction_up, direction_down)
        
        # Prepare response
        response = {
            'buy_prob': round(buy_prob, 4),
            'sell_prob': round(sell_prob, 4),
            'direction_up': round(direction_up, 4),
            'direction_down': round(direction_down, 4),
            'regime': regime_name,
            'regime_confidence': round(regime_confidence, 4),
            'confidence': round(confidence, 4),
            'latency_ms': round((time.time() - start_time) * 1000, 2)
        }
        
        return response


# Global predictor instance
predictor = None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    
    Request body:
    {
        "data": [
            {"time": "2024-01-01 00:00:00", "open": 100.0, "high": 101.0, 
             "low": 99.0, "close": 100.5, "volume": 1000},
            ...
        ]
    }
    
    Response:
    {
        "buy_prob": 0.71,
        "sell_prob": 0.24,
        "direction_up": 0.69,
        "direction_down": 0.25,
        "regime": "BULL_WEAK",
        "regime_confidence": 0.82,
        "confidence": 0.82,
        "latency_ms": 45.2
    }
    """
    try:
        # Parse request
        data = request.get_json()
        
        if 'data' not in data:
            return jsonify({'error': 'Missing "data" field'}), 400
        
        ohlcv_data = data['data']
        
        # Make prediction
        result = predictor.predict(ohlcv_data)
        
        # Log prediction if enabled
        if predictor.config['api']['log_predictions']:
            logger.info(f"Prediction: Buy={result['buy_prob']:.2f}, "
                       f"Sell={result['sell_prob']:.2f}, "
                       f"Regime={result['regime']}, "
                       f"Latency={result['latency_ms']:.1f}ms")
        
        return jsonify(result)
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/info', methods=['GET'])
def info():
    """Model information endpoint"""
    return jsonify({
        'model': 'Natron Transformer',
        'version': '2.0.0',
        'sequence_length': 96,
        'features': predictor.config['features']['total_features'],
        'device': predictor.device,
        'tasks': ['buy', 'sell', 'direction', 'regime']
    })


def run_server(config_path: str = 'config/config.yaml'):
    """
    Run the API server.
    
    Args:
        config_path: Path to configuration file
    """
    global predictor
    
    # Initialize predictor
    logger.info("Initializing Natron Predictor...")
    predictor = NatronPredictor(config_path)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    host = config['api']['host']
    port = config['api']['port']
    
    logger.info(f"🚀 Starting API server on {host}:{port}")
    
    # Run Flask app
    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'config/config.yaml'
    
    run_server(config_path)
