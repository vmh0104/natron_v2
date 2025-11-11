"""
Natron Transformer - API Server
Flask API for real-time inference and MT5 integration
"""

import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import time
from typing import Dict, List

from model import create_model
from feature_engine import FeatureEngine
from dataset import load_scaler


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
MODEL = None
SCALER = None
FEATURE_ENGINE = None
CONFIG = None
DEVICE = None

REGIME_NAMES = {
    0: 'BULL_STRONG',
    1: 'BULL_WEAK',
    2: 'RANGE',
    3: 'BEAR_WEAK',
    4: 'BEAR_STRONG',
    5: 'VOLATILE'
}


def initialize_model(model_path: str, scaler_path: str, config):
    """Initialize model and components"""
    global MODEL, SCALER, FEATURE_ENGINE, CONFIG, DEVICE
    
    logger.info("🚀 Initializing Natron API Server...")
    
    CONFIG = config
    DEVICE = config.device
    
    # Load model
    logger.info(f"📂 Loading model from {model_path}")
    MODEL = create_model(config)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    MODEL.load_state_dict(checkpoint['model_state_dict'])
    MODEL.to(DEVICE)
    MODEL.eval()
    logger.info("✅ Model loaded")
    
    # Load scaler
    logger.info(f"📂 Loading scaler from {scaler_path}")
    SCALER = load_scaler(scaler_path)
    logger.info("✅ Scaler loaded")
    
    # Initialize feature engine
    FEATURE_ENGINE = FeatureEngine()
    logger.info("✅ Feature engine initialized")
    
    logger.info(f"🔥 Server ready on device: {DEVICE}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'device': str(DEVICE),
        'timestamp': time.time()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    
    Expected input:
    {
        "candles": [
            {"time": "2023-01-01 00:00:00", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 1000},
            ...  # 96 candles total
        ]
    }
    
    Returns:
    {
        "buy_prob": 0.71,
        "sell_prob": 0.24,
        "direction": "UP",
        "direction_probs": {"UP": 0.69, "DOWN": 0.21, "NEUTRAL": 0.10},
        "regime": "BULL_WEAK",
        "regime_confidence": 0.82,
        "confidence": 0.82,
        "processing_time_ms": 15.3
    }
    """
    start_time = time.time()
    
    try:
        # Parse input
        data = request.get_json()
        
        if 'candles' not in data:
            return jsonify({'error': 'Missing "candles" in request'}), 400
        
        candles = data['candles']
        
        if len(candles) != CONFIG.data.sequence_length:
            return jsonify({
                'error': f'Expected {CONFIG.data.sequence_length} candles, got {len(candles)}'
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        
        if not all(col in df.columns for col in required_cols):
            return jsonify({
                'error': f'Missing required columns. Need: {required_cols}'
            }), 400
        
        # Extract features
        features = FEATURE_ENGINE.compute_all_features(df)
        
        # Normalize
        features_normalized = SCALER.transform(features.values)
        
        # Convert to tensor
        input_tensor = torch.from_numpy(features_normalized).float()
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)  # (1, 96, n_features)
        
        # Inference
        with torch.no_grad():
            outputs = MODEL(input_tensor)
        
        # Parse outputs
        buy_probs = torch.softmax(outputs['buy'], dim=1)[0].cpu().numpy()
        sell_probs = torch.softmax(outputs['sell'], dim=1)[0].cpu().numpy()
        direction_probs = torch.softmax(outputs['direction'], dim=1)[0].cpu().numpy()
        regime_probs = torch.softmax(outputs['regime'], dim=1)[0].cpu().numpy()
        
        # Get predictions
        direction_pred = int(direction_probs.argmax())
        regime_pred = int(regime_probs.argmax())
        
        direction_map = {0: 'DOWN', 1: 'UP', 2: 'NEUTRAL'}
        
        # Calculate confidence (max probability)
        confidence = max(
            buy_probs[1],
            sell_probs[1],
            direction_probs.max(),
            regime_probs.max()
        )
        
        # Prepare response
        response = {
            'buy_prob': float(buy_probs[1]),
            'sell_prob': float(sell_probs[1]),
            'direction': direction_map[direction_pred],
            'direction_probs': {
                'UP': float(direction_probs[1]),
                'DOWN': float(direction_probs[0]),
                'NEUTRAL': float(direction_probs[2])
            },
            'regime': REGIME_NAMES[regime_pred],
            'regime_confidence': float(regime_probs[regime_pred]),
            'regime_probs': {
                REGIME_NAMES[i]: float(regime_probs[i])
                for i in range(len(regime_probs))
            },
            'confidence': float(confidence),
            'processing_time_ms': (time.time() - start_time) * 1000
        }
        
        # Add signal recommendation
        if buy_probs[1] > CONFIG.inference.confidence_threshold and buy_probs[1] > sell_probs[1]:
            response['signal'] = 'BUY'
        elif sell_probs[1] > CONFIG.inference.confidence_threshold and sell_probs[1] > buy_probs[1]:
            response['signal'] = 'SELL'
        else:
            response['signal'] = 'HOLD'
        
        logger.info(f"✅ Prediction: {response['signal']} (confidence: {confidence:.3f}, time: {response['processing_time_ms']:.1f}ms)")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint
    
    Expected input:
    {
        "sequences": [
            [{"time": ..., "open": ..., ...}, ...],  # 96 candles
            [{"time": ..., "open": ..., ...}, ...],  # 96 candles
            ...
        ]
    }
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        sequences = data['sequences']
        
        results = []
        
        for seq in sequences:
            # Process each sequence
            df = pd.DataFrame(seq)
            features = FEATURE_ENGINE.compute_all_features(df)
            features_normalized = SCALER.transform(features.values)
            input_tensor = torch.from_numpy(features_normalized).float().unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = MODEL(input_tensor)
            
            buy_probs = torch.softmax(outputs['buy'], dim=1)[0].cpu().numpy()
            sell_probs = torch.softmax(outputs['sell'], dim=1)[0].cpu().numpy()
            direction_probs = torch.softmax(outputs['direction'], dim=1)[0].cpu().numpy()
            regime_probs = torch.softmax(outputs['regime'], dim=1)[0].cpu().numpy()
            
            direction_pred = int(direction_probs.argmax())
            regime_pred = int(regime_probs.argmax())
            direction_map = {0: 'DOWN', 1: 'UP', 2: 'NEUTRAL'}
            
            results.append({
                'buy_prob': float(buy_probs[1]),
                'sell_prob': float(sell_probs[1]),
                'direction': direction_map[direction_pred],
                'regime': REGIME_NAMES[regime_pred]
            })
        
        total_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'results': results,
            'count': len(results),
            'total_time_ms': total_time,
            'avg_time_ms': total_time / len(results)
        })
    
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        from model import count_parameters
        
        return jsonify({
            'model_name': 'Natron Transformer v2',
            'parameters': count_parameters(MODEL),
            'device': str(DEVICE),
            'sequence_length': CONFIG.data.sequence_length,
            'n_features': CONFIG.model.input_dim,
            'd_model': CONFIG.model.d_model,
            'n_layers': CONFIG.model.num_encoder_layers,
            'n_heads': CONFIG.model.nhead,
            'tasks': ['buy', 'sell', 'direction', 'regime']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def create_app(config, model_path: str = None, scaler_path: str = None):
    """Create and configure Flask app"""
    
    # Use defaults if not provided
    if model_path is None:
        model_path = config.inference.model_path
    if scaler_path is None:
        scaler_path = os.path.join(config.output_dir, 'scaler.pkl')
    
    # Initialize model
    initialize_model(model_path, scaler_path, config)
    
    return app


def run_server(config, model_path: str = None, scaler_path: str = None):
    """Run Flask server"""
    app = create_app(config, model_path, scaler_path)
    
    logger.info(f"\n🌐 Starting Natron API Server")
    logger.info(f"   Host: {config.inference.api_host}")
    logger.info(f"   Port: {config.inference.api_port}")
    logger.info(f"   Device: {config.device}")
    logger.info(f"\n📡 Available endpoints:")
    logger.info(f"   GET  /health          - Health check")
    logger.info(f"   POST /predict         - Single prediction")
    logger.info(f"   POST /batch_predict   - Batch prediction")
    logger.info(f"   GET  /model_info      - Model information")
    
    app.run(
        host=config.inference.api_host,
        port=config.inference.api_port,
        debug=config.inference.api_debug,
        threaded=True
    )


if __name__ == "__main__":
    from config import load_config
    import sys
    
    config = load_config()
    
    # Override paths if provided
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    scaler_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    run_server(config, model_path, scaler_path)
