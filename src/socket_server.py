"""
MQL5 Socket Server: Real-time bridge between MetaTrader 5 and Natron AI
"""
import socket
import json
import threading
import pandas as pd
import numpy as np
import torch
import yaml
from typing import Optional

from src.model import NatronTransformer
from src.feature_engine import FeatureEngine
from src.sequence_creator import SequenceCreator


class NatronSocketServer:
    """Socket server for MQL5 integration"""
    
    def __init__(self, model_path: str, config_path: str, host: str = '127.0.0.1', port: int = 8888):
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
        
        # Socket settings
        self.host = host
        self.port = port
        self.socket = None
        
        # Buffer for OHLCV data
        self.ohlcv_buffer = []
    
    def _load_model(self, model_path: str) -> NatronTransformer:
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
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
    
    def predict_from_ohlcv(self, ohlcv_data: list) -> dict:
        """
        Predict from OHLCV data list
        
        Args:
            ohlcv_data: List of dicts with keys: time, open, high, low, close, volume
        
        Returns:
            dict with predictions
        """
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data)
        
        # Ensure we have enough data
        if len(df) < self.config['data']['sequence_length']:
            return {'error': f'Need at least {self.config['data']['sequence_length']} candles'}
        
        # Generate features
        features_df = self.feature_engine.fit_transform(df)
        
        # Create sequence
        feature_cols = [c for c in features_df.columns if c != 'time']
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
        
        confidence = (buy_prob + sell_prob + direction_probs[direction_idx] + regime_probs[regime_idx]) / 4
        
        return {
            'buy_prob': float(buy_prob),
            'sell_prob': float(sell_prob),
            'direction': int(direction_idx),
            'direction_up': float(direction_probs[1]),
            'regime': self.regime_names[regime_idx],
            'regime_id': int(regime_idx),
            'confidence': float(confidence),
            'status': 'success'
        }
    
    def handle_client(self, client_socket, address):
        """Handle client connection"""
        print(f"📡 Client connected: {address}")
        
        try:
            while True:
                # Receive data
                data = client_socket.recv(4096)
                if not data:
                    break
                
                try:
                    # Parse JSON
                    message = json.loads(data.decode('utf-8'))
                    
                    if message.get('action') == 'predict':
                        # Get OHLCV data
                        ohlcv_data = message.get('ohlcv', [])
                        
                        if len(ohlcv_data) == 0:
                            response = {'error': 'No OHLCV data provided', 'status': 'error'}
                        else:
                            # Predict
                            response = self.predict_from_ohlcv(ohlcv_data)
                        
                        # Send response
                        response_json = json.dumps(response)
                        client_socket.send(response_json.encode('utf-8'))
                    
                    elif message.get('action') == 'ping':
                        # Health check
                        response = {'status': 'pong', 'model': 'natron_v2'}
                        client_socket.send(json.dumps(response).encode('utf-8'))
                    
                    else:
                        response = {'error': 'Unknown action', 'status': 'error'}
                        client_socket.send(json.dumps(response).encode('utf-8'))
                
                except json.JSONDecodeError as e:
                    response = {'error': f'Invalid JSON: {str(e)}', 'status': 'error'}
                    client_socket.send(json.dumps(response).encode('utf-8'))
                
                except Exception as e:
                    response = {'error': str(e), 'status': 'error'}
                    client_socket.send(json.dumps(response).encode('utf-8'))
        
        except Exception as e:
            print(f"❌ Error handling client {address}: {e}")
        
        finally:
            client_socket.close()
            print(f"🔌 Client disconnected: {address}")
    
    def start(self):
        """Start socket server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        
        print(f"🚀 Natron Socket Server started on {self.host}:{self.port}")
        print("   Waiting for MQL5 EA connections...")
        
        try:
            while True:
                client_socket, address = self.socket.accept()
                # Handle each client in a separate thread
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
        
        except KeyboardInterrupt:
            print("\n⏹️  Shutting down server...")
        
        finally:
            if self.socket:
                self.socket.close()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron Socket Server for MQL5')
    parser.add_argument('--model_path', type=str, default='model/natron_v2.pt')
    parser.add_argument('--config_path', type=str, default='config/config.yaml')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8888)
    
    args = parser.parse_args()
    
    server = NatronSocketServer(
        model_path=args.model_path,
        config_path=args.config_path,
        host=args.host,
        port=args.port
    )
    
    server.start()


if __name__ == '__main__':
    main()
