"""
MQL5 Socket Bridge Server - Real-time Trading Integration
"""

import os
import socket
import json
import threading
import numpy as np
import pandas as pd
import torch
import yaml
import pickle
from src.model import create_model
from src.feature_engine import FeatureEngine


class MQL5BridgeServer:
    """Socket server for MQL5 integration"""
    
    def __init__(self, model_path: str, config_path: str = "config.yaml", 
                 host: str = "127.0.0.1", port: int = 8888):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.config_path = config_path
        
        # Load model and config
        self._load_model()
        
        # Buffer for candles
        self.candle_buffer = []
        self.buffer_lock = threading.Lock()
        
    def _load_model(self):
        """Load model and scaler"""
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['training']['device'] if torch.cuda.is_available() else 'cpu')
        
        # Load feature scaler
        scaler_path = self.model_path.replace('.pt', '_scaler.pkl')
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(os.path.dirname(self.model_path), 'feature_scaler.pkl')
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.feature_scaler = pickle.load(f)
        else:
            self.feature_scaler = None
        
        # Create model
        model_config = {
            'num_features': self.config['features']['num_features'],
            'd_model': self.config['model']['d_model'],
            'nhead': self.config['model']['nhead'],
            'num_layers': self.config['model']['num_layers'],
            'dim_feedforward': self.config['model']['dim_feedforward'],
            'dropout': self.config['model']['dropout'],
            'max_seq_length': self.config['model']['max_seq_length'],
            'activation': self.config['model']['activation']
        }
        
        self.model = create_model(model_config)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(self.device)
        
        self.feature_engine = FeatureEngine()
        
        print(f"✅ Model loaded: {self.model_path}")
    
    def _process_candles(self, candles: list) -> dict:
        """Process candles and return predictions"""
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        
        # Extract features
        features_df = self.feature_engine.extract_all_features(df)
        
        # Take last N candles
        sequence_length = self.config['data']['sequence_length']
        features_sequence = features_df.iloc[-sequence_length:].values
        
        # Normalize
        if self.feature_scaler is not None:
            features_sequence = self.feature_scaler.transform(features_sequence)
        else:
            features_sequence = (features_sequence - features_sequence.mean(axis=0)) / (features_sequence.std(axis=0) + 1e-8)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_sequence).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(features_tensor)
        
        # Extract predictions
        buy_prob = outputs['buy_prob'].item()
        sell_prob = outputs['sell_prob'].item()
        
        direction_logits = outputs['direction_logits']
        direction_probs = torch.exp(direction_logits).cpu().numpy()[0]
        direction_idx = np.argmax(direction_probs)
        direction_map = {0: "Down", 1: "Up", 2: "Neutral"}
        direction = direction_map[direction_idx]
        
        regime_logits = outputs['regime_logits']
        regime_probs = torch.exp(regime_logits).cpu().numpy()[0]
        regime_idx = np.argmax(regime_probs)
        regime_names = {
            0: "BULL_STRONG", 1: "BULL_WEAK", 2: "RANGE",
            3: "BEAR_WEAK", 4: "BEAR_STRONG", 5: "VOLATILE"
        }
        regime = regime_names[regime_idx]
        
        return {
            "buy_prob": float(buy_prob),
            "sell_prob": float(sell_prob),
            "direction": direction,
            "regime": regime,
            "confidence": float(max(buy_prob, sell_prob, direction_probs.max(), regime_probs.max()))
        }
    
    def _handle_client(self, client_socket, address):
        """Handle client connection"""
        print(f"📡 Client connected: {address}")
        
        try:
            while True:
                # Receive data
                data = client_socket.recv(4096)
                if not data:
                    break
                
                try:
                    message = json.loads(data.decode('utf-8'))
                    msg_type = message.get('type', '')
                    
                    if msg_type == 'candles':
                        # Update candle buffer
                        candles = message.get('candles', [])
                        with self.buffer_lock:
                            self.candle_buffer.extend(candles)
                            # Keep only last N*2 candles
                            max_buffer = self.config['data']['sequence_length'] * 2
                            if len(self.candle_buffer) > max_buffer:
                                self.candle_buffer = self.candle_buffer[-max_buffer:]
                        
                        # Process if we have enough candles
                        if len(self.candle_buffer) >= self.config['data']['sequence_length']:
                            predictions = self._process_candles(self.candle_buffer)
                            response = {
                                "status": "ok",
                                "predictions": predictions
                            }
                        else:
                            response = {
                                "status": "buffering",
                                "candles_received": len(self.candle_buffer),
                                "candles_needed": self.config['data']['sequence_length']
                            }
                    
                    elif msg_type == 'predict':
                        # Direct prediction request
                        candles = message.get('candles', [])
                        if len(candles) < self.config['data']['sequence_length']:
                            response = {
                                "status": "error",
                                "message": f"Need at least {self.config['data']['sequence_length']} candles"
                            }
                        else:
                            predictions = self._process_candles(candles)
                            response = {
                                "status": "ok",
                                "predictions": predictions
                            }
                    
                    else:
                        response = {"status": "error", "message": f"Unknown message type: {msg_type}"}
                    
                    # Send response
                    response_json = json.dumps(response)
                    client_socket.send(response_json.encode('utf-8'))
                
                except json.JSONDecodeError as e:
                    error_response = {"status": "error", "message": f"Invalid JSON: {str(e)}"}
                    client_socket.send(json.dumps(error_response).encode('utf-8'))
        
        except Exception as e:
            print(f"❌ Error handling client {address}: {str(e)}")
        
        finally:
            client_socket.close()
            print(f"🔌 Client disconnected: {address}")
    
    def start(self):
        """Start socket server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        
        print(f"\n🚀 MQL5 Bridge Server started on {self.host}:{self.port}")
        print("   Waiting for connections...")
        
        try:
            while True:
                client_socket, address = server_socket.accept()
                # Handle each client in a separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
        
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
            server_socket.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MQL5 Socket Bridge Server")
    parser.add_argument("--model-path", type=str, default="model/natron_v2.pt", help="Path to model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host")
    parser.add_argument("--port", type=int, default=8888, help="Port")
    
    args = parser.parse_args()
    
    server = MQL5BridgeServer(args.model_path, args.config, args.host, args.port)
    server.start()
