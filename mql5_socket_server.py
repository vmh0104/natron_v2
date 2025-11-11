"""
MQL5 Socket Server Integration
Real-time bridge between MetaTrader 5 and Natron AI Model
"""
import socket
import json
import threading
import yaml
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from model import NatronTransformer
from feature_engine import FeatureEngine


class MQL5SocketServer:
    """
    Socket server for MQL5 EA integration
    Receives real-time OHLCV data and returns AI predictions
    Target latency: <50ms
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "config.yaml",
        host: str = "localhost",
        port: int = 8888,
        timeout: float = 5.0
    ):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model(model_path)
        self.feature_engine = FeatureEngine()
        
        # Socket configuration
        self.host = host
        self.port = port
        self.timeout = timeout
        
        # Buffer for recent candles
        self.candle_buffer: List[Dict] = []
        self.max_buffer_size = self.config['data']['sequence_length'] * 2
        
        # Regime mapping
        self.regime_map = {
            0: "BULL_STRONG",
            1: "BULL_WEAK",
            2: "RANGE",
            3: "BEAR_WEAK",
            4: "BEAR_STRONG",
            5: "VOLATILE"
        }
        
        self.running = False
        self.server_socket = None
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model(self, model_path: str) -> NatronTransformer:
        """Load trained model"""
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
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        
        return model
    
    def add_candle(self, candle: Dict):
        """Add new candle to buffer"""
        self.candle_buffer.append(candle)
        if len(self.candle_buffer) > self.max_buffer_size:
            self.candle_buffer.pop(0)
    
    def predict(self, candles: Optional[List[Dict]] = None) -> Dict:
        """
        Make prediction from candles
        
        Args:
            candles: Optional list of candles. If None, uses buffer
            
        Returns:
            Prediction dictionary
        """
        if candles is None:
            candles = self.candle_buffer
        
        if len(candles) < self.config['data']['sequence_length']:
            return {
                "error": f"Insufficient candles. Need {self.config['data']['sequence_length']}, got {len(candles)}"
            }
        
        # Use last N candles
        df = pd.DataFrame(candles[-self.config['data']['sequence_length']:])
        
        # Generate features
        features_df = self.feature_engine.generate_all_features(df)
        features = features_df.values.astype(np.float32)
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # Predict
        sequence = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(sequence)
        
        # Extract results
        buy_prob = predictions['buy'].item()
        sell_prob = predictions['sell'].item()
        direction_probs = torch.softmax(predictions['direction'], dim=1).cpu().numpy()[0]
        regime_probs = torch.softmax(predictions['regime'], dim=1).cpu().numpy()[0]
        
        regime_id = np.argmax(regime_probs)
        confidence = (buy_prob + sell_prob + direction_probs.max() + regime_probs.max()) / 4
        
        return {
            "buy_prob": float(buy_prob),
            "sell_prob": float(sell_prob),
            "direction_up": float(direction_probs[1]),
            "direction_down": float(direction_probs[0]),
            "direction_neutral": float(direction_probs[2]),
            "regime": self.regime_map[regime_id],
            "regime_id": int(regime_id),
            "confidence": float(confidence),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    def handle_client(self, client_socket: socket.socket, address: tuple):
        """Handle client connection"""
        print(f"📡 Client connected: {address}")
        
        try:
            while self.running:
                # Receive data
                data = client_socket.recv(8192).decode('utf-8')
                
                if not data:
                    break
                
                try:
                    # Parse JSON message
                    message = json.loads(data)
                    
                    # Handle different message types
                    msg_type = message.get('type', 'predict')
                    
                    if msg_type == 'predict':
                        # Get candles from message or use buffer
                        candles = message.get('candles', None)
                        
                        if candles:
                            # Update buffer
                            for candle in candles:
                                self.add_candle(candle)
                        
                        # Make prediction
                        result = self.predict(candles)
                        
                        # Send response
                        response = json.dumps(result)
                        client_socket.sendall(response.encode('utf-8'))
                    
                    elif msg_type == 'add_candle':
                        # Add single candle to buffer
                        candle = message.get('candle')
                        if candle:
                            self.add_candle(candle)
                            client_socket.sendall(json.dumps({"status": "ok"}).encode('utf-8'))
                    
                    elif msg_type == 'predict_buffer':
                        # Predict using buffer
                        result = self.predict()
                        response = json.dumps(result)
                        client_socket.sendall(response.encode('utf-8'))
                    
                    else:
                        error_response = json.dumps({"error": f"Unknown message type: {msg_type}"})
                        client_socket.sendall(error_response.encode('utf-8'))
                
                except json.JSONDecodeError as e:
                    error_response = json.dumps({"error": f"Invalid JSON: {str(e)}"})
                    client_socket.sendall(error_response.encode('utf-8'))
                
                except Exception as e:
                    error_response = json.dumps({"error": str(e)})
                    client_socket.sendall(error_response.encode('utf-8'))
        
        except Exception as e:
            print(f"❌ Error handling client {address}: {e}")
        
        finally:
            client_socket.close()
            print(f"🔌 Client disconnected: {address}")
    
    def start(self):
        """Start socket server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(self.timeout)
        
        self.running = True
        
        print(f"\n🚀 MQL5 Socket Server Started")
        print(f"   Host: {self.host}")
        print(f"   Port: {self.port}")
        print(f"   Model: Loaded on {self.device}")
        print(f"   Waiting for connections...")
        print("="*60)
        
        try:
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"❌ Server error: {e}")
        
        except KeyboardInterrupt:
            print("\n⏹ Shutting down server...")
            self.stop()
    
    def stop(self):
        """Stop socket server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        print("✅ Server stopped")


def main():
    """Main function"""
    import os
    
    config_path = os.getenv('CONFIG_PATH', 'config.yaml')
    model_path = os.getenv('MODEL_PATH', 'models/natron_v2.pt')
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print("   Please train the model first using train.py")
        return
    
    # Load socket config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    socket_config = config.get('socket', {})
    
    server = MQL5SocketServer(
        model_path=model_path,
        config_path=config_path,
        host=socket_config.get('host', 'localhost'),
        port=socket_config.get('port', 8888),
        timeout=socket_config.get('timeout', 5.0)
    )
    
    server.start()


if __name__ == "__main__":
    main()
