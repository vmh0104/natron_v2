"""
MQL5 Socket Bridge: Real-time communication between MetaTrader 5 and Natron AI
Python socket server that receives OHLCV data from MQL5 EA and returns predictions
"""

import socket
import json
import threading
import pandas as pd
import numpy as np
from typing import Dict, Optional
import yaml
import os

from api_server import NatronPredictor


class MQL5SocketBridge:
    """Socket server for MQL5 integration"""
    
    def __init__(self, 
                 model_path: str,
                 config_path: str = "config.yaml",
                 host: str = "localhost",
                 port: int = 8888):
        """
        Args:
            model_path: Path to trained Natron model
            config_path: Path to config file
            host: Host to bind socket server
            port: Port to bind socket server
        """
        self.host = host
        self.port = port
        self.predictor = NatronPredictor(model_path, config_path)
        self.socket = None
        self.running = False
    
    def start_server(self):
        """Start socket server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        self.running = True
        
        print(f"🚀 MQL5 Socket Bridge started on {self.host}:{self.port}")
        print("Waiting for MQL5 EA connections...")
        
        while self.running:
            try:
                client_socket, address = self.socket.accept()
                print(f"📡 Client connected: {address}")
                
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
            
            except Exception as e:
                if self.running:
                    print(f"Error accepting connection: {e}")
    
    def handle_client(self, client_socket: socket.socket, address: tuple):
        """Handle client connection"""
        try:
            while self.running:
                # Receive data
                data = client_socket.recv(8192)
                if not data:
                    break
                
                # Parse JSON
                try:
                    message = json.loads(data.decode('utf-8'))
                except json.JSONDecodeError:
                    response = {"error": "Invalid JSON"}
                    client_socket.send(json.dumps(response).encode('utf-8'))
                    continue
                
                # Process request
                if message.get('type') == 'predict':
                    result = self.process_prediction_request(message)
                    response = json.dumps(result)
                    client_socket.send(response.encode('utf-8'))
                
                elif message.get('type') == 'ping':
                    response = json.dumps({"status": "pong"})
                    client_socket.send(response.encode('utf-8'))
                
                else:
                    response = json.dumps({"error": "Unknown request type"})
                    client_socket.send(response.encode('utf-8'))
        
        except Exception as e:
            print(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
            print(f"Client disconnected: {address}")
    
    def process_prediction_request(self, message: Dict) -> Dict:
        """Process prediction request from MQL5"""
        try:
            # Extract candles data
            candles = message.get('candles', [])
            
            if not candles:
                return {"error": "No candles data provided"}
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            
            # Validate columns
            required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return {"error": f"Missing columns: {missing_cols}"}
            
            # Predict
            result = self.predictor.predict(df)
            
            return {
                "status": "success",
                "prediction": result
            }
        
        except Exception as e:
            return {"error": str(e)}
    
    def stop_server(self):
        """Stop socket server"""
        self.running = False
        if self.socket:
            self.socket.close()
        print("Socket server stopped")


def main():
    """Run MQL5 socket bridge"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MQL5 Socket Bridge for Natron AI')
    parser.add_argument('--model', type=str, default='models/natron_v2.pt',
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host to bind socket server')
    parser.add_argument('--port', type=int, default=8888,
                       help='Port to bind socket server')
    
    args = parser.parse_args()
    
    # Create and start bridge
    bridge = MQL5SocketBridge(
        model_path=args.model,
        config_path=args.config,
        host=args.host,
        port=args.port
    )
    
    try:
        bridge.start_server()
    except KeyboardInterrupt:
        print("\nShutting down...")
        bridge.stop_server()


if __name__ == "__main__":
    main()
