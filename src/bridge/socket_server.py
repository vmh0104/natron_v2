"""
Socket Server for MQL5 Bridge
Receives OHLCV data from MetaTrader 5 and returns AI predictions
"""

import socket
import json
import threading
import logging
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api_server import NatronPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MQL5SocketServer:
    """
    Socket server for MQL5 communication.
    Handles multiple concurrent connections from MetaTrader 5 terminals.
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 9999, config_path: str = 'config/config.yaml'):
        """
        Args:
            host: Server host address
            port: Server port
            config_path: Path to configuration file
        """
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        
        # Initialize AI predictor
        logger.info("Initializing Natron AI Predictor...")
        self.predictor = NatronPredictor(config_path)
        logger.info("✅ Predictor ready")
        
        # Statistics
        self.total_requests = 0
        self.total_predictions = 0
        self.start_time = time.time()
        
    def start(self):
        """Start the socket server"""
        try:
            # Create socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            self.running = True
            
            logger.info("="*60)
            logger.info("🚀 MQL5 Socket Server Started")
            logger.info("="*60)
            logger.info(f"   Host: {self.host}")
            logger.info(f"   Port: {self.port}")
            logger.info(f"   Waiting for MetaTrader 5 connections...")
            logger.info("="*60)
            
            # Accept connections
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    logger.info(f"📡 New connection from {address}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except Exception as e:
                    if self.running:
                        logger.error(f"Error accepting connection: {e}")
                    
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            self.stop()
    
    def handle_client(self, client_socket: socket.socket, address: tuple):
        """
        Handle a client connection.
        
        Args:
            client_socket: Client socket
            address: Client address
        """
        try:
            while self.running:
                # Receive data
                data = client_socket.recv(65536)  # 64KB buffer
                
                if not data:
                    break
                
                self.total_requests += 1
                
                # Decode request
                try:
                    request_str = data.decode('utf-8')
                    request = json.loads(request_str)
                    
                    # Process request
                    response = self.process_request(request)
                    
                    # Send response
                    response_str = json.dumps(response)
                    client_socket.sendall(response_str.encode('utf-8'))
                    
                    self.total_predictions += 1
                    
                    # Log request
                    if request.get('action') == 'predict':
                        logger.info(f"✅ Prediction sent to {address[0]}: "
                                  f"Buy={response.get('buy_prob', 0):.2f}, "
                                  f"Sell={response.get('sell_prob', 0):.2f}, "
                                  f"Regime={response.get('regime', 'N/A')}, "
                                  f"Latency={response.get('latency_ms', 0):.1f}ms")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from {address}: {e}")
                    error_response = json.dumps({'error': 'Invalid JSON'})
                    client_socket.sendall(error_response.encode('utf-8'))
                
                except Exception as e:
                    logger.error(f"Error processing request from {address}: {e}", exc_info=True)
                    error_response = json.dumps({'error': str(e)})
                    client_socket.sendall(error_response.encode('utf-8'))
        
        except Exception as e:
            logger.error(f"Client handler error for {address}: {e}")
        
        finally:
            client_socket.close()
            logger.info(f"📡 Connection closed: {address}")
    
    def process_request(self, request: dict) -> dict:
        """
        Process a request from MQL5.
        
        Args:
            request: Request dictionary with 'action' and 'data'
            
        Returns:
            Response dictionary
        """
        action = request.get('action', '')
        
        if action == 'predict':
            # Get OHLCV data
            ohlcv_data = request.get('data', [])
            
            if not ohlcv_data:
                return {'error': 'No data provided'}
            
            # Make prediction
            try:
                prediction = self.predictor.predict(ohlcv_data)
                return prediction
            except Exception as e:
                logger.error(f"Prediction error: {e}", exc_info=True)
                return {'error': f'Prediction failed: {str(e)}'}
        
        elif action == 'health':
            # Health check
            uptime = time.time() - self.start_time
            return {
                'status': 'healthy',
                'uptime_seconds': int(uptime),
                'total_requests': self.total_requests,
                'total_predictions': self.total_predictions
            }
        
        elif action == 'info':
            # Server info
            return {
                'server': 'Natron MQL5 Bridge',
                'version': '2.0.0',
                'model': 'Natron Transformer',
                'device': self.predictor.device
            }
        
        else:
            return {'error': f'Unknown action: {action}'}
    
    def stop(self):
        """Stop the server"""
        logger.info("Stopping server...")
        self.running = False
        
        if self.server_socket:
            self.server_socket.close()
        
        logger.info("✅ Server stopped")
    
    def print_stats(self):
        """Print server statistics"""
        uptime = time.time() - self.start_time
        
        logger.info("\n" + "="*60)
        logger.info("📊 Server Statistics")
        logger.info("="*60)
        logger.info(f"   Uptime: {int(uptime)} seconds")
        logger.info(f"   Total Requests: {self.total_requests}")
        logger.info(f"   Total Predictions: {self.total_predictions}")
        
        if uptime > 0:
            logger.info(f"   Requests/sec: {self.total_requests / uptime:.2f}")
        
        logger.info("="*60)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron MQL5 Socket Server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=9999,
                       help='Server port (default: 9999)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Create and start server
    server = MQL5SocketServer(
        host=args.host,
        port=args.port,
        config_path=args.config
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Keyboard interrupt received")
        server.print_stats()
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
