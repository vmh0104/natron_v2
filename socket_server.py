"""
Natron Transformer - Socket Server for MT5 Integration
Low-latency real-time trading signals for MetaTrader 5
"""

import asyncio
import websockets
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict
import logging

from model import create_model
from feature_engine import FeatureEngine
from dataset import load_scaler
from config import load_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NatronSocketServer:
    """WebSocket server for MT5 integration"""
    
    def __init__(self, config, model_path: str, scaler_path: str):
        self.config = config
        self.device = config.device
        
        # Load model
        logger.info(f"📂 Loading model from {model_path}")
        self.model = create_model(config)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load scaler
        logger.info(f"📂 Loading scaler from {scaler_path}")
        self.scaler = load_scaler(scaler_path)
        
        # Feature engine
        self.feature_engine = FeatureEngine()
        
        # Regime names
        self.regime_names = {
            0: 'BULL_STRONG',
            1: 'BULL_WEAK',
            2: 'RANGE',
            3: 'BEAR_WEAK',
            4: 'BEAR_STRONG',
            5: 'VOLATILE'
        }
        
        logger.info("✅ Socket server initialized")
    
    async def handle_client(self, websocket, path):
        """Handle client connection"""
        client_id = id(websocket)
        logger.info(f"🔌 Client connected: {client_id}")
        
        try:
            async for message in websocket:
                response = await self.process_message(message)
                await websocket.send(json.dumps(response))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"❌ Error handling client {client_id}: {str(e)}")
            await websocket.send(json.dumps({
                'error': str(e),
                'status': 'error'
            }))
    
    async def process_message(self, message: str) -> Dict:
        """Process incoming message from MT5"""
        try:
            data = json.loads(message)
            
            if 'command' not in data:
                return {'error': 'Missing command', 'status': 'error'}
            
            command = data['command']
            
            if command == 'ping':
                return {'status': 'ok', 'message': 'pong'}
            
            elif command == 'predict':
                return await self.predict(data)
            
            elif command == 'info':
                return self.get_info()
            
            else:
                return {'error': f'Unknown command: {command}', 'status': 'error'}
        
        except json.JSONDecodeError:
            return {'error': 'Invalid JSON', 'status': 'error'}
        except Exception as e:
            logger.error(f"❌ Error processing message: {str(e)}")
            return {'error': str(e), 'status': 'error'}
    
    async def predict(self, data: Dict) -> Dict:
        """Run prediction"""
        import time
        start_time = time.time()
        
        try:
            if 'candles' not in data:
                return {'error': 'Missing candles', 'status': 'error'}
            
            candles = data['candles']
            
            if len(candles) != self.config.data.sequence_length:
                return {
                    'error': f'Expected {self.config.data.sequence_length} candles, got {len(candles)}',
                    'status': 'error'
                }
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            
            # Extract features
            features = self.feature_engine.compute_all_features(df)
            
            # Normalize
            features_normalized = self.scaler.transform(features.values)
            
            # Convert to tensor
            input_tensor = torch.from_numpy(features_normalized).float()
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
            
            # Parse outputs
            buy_probs = torch.softmax(outputs['buy'], dim=1)[0].cpu().numpy()
            sell_probs = torch.softmax(outputs['sell'], dim=1)[0].cpu().numpy()
            direction_probs = torch.softmax(outputs['direction'], dim=1)[0].cpu().numpy()
            regime_probs = torch.softmax(outputs['regime'], dim=1)[0].cpu().numpy()
            
            direction_pred = int(direction_probs.argmax())
            regime_pred = int(regime_probs.argmax())
            
            direction_map = {0: 'DOWN', 1: 'UP', 2: 'NEUTRAL'}
            
            # Calculate signal
            if buy_probs[1] > self.config.inference.confidence_threshold and buy_probs[1] > sell_probs[1]:
                signal = 'BUY'
            elif sell_probs[1] > self.config.inference.confidence_threshold and sell_probs[1] > buy_probs[1]:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            processing_time = (time.time() - start_time) * 1000
            
            response = {
                'status': 'ok',
                'signal': signal,
                'buy_prob': float(buy_probs[1]),
                'sell_prob': float(sell_probs[1]),
                'direction': direction_map[direction_pred],
                'direction_probs': {
                    'UP': float(direction_probs[1]),
                    'DOWN': float(direction_probs[0]),
                    'NEUTRAL': float(direction_probs[2])
                },
                'regime': self.regime_names[regime_pred],
                'regime_confidence': float(regime_probs[regime_pred]),
                'confidence': float(max(buy_probs[1], sell_probs[1])),
                'processing_time_ms': processing_time,
                'timestamp': time.time()
            }
            
            logger.info(f"✅ Prediction: {signal} (latency: {processing_time:.1f}ms)")
            
            return response
        
        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            return {'error': str(e), 'status': 'error'}
    
    def get_info(self) -> Dict:
        """Get server information"""
        from model import count_parameters
        
        return {
            'status': 'ok',
            'model_name': 'Natron Transformer v2',
            'parameters': count_parameters(self.model),
            'device': str(self.device),
            'sequence_length': self.config.data.sequence_length,
            'tasks': ['buy', 'sell', 'direction', 'regime']
        }
    
    async def start(self):
        """Start WebSocket server"""
        host = self.config.inference.socket_host
        port = self.config.inference.socket_port
        
        logger.info(f"\n🔌 Starting Natron Socket Server")
        logger.info(f"   Host: {host}")
        logger.info(f"   Port: {port}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Max Latency Target: {self.config.inference.max_latency_ms}ms")
        
        async with websockets.serve(self.handle_client, host, port):
            logger.info(f"✅ Server listening on ws://{host}:{port}")
            await asyncio.Future()  # Run forever


async def main():
    """Main entry point"""
    import sys
    import os
    
    # Load config
    config = load_config()
    
    # Get model and scaler paths
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = config.inference.model_path
    
    if len(sys.argv) > 2:
        scaler_path = sys.argv[2]
    else:
        scaler_path = os.path.join(config.output_dir, 'scaler.pkl')
    
    # Create and start server
    server = NatronSocketServer(config, model_path, scaler_path)
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
