# Quick Reference Guide - Natron Transformer V2

## 🚀 Common Commands

### Training
```bash
# Full training pipeline
python train.py --config config.yaml

# Skip pretraining
python train.py --config config.yaml --skip-pretrain

# Pretraining only
python train.py --config config.yaml --pretrain-only
```

### API Server
```bash
# Start Flask API
python api_server.py --model checkpoints/supervised/natron_v2.pt --port 5000

# Test API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

### MQL5 Bridge
```bash
# Start socket bridge
python mql5_bridge.py --model checkpoints/supervised/natron_v2.pt --port 8888
```

### Testing
```bash
# Test all components
python test_components.py
```

## 📊 Data Format

### Input CSV (`data/data_export.csv`)
```csv
time,open,high,low,close,volume
2024-01-01 00:00:00,1.0850,1.0860,1.0845,1.0855,1500
2024-01-01 00:15:00,1.0855,1.0865,1.0850,1.0860,1800
```

### API Request
```json
{
  "candles": [
    {
      "time": 1234567890,
      "open": 1.0850,
      "high": 1.0860,
      "low": 1.0845,
      "close": 1.0855,
      "volume": 1500
    }
  ]
}
```

### API Response
```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "direction": "Up",
  "regime": "BULL_WEAK",
  "confidence": 0.82
}
```

## 🔧 Configuration Keys

### Model Architecture
- `model.d_model`: Model dimension (default: 256)
- `model.nhead`: Attention heads (default: 8)
- `model.num_layers`: Encoder layers (default: 6)
- `model.dim_feedforward`: FFN dimension (default: 1024)

### Training
- `training.batch_size`: Batch size (default: 32)
- `training.learning_rate`: Learning rate (default: 1e-4)
- `training.num_epochs_pretrain`: Pretrain epochs (default: 50)
- `training.num_epochs_supervised`: Supervised epochs (default: 100)

### Data
- `data.sequence_length`: Candles per sample (default: 96)
- `data.train_split`: Training split (default: 0.7)
- `data.val_split`: Validation split (default: 0.15)
- `data.test_split`: Test split (default: 0.15)

### Labeling
- `labeling.buy_threshold`: Min conditions for BUY (default: 2)
- `labeling.sell_threshold`: Min conditions for SELL (default: 2)
- `labeling.balance_labels`: Enable label balancing (default: true)
- `labeling.neutral_buffer`: Buffer for neutral direction (default: 0.001)

## 📁 File Locations

- **Model checkpoints**: `checkpoints/supervised/natron_v2.pt`
- **Config**: `config.yaml`
- **Data**: `data/data_export.csv`
- **Logs**: `logs/`
- **MQL5 EA**: `mql5/NatronAI_EA.mq5`

## 🐛 Troubleshooting

### CUDA Out of Memory
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### Label Imbalance
Check label statistics during data preparation. Adjust thresholds:
```yaml
labeling:
  buy_threshold: 3  # Increase to reduce BUY signals
  sell_threshold: 3  # Increase to reduce SELL signals
```

### Model Not Loading
Ensure model path is correct and checkpoint format matches:
- Direct state_dict: `model.state_dict()`
- Checkpoint dict: `checkpoint['model_state_dict']`

## 📈 Performance Monitoring

### During Training
- Monitor loss curves
- Check label distribution output
- Watch validation loss for overfitting

### Model Evaluation
- Test on held-out test set
- Evaluate per-task metrics (buy/sell accuracy, direction accuracy, regime accuracy)
- Check prediction confidence distribution

## 🔐 Security Checklist

- [ ] API server not exposed to public internet
- [ ] MQL5 bridge uses localhost or VPN
- [ ] Model files secured (proprietary IP)
- [ ] API endpoints have authentication (if needed)

## 📝 Notes

- Minimum sequence length: 96 candles
- Minimum data required: ~200 candles (for feature calculation)
- GPU recommended for training (CPU inference possible but slow)
- Model size: ~2-5MB (depends on architecture)
