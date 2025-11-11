# 🚀 Natron Transformer - Quick Start Guide

Get up and running with Natron in 5 minutes!

---

## Step 1: Installation (2 minutes)

```bash
# Clone and navigate
cd /workspace

# Install dependencies
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Expected output**: `CUDA: True`

---

## Step 2: Generate Sample Data (1 minute)

```bash
# Generate 10,000 candles of sample trading data
python scripts/generate_sample_data.py \
    --output data_export.csv \
    --candles 10000 \
    --timeframe 15min
```

**Output**: `data_export.csv` with synthetic OHLCV data

---

## Step 3: Train the Model (varies)

### Quick Training (Testing)

```bash
# Fast training with reduced epochs for testing
python train.py \
    --data data_export.csv \
    --config config/config.yaml
```

**Estimated time**:
- With GPU: 1-2 hours
- With CPU: 4-8 hours

### Production Training (Recommended)

Edit `config/config.yaml` first:

```yaml
training:
  pretrain:
    epochs: 50      # Increase for better results
  supervised:
    epochs: 100     # Increase for better results
```

Then train:

```bash
python train.py --data data_export.csv
```

**Estimated time**:
- With GPU: 4-8 hours
- With CPU: 12-24 hours

---

## Step 4: Start API Server (30 seconds)

```bash
# Start Flask server
python src/inference/api_server.py \
    --model model/natron_v2.pt \
    --config config/config.yaml \
    --scaler model/scaler.pkl
```

**Expected output**:
```
🚀 Starting Natron API Server on 0.0.0.0:5000
   Health check: http://0.0.0.0:5000/health
   Prediction:   http://0.0.0.0:5000/predict
```

---

## Step 5: Test Predictions (1 minute)

### Health Check

```bash
curl http://localhost:5000/health
```

**Expected**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### Make Prediction

Create `test_request.json`:

```json
{
  "candles": [
    {"time": "2024-01-01 00:00:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 5000},
    ... (add 95 more candles for total of 96)
  ]
}
```

Or use Python:

```python
import requests
import pandas as pd
import json

# Load your data
df = pd.read_csv('data_export.csv')

# Get last 96 candles
candles = df.tail(96).to_dict('records')

# Make request
response = requests.post(
    'http://localhost:5000/predict',
    json={'candles': candles}
)

print(response.json())
```

**Expected output**:
```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_probs": [0.15, 0.69, 0.16],
  "direction_pred": "up",
  "regime": "BULL_WEAK",
  "confidence": 0.82,
  "predictions": {
    "buy": 1,
    "sell": 0,
    "direction": 1,
    "regime": 1
  }
}
```

---

## Common Issues & Solutions

### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```yaml
# In config/config.yaml, reduce batch size
training:
  supervised:
    batch_size: 32  # Instead of 64
```

### 2. Import Errors

**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
```bash
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Data Format Error

**Error**: `KeyError: 'close'`

**Solution**: Ensure your CSV has columns: `time, open, high, low, close, volume`

### 4. API Not Starting

**Error**: `Address already in use`

**Solution**:
```bash
# Use different port
python src/inference/api_server.py --port 5001
```

---

## Next Steps

### 1. Evaluate Model

```bash
python scripts/evaluate_model.py \
    --model model/natron_v2.pt \
    --data data_export.csv \
    --output-dir evaluation
```

### 2. Fine-tune for Your Data

- Collect real market data
- Adjust label thresholds in `config/config.yaml`
- Retrain with more epochs

### 3. Deploy to Production

See `README.md` for:
- Docker deployment
- Systemd service setup
- MQL5 integration

### 4. Integrate with MQL5

Create MetaTrader Expert Advisor:

```mql5
// In your EA
string response = SendHTTPRequest(
    "http://your-server:5000/predict",
    candle_data_json
);

double buy_prob = ParseJSON(response, "buy_prob");
if(buy_prob > 0.7) {
    OrderSend(...); // Place buy order
}
```

---

## Performance Benchmarks

| Hardware | Training Time | Inference Time |
|----------|--------------|----------------|
| RTX 3080 | 4-6 hours | 20-30 ms |
| RTX 3060 | 6-8 hours | 30-40 ms |
| GTX 1660 | 10-12 hours | 50-60 ms |
| CPU (16 core) | 20-24 hours | 200-300 ms |

---

## Directory Structure After Training

```
workspace/
├── model/
│   ├── natron_v2.pt              # Final trained model
│   ├── natron_v2_best.pt         # Best validation model
│   ├── pretrained_encoder.pt     # Pretrained encoder
│   └── scaler.pkl                # Feature scaler
├── logs/
│   └── training.log              # Training logs
├── evaluation/
│   ├── buy_confusion_matrix.png
│   ├── sell_confusion_matrix.png
│   ├── direction_confusion_matrix.png
│   └── regime_confusion_matrix.png
└── data_export.csv               # Your training data
```

---

## Monitoring Training

### Real-time GPU Usage

```bash
watch -n 1 nvidia-smi
```

### Training Progress

```bash
# In another terminal
tail -f logs/training.log
```

### Check Model Size

```bash
ls -lh model/natron_v2.pt
```

Expected: ~50-200 MB depending on configuration

---

## Troubleshooting Training

### Training Too Slow?

1. Enable mixed precision: `mixed_precision: true` in config
2. Increase `num_workers` for DataLoader
3. Reduce sequence length or features (advanced)

### Model Not Learning?

1. Check label distribution (should be shown during training)
2. Increase learning rate slightly: `lr: 2e-4`
3. Reduce regularization: `weight_decay: 1e-6`
4. Train for more epochs

### Overfitting?

1. Increase dropout: `dropout: 0.2`
2. Increase weight decay: `weight_decay: 1e-4`
3. Use more training data
4. Enable label smoothing: `label_smoothing: 0.15`

---

## Support

- **Documentation**: See `README.md`
- **Configuration**: See `config/config.yaml` comments
- **Examples**: See `scripts/` directory

---

**You're all set! Happy trading! 🚀📈**
