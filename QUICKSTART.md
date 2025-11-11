# 🚀 Natron Transformer - Quick Start Guide

Get started with Natron in 5 minutes!

---

## ⚡ Installation (1 minute)

```bash
# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, 8GB RAM, CUDA optional (for GPU)

---

## 🎯 Quick Demo (2 minutes)

Run the automated demo:

```bash
./run_demo.sh
```

This will:
1. Create virtual environment
2. Install dependencies
3. Train model (quick mode)
4. Start API server
5. Test prediction

---

## 📊 Full Training Pipeline (30-60 minutes)

### Step 1: Prepare Data

Place your OHLCV CSV in `data/data_export.csv`:

```csv
time,open,high,low,close,volume
2024-01-01 00:00:00,100.0,101.5,99.5,101.0,1200
2024-01-01 00:15:00,101.0,102.0,100.5,101.8,1350
...
```

Or use auto-generated sample data (runs automatically if no data found).

### Step 2: Train Model

```bash
# Full training (pretrain + supervised + RL)
python main.py

# Quick training (supervised only) - 10-15 min
python main.py --skip-pretrain --skip-rl

# Custom config
python main.py --config my_config.yaml
```

Training time:
- Phase 1 (Pretrain): ~2-4 hours
- Phase 2 (Supervised): ~4-8 hours  
- Phase 3 (RL): ~2-4 hours
- **Quick mode**: ~10-15 minutes (skip pretrain & RL)

### Step 3: Start API Server

```bash
python src/api_server.py
```

Server runs on `http://localhost:5000`

---

## 🧪 Test Predictions

### Option 1: Python Script

```bash
python test_api.py
```

This will:
- Test health endpoint
- Make sample predictions
- Benchmark latency

### Option 2: cURL

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

### Option 3: Python Code

```python
import requests
import json

# Generate or load your 96 OHLCV candles
data = [
    {"time": "2024-01-01 00:00:00", "open": 100.0, 
     "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000},
    # ... 95 more candles
]

# Make prediction
response = requests.post(
    "http://localhost:5000/predict",
    json={"data": data}
)

result = response.json()
print(f"Buy: {result['buy_prob']:.2f}")
print(f"Sell: {result['sell_prob']:.2f}")
print(f"Regime: {result['regime']}")
```

---

## 📈 Expected Output

```json
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
```

---

## 🎯 Trading Signals

**Buy Signal:**
- `buy_prob > 0.6` → Strong buy
- `buy_prob > 0.5` → Weak buy

**Sell Signal:**
- `sell_prob > 0.6` → Strong sell
- `sell_prob > 0.5` → Weak sell

**Direction:**
- `direction_up > 0.6` → Likely upward movement
- `direction_down > 0.6` → Likely downward movement

**Regime:**
- `BULL_STRONG` / `BULL_WEAK` → Bullish market
- `BEAR_STRONG` / `BEAR_WEAK` → Bearish market
- `RANGE` → Sideways market
- `VOLATILE` → High volatility

---

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
# Model size
model:
  d_model: 256        # Hidden dimension
  nhead: 8            # Attention heads
  num_encoder_layers: 6

# Training
supervised:
  epochs: 100
  learning_rate: 5.0e-5
  batch_size: 64

# Data
data:
  sequence_length: 96  # Number of candles
  test_split: 0.2
```

---

## 🔧 Troubleshooting

### CUDA not available

```
⚠️ CUDA not available, falling back to CPU
```

✅ This is normal if no GPU. Training will be slower but still works.

### Data not found

```
❌ Data file not found: data/data_export.csv
```

✅ System auto-generates sample data. Replace with real data for production.

### Memory error

```
RuntimeError: CUDA out of memory
```

✅ Reduce batch size in `config/config.yaml`:
```yaml
supervised:
  batch_size: 32  # Reduce from 64
```

### Port already in use

```
OSError: [Errno 48] Address already in use
```

✅ Change port in `config/config.yaml`:
```yaml
api:
  port: 5001  # Change from 5000
```

---

## 📚 Next Steps

1. **Read full documentation:** `README.md`
2. **Customize features:** Edit `src/feature_engine.py`
3. **Tune hyperparameters:** Edit `config/config.yaml`
4. **Backtest results:** Implement backtesting module
5. **Deploy to production:** Use Docker or systemd

---

## 🔗 Useful Commands

```bash
# Training
python main.py                          # Full pipeline
python main.py --skip-pretrain          # Skip pretraining
python main.py --skip-rl                # Skip RL
python main.py --data path/to/data.csv  # Custom data

# API
python src/api_server.py                # Start server
python test_api.py                      # Test API

# Testing modules
python src/feature_engine.py            # Test features
python src/label_generator.py           # Test labels
python src/model.py                     # Test model
```

---

## 💡 Tips

1. **Start with quick training** (`--skip-pretrain --skip-rl`) to verify setup
2. **Use GPU** for 10-20× speedup
3. **Collect more data** (5000+ candles) for better results
4. **Tune thresholds** based on your risk tolerance
5. **Backtest thoroughly** before live trading

---

## ⚠️ Important Notes

- **Paper trade first** - Never use untested models live
- **Monitor performance** - Models degrade over time
- **Risk management** - Always use stop losses
- **Market conditions** - Models may fail in extreme volatility

---

## 🎉 You're Ready!

Your Natron Transformer is now set up and ready for algorithmic trading.

**Next:** Read the full `README.md` for advanced features and deployment options.

---

**Questions?** Check the documentation or open an issue on GitHub.

**Happy Trading! 📈**
