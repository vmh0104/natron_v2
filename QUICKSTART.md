# 🚀 Natron Transformer - Quick Start Guide

Get up and running in 5 minutes!

---

## ⚡ Installation (1 minute)

```bash
# Clone repository
git clone <your-repo>
cd natron-transformer

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 📊 Step 1: Prepare Data (Optional)

If you have your own OHLCV data, place it in `data_export.csv`:

```csv
time,open,high,low,close,volume
2023-01-01 00:00:00,100.0,101.5,99.5,100.5,5000
2023-01-01 01:00:00,100.5,102.0,100.0,101.8,6500
...
```

**Note:** If no data file exists, the system will generate synthetic data automatically.

---

## 🎯 Step 2: Train Model (2 minutes setup, 6-12 hours training)

### Option A: Full Training Pipeline (Recommended)

```bash
python main.py --mode train
```

This runs:
1. **Phase 1:** Pretraining (unsupervised) - ~2-4 hours
2. **Phase 2:** Supervised fine-tuning - ~4-8 hours
3. Saves model to `model/natron_v2.pt`

### Option B: Quick Test (Skip Pretraining)

Edit `config.yaml`:
```yaml
pretrain:
  enabled: false  # Set to false
```

Then run:
```bash
python main.py --mode train
```

This skips Phase 1 and goes straight to supervised training (~4-8 hours).

---

## 🌐 Step 3: Start API Server (30 seconds)

Once training is complete:

```bash
python main.py --mode api
```

Server starts at `http://localhost:5000`

You should see:
```
🚀 NATRON INFERENCE API
======================================================================
🖥️  Using device: cuda
✅ Scaler loaded from model/scaler.pkl
✅ Model loaded successfully!
✅ Natron API initialized successfully!

🌐 Starting server at http://0.0.0.0:5000

📍 Endpoints:
   GET  /health       - Health check
   GET  /info         - Model information
   POST /predict      - Predict from JSON
   POST /predict_csv  - Predict from CSV file
```

---

## 🧪 Step 4: Test Predictions (30 seconds)

### Option A: Built-in Test

```bash
python main.py --mode test
```

### Option B: Example Script

```bash
python examples/example_usage.py
```

### Option C: Manual cURL

```bash
curl http://localhost:5000/health
```

---

## 📈 Example API Usage

### Python Client

```python
import requests
import pandas as pd

# Create 96-candle data
candles = [
    {
        "time": "2024-01-01 00:00:00",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000
    },
    # ... 95 more candles
]

# Make prediction
response = requests.post(
    'http://localhost:5000/predict',
    json={'candles': candles}
)

result = response.json()
print(f"Buy Probability: {result['buy_prob']:.2f}")
print(f"Direction: {result['direction']}")
print(f"Regime: {result['regime']}")
```

---

## 🎨 Visualization (TensorBoard)

Monitor training in real-time:

```bash
tensorboard --logdir logs/
```

Open browser: `http://localhost:6006`

---

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
# Quick adjustments for faster training
supervised:
  epochs: 50              # Reduce from 100
  batch_size: 32          # Reduce if GPU memory issues
  early_stopping_patience: 10

# For production
api:
  host: "0.0.0.0"
  port: 5000
```

---

## 📊 Expected Results

After training, you should see:

### Training Metrics
- **Buy/Sell Accuracy:** ~60-75%
- **Direction Accuracy:** ~50-65% (3-class)
- **Regime Accuracy:** ~40-60% (6-class)

### Inference Performance
- **Latency:** <50ms on GPU
- **Throughput:** ~100+ predictions/sec

---

## 🚨 Troubleshooting

### Issue: CUDA Out of Memory

**Solution:** Reduce batch size
```yaml
supervised:
  batch_size: 32  # or 16
```

### Issue: Training Too Slow

**Solution:** Reduce epochs or disable pretraining
```yaml
pretrain:
  enabled: false
supervised:
  epochs: 50
```

### Issue: API Won't Start

**Solution:** Check if model exists
```bash
ls -lh model/natron_v2.pt
```

If missing, run training first:
```bash
python main.py --mode supervised
```

### Issue: Low Accuracy

**Solutions:**
1. More training data (>5000 samples)
2. Longer training (more epochs)
3. Tune hyperparameters in config.yaml
4. Enable pretraining

---

## 📱 Next Steps

### 1. Integrate with MetaTrader 5

See `examples/mql5_integration_template.mq5` for MQL5 EA template.

### 2. Optimize Hyperparameters

Experiment with:
- Model size (d_model, num_layers)
- Learning rate
- Loss weights
- Label thresholds

### 3. Add Custom Features

Edit `src/feature_engine.py` to add your own indicators.

### 4. Deploy to Production

Options:
- Docker container
- Systemd service
- Cloud deployment (GCP, AWS)

See `ARCHITECTURE.md` for deployment details.

---

## 🎓 Learning Path

1. **Beginner:** Run with default config, test API
2. **Intermediate:** Tune hyperparameters, add features
3. **Advanced:** Implement Phase 3 RL, custom environments
4. **Expert:** Deploy to production, MQL5 integration

---

## 📚 Additional Resources

- **Full Documentation:** `README.md`
- **Architecture Details:** `ARCHITECTURE.md`
- **API Examples:** `examples/example_usage.py`
- **MQL5 Template:** `examples/mql5_integration_template.mq5`

---

## ✅ Checklist

- [ ] Installed dependencies
- [ ] Prepared data (or using synthetic)
- [ ] Trained model (both phases)
- [ ] Started API server
- [ ] Tested predictions
- [ ] Monitored with TensorBoard
- [ ] Reviewed metrics

---

## 💡 Pro Tips

1. **Start Small:** Use synthetic data first to test pipeline
2. **Monitor Training:** Keep TensorBoard open during training
3. **Save Checkpoints:** Models are saved every 10 epochs
4. **Validate Often:** Check validation metrics to avoid overfitting
5. **Document Changes:** Track config changes and results

---

## 🎉 You're Ready!

Your Natron Transformer is now trained and ready for inference.

**What's Next?**
- Fine-tune hyperparameters
- Integrate with your trading system
- Deploy to production

---

**Need Help?**
- Check `README.md` for detailed documentation
- Review `ARCHITECTURE.md` for technical details
- Open an issue on GitHub

---

*Happy Trading! 📈*

**Natron Transformer V2** - Institutional-Grade AI for Financial Markets
