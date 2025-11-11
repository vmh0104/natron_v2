# 🧠 Natron Transformer V2

**Multi-Task Financial Trading Model with End-to-End GPU Pipeline**

A state-of-the-art deep learning system for financial market prediction, combining Transformer architecture with multi-task learning to predict Buy/Sell signals, directional movement, and market regime classification.

---

## 🎯 Overview

Natron Transformer is a sophisticated AI model designed for financial trading that:

- **Learns Market Structure**: Unsupervised pretraining discovers latent market patterns
- **Multi-Task Prediction**: Simultaneously predicts Buy, Sell, Direction, and Regime
- **Bias-Reduced Labeling**: Institutional-grade labeling with adaptive thresholds
- **GPU-Optimized**: PyTorch 2.x with mixed precision training for maximum performance
- **Production-Ready**: Flask API for real-time inference and MQL5 integration

### Model Architecture

```
Input: 96 OHLCV Candles → Feature Engine (100 features)
                              ↓
                    Feature Embedding (256D)
                              ↓
                    Positional Encoding
                              ↓
              Transformer Encoder (6 layers, 8 heads)
                              ↓
                    Attention Pooling
                              ↓
        ┌──────────┬──────────┬──────────┬──────────┐
        ↓          ↓          ↓          ↓          ↓
     Buy Head  Sell Head  Direction  Regime Head
     (Binary)   (Binary)   (3-class)  (6-class)
```

---

## 📊 Features

### Feature Engineering (~100 Indicators)

| Category | Count | Examples |
|----------|-------|----------|
| Moving Averages | 13 | MA, EMA, crossovers, slopes |
| Momentum | 13 | RSI, ROC, CCI, Stochastic, MACD |
| Volatility | 15 | ATR, Bollinger Bands, Keltner |
| Volume | 9 | OBV, VWAP, MFI, Volume ratio |
| Price Patterns | 8 | Doji, Gaps, Shadows, Body% |
| Returns | 8 | Log returns, intraday, cumulative |
| Trend Strength | 6 | ADX, +DI, -DI, Aroon |
| Statistical | 6 | Skewness, Kurtosis, Z-score, Hurst |
| Support/Resistance | 4 | Distance to High/Low 20-50 |
| Smart Money | 6 | Swing HL, BOS, CHoCH |
| Market Profile | 10 | POC, VAH, VAL, Entropy |

### Label Types

1. **Buy/Sell** (Binary): Multi-condition institutional signals
2. **Direction** (3-class): Up, Down, Neutral with bias reduction
3. **Regime** (6-class):
   - 0: BULL_STRONG
   - 1: BULL_WEAK
   - 2: RANGE
   - 3: BEAR_WEAK
   - 4: BEAR_STRONG
   - 5: VOLATILE

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd workspace

# Install dependencies
pip install -r requirements.txt

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Training

```bash
# Prepare your data
# Expected format: CSV with columns [time, open, high, low, close, volume]

# Start training (all phases)
python train.py --data data_export.csv --config config/config.yaml

# Skip pretraining (if you have pretrained model)
python train.py --data data_export.csv --load-pretrained model/pretrained_encoder.pt

# Skip RL phase (recommended for production)
python train.py --data data_export.csv --skip-rl
```

### Inference (API Server)

```bash
# Start Flask API server
python src/inference/api_server.py \
    --model model/natron_v2.pt \
    --config config/config.yaml \
    --scaler model/scaler.pkl \
    --host 0.0.0.0 \
    --port 5000

# Test prediction
curl -X POST http://localhost:5000/predict \
    -H 'Content-Type: application/json' \
    -d '{
        "candles": [
            {"time": "2024-01-01 00:00:00", "open": 100.0, 
             "high": 101.0, "low": 99.0, "close": 100.5, "volume": 5000},
            ... (96 candles total)
        ]
    }'

# Expected response
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

## 🏗️ Project Structure

```
workspace/
├── config/
│   └── config.yaml              # Configuration file
├── src/
│   ├── data/
│   │   ├── feature_engine.py    # Technical indicator generation
│   │   ├── label_generator.py   # Bias-reduced labeling
│   │   └── sequence_creator.py  # Sequence construction & DataLoaders
│   ├── models/
│   │   └── transformer.py       # Multi-task Transformer architecture
│   ├── training/
│   │   ├── pretrain.py          # Phase 1: Unsupervised pretraining
│   │   ├── supervised.py        # Phase 2: Supervised fine-tuning
│   │   └── rl_trainer.py        # Phase 3: RL (optional)
│   └── inference/
│       └── api_server.py        # Flask API for predictions
├── model/                       # Saved models and checkpoints
├── logs/                        # Training logs
├── train.py                     # Main training script
├── requirements.txt
└── README.md
```

---

## 🔥 Training Phases

### Phase 1: Pretraining (Unsupervised)

**Goal**: Learn latent market structure without labels

**Methods**:
- Masked Token Reconstruction (15% masking)
- Contrastive Learning (InfoNCE)

**Duration**: 50 epochs (~2-4 hours on GPU)

**Output**: Pretrained encoder with learned market representations

### Phase 2: Supervised Fine-Tuning

**Goal**: Predict Buy/Sell/Direction/Regime

**Loss Function**:
```python
Total Loss = w_buy × L_buy + w_sell × L_sell + 
             w_dir × L_direction + w_regime × L_regime
```

**Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)

**Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)

**Duration**: 100 epochs (~4-8 hours on GPU)

**Output**: Fully trained model ready for deployment

### Phase 3: Reinforcement Learning (Optional)

**Goal**: Optimize trading reward and stability

**Algorithm**: PPO (Proximal Policy Optimization)

**Reward Function**:
```
R = profit - α × turnover - β × drawdown
```

**Status**: Experimental (supervised model recommended for production)

---

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
data:
  sequence_length: 96          # Input sequence length
  train_split: 0.7             # Training data ratio
  val_split: 0.15              # Validation data ratio

model:
  d_model: 256                 # Model dimension
  nhead: 8                     # Attention heads
  num_encoder_layers: 6        # Transformer layers
  dim_feedforward: 1024        # FFN dimension
  dropout: 0.1

training:
  pretrain:
    enabled: true
    epochs: 50
    batch_size: 128
    lr: 1e-4
    mask_ratio: 0.15
    
  supervised:
    enabled: true
    epochs: 100
    batch_size: 64
    lr: 1e-4
    loss_weights:
      buy: 1.0
      sell: 1.0
      direction: 1.5
      regime: 1.2
```

---

## 📈 Performance Metrics

The model is evaluated on:

- **Accuracy**: Correct predictions / Total predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Loss**: Multi-task weighted cross-entropy

### Expected Performance (After Training)

| Task | Accuracy | F1 Score |
|------|----------|----------|
| Buy | 75-85% | 0.70-0.80 |
| Sell | 75-85% | 0.70-0.80 |
| Direction | 60-70% | 0.58-0.68 |
| Regime | 65-75% | 0.62-0.72 |

*Note: Performance varies based on data quality and market conditions*

---

## 🔌 MQL5 Integration

### Socket Communication

```mql5
// MQL5 EA → Python Server
string request = "{"
    "\"candles\": [" +
    "{\"time\":\"2024-01-01 00:00:00\"," +
    " \"open\":100.0,\"high\":101.0,\"low\":99.0," +
    " \"close\":100.5,\"volume\":5000}," +
    "... (96 candles)" +
    "]" +
"}";

// Send via socket/HTTP
string response = SendToPythonServer(request);

// Parse response
double buy_prob = ParseBuyProb(response);
double sell_prob = ParseSellProb(response);
string regime = ParseRegime(response);

// Execute trade logic
if(buy_prob > 0.7 && regime == "BULL_STRONG") {
    OpenBuyOrder();
}
```

**Latency Target**: <50ms per prediction

---

## 🛠️ System Requirements

### Minimum

- **OS**: Ubuntu 20.04+ / Debian 11+
- **Python**: 3.10+
- **RAM**: 16 GB
- **Storage**: 10 GB free
- **GPU**: NVIDIA GPU with 6GB VRAM (GTX 1060 or better)
- **CUDA**: 11.8+

### Recommended

- **RAM**: 32 GB
- **GPU**: NVIDIA RTX 3080 or better (12GB+ VRAM)
- **CUDA**: 12.1+
- **Storage**: SSD with 50GB free

### Cloud Deployment

- **GCP**: n1-standard-8 with 1x NVIDIA T4
- **AWS**: g4dn.2xlarge
- **Azure**: NC6s v3

---

## 📊 Data Requirements

### Input Format

CSV file with columns:

```
time,open,high,low,close,volume
2024-01-01 00:00:00,1.2345,1.2360,1.2340,1.2355,5000
2024-01-01 00:15:00,1.2355,1.2370,1.2350,1.2365,5200
...
```

### Minimum Data Size

- **Training**: 10,000+ candles
- **Recommended**: 50,000+ candles for robust training

### Timeframes

- M15 (15-minute)
- H1 (1-hour)
- H4 (4-hour)
- D1 (daily)

---

## 🔍 Monitoring & Logging

Training logs are saved to `logs/` directory:

```bash
# View training progress
tail -f logs/training.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# TensorBoard (if implemented)
tensorboard --logdir logs/
```

---

## 🚦 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cuda-toolkit-12-1

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", \
     "src.inference.api_server:app"]
```

### Systemd Service

```ini
[Unit]
Description=Natron API Server
After=network.target

[Service]
Type=simple
User=natron
WorkingDirectory=/opt/natron
ExecStart=/opt/natron/venv/bin/python src/inference/api_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size in config.yaml
batch_size: 32  # Instead of 64

# Or disable mixed precision
mixed_precision: false
```

### Slow Training

- Enable mixed precision training (AMP)
- Increase num_workers in DataLoader
- Use faster storage (NVMe SSD)
- Upgrade GPU

### Poor Model Performance

- Increase training data
- Adjust label thresholds in config
- Tune loss weights
- Add more training epochs

---

## 📚 References

### Papers

- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)

### Technical Indicators

- Investopedia Technical Analysis
- Trading View Indicator Reference
- Smart Money Concepts (ICT)

---

## 📝 License

Copyright © 2024 Natron AI. All rights reserved.

---

## 🤝 Support

For issues, questions, or contributions:

- **Issues**: GitHub Issues
- **Documentation**: Wiki
- **Community**: Discord/Telegram

---

## 🎯 Roadmap

### v2.1 (Q1 2025)
- [ ] Multi-timeframe analysis
- [ ] Sentiment analysis integration
- [ ] Enhanced regime detection

### v2.2 (Q2 2025)
- [ ] Real-time streaming inference
- [ ] Portfolio optimization
- [ ] Risk management module

### v3.0 (Q3 2025)
- [ ] Multi-asset support
- [ ] Distributed training
- [ ] Model compression (quantization)

---

**Built with ❤️ for the trading community**

*"Natron doesn't just predict Buy/Sell — it learns the grammar of the market."*
